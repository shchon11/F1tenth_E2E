"""The `tcs` controller arm: the car's own `TractionGuard`, running inside the simulator loop.

One implementation, three callers. `f1sim_ros/f1sim_ros/traction.py` is the module that ships on the
car; `scripts/replay_traction.py` replays it over the recorded bags; this runs the same class,
unmodified, on every simulated car, so what is trained against is what drives.

Where it sits
-------------
Between the policy (or the plan tracker) and the VESC, exactly as `policy_node` puts it: the
per-step speed command computed in `gym_env.step` is handed to `guard.shape` before it reaches
`Simulator.step`. An externally driven car (`env.set_external_cmd`, i.e. teleop or the ROS link)
bypasses it, the same way the mux output on the car is not the node's to shape.

What it is fed
--------------
The simulated sensors, and only those -- no state, no `P`:

* wheel speed and its timestamp from `StepResult.odom[:, 3]` / `StepResult.odom_t`, i.e. the
  ERPM-quantised, jitter-stamped channel `odom.py` now produces;
* body longitudinal acceleration from the last IMU sample of the step, `StepResult.imu[:, -1, 3]`,
  in m/s^2 (the emulator's units; the car publishes g and `policy_node` scales it);
* motor current from `StepResult.motor_current`, the optional third input, so the spin gate is
  configured here the way it is on the car rather than disabled.

Those all come from the *previous* control step, because that is the most recent sample a command
being issued now could have been computed from. Feeding this step's odometry would hand the guard
the consequence of the command it is about to shape.

Needs `vehicle.wheel_model`. With the switch off the wheel speed IS the body speed, the residual is
identically zero, and this arm would be a `legacy` run wearing another arm's name -- so it refuses.

Cost
----
`TractionGuard` is host Python over small scalars: B `update` + `shape` pairs per control step, and
one device->host transfer to fetch the four inputs plus one host->device transfer to return the
shaped speed. It is therefore **outside** the CUDA-graph fastpath by construction -- nothing here is
in `sim._roll`, which is the only thing `viewer/graph_fastpath.py` captures -- but the two transfers
are a synchronise per step, which on the graphs backend is the cost that matters rather than the
Python. Measured numbers are in `docs/research/wheel-model-2026-09-13.md`.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from f1sim_ros.traction import LOCK, SPIN, TractionGuard, TractionParams


class TractionArm:
    """B independent `TractionGuard`s installed on one environment.

    `install()` puts `shape` on `env.cmd_shaper` and `release()` takes it off again; the arm owns
    nothing else about the environment. `reset(mask)` forgets the episodes that just ended -- a new
    episode is a different car on a different surface, and carrying a latched release across it
    would release a brake that belongs to a car that no longer exists.
    """

    def __init__(self, env, params: Optional[TractionParams] = None, device=None):
        sim = env.sim
        if not getattr(sim, "wheel_model", False):
            raise RuntimeError(
                "the tcs arm needs vehicle.wheel_model: with it off the simulated wheel speed IS "
                "the body speed, so the guard's residual is identically zero and it can never "
                "fire. Turn the wheel model on or drop the arm -- running it here would record a "
                "legacy run under another arm's name.")
        self.env = env
        self.B = int(env.B)
        self.device = torch.device(device or env.device)
        self.params = (params or TractionParams()).validate()
        self.guards = [TractionGuard(self.params) for _ in range(self.B)]
        self._prev = None                       # the callable `cmd_shaper` held before install
        self._installed = False
        #: Counters, read once per update by `metrics()`. Cheap: Python ints, no device traffic.
        self.locks = 0
        self.spins = 0
        self.active_steps = 0
        self.steps = 0
        self.d_up = 0.0                         # largest release this arm has made [m/s]
        self.d_dn = 0.0                         # largest cap [m/s]

    # -- lifecycle ---------------------------------------------------------------
    def install(self) -> "TractionArm":
        if self._installed:
            raise RuntimeError("already installed")
        if getattr(self.env, "cmd_shaper", None) is not None:
            raise RuntimeError(f"env.cmd_shaper is already {self.env.cmd_shaper!r}; two shapers on "
                               f"one command path is not something to resolve by ordering")
        self._prev = getattr(self.env, "cmd_shaper", None)
        self.env.cmd_shaper = self.shape
        self._installed = True
        return self

    def release(self) -> None:
        if self._installed:
            self.env.cmd_shaper = self._prev
            self._installed = False

    def reset(self, mask: Optional[torch.Tensor]) -> None:
        """Forget the guards of the envs whose episode just ended."""
        if mask is None:
            return
        for i in torch.nonzero(mask.reshape(-1)).flatten().tolist():
            self.guards[i].reset()

    # -- per step ----------------------------------------------------------------
    @torch.no_grad()
    def shape(self, cmd: torch.Tensor) -> torch.Tensor:
        """Shape the (B, 2) command. Called from `gym_env.step`, before the simulator sees it."""
        r = self.env.last_result
        if r is None or r.odom_t is None:
            return cmd                          # before the first step there is nothing to detect on
        ax = (r.imu[:, -1, 3] if (r.imu is not None and r.imu.shape[1] > 0)
              else torch.full((self.B,), float("nan"), device=cmd.device))
        cur = (r.motor_current if r.motor_current is not None
               else torch.full((self.B,), float("nan"), device=cmd.device))
        # One transfer down, one up. Reading these four per env instead would be B synchronises a
        # step, which is the whole cost of the arm and none of its value.
        rows = torch.stack([r.odom_t, r.odom[:, 3], ax, cur, cmd[:, 1]], 1).cpu().tolist()
        out = []
        n_active = 0
        for i, (t, v, a, c, v_cmd) in enumerate(rows):
            g = self.guards[i]
            st = g.update(t, v, None if math.isnan(a) else a, None if math.isnan(c) else c)
            if st.changed:
                self.locks += st.state == LOCK
                self.spins += st.state == SPIN
            n_active += st.active
            shaped = g.shape(v_cmd)
            d = shaped - v_cmd
            if d > self.d_up:
                self.d_up = d
            if d < self.d_dn:
                self.d_dn = d
            out.append(shaped)
        self.steps += 1
        self.active_steps += n_active
        v_new = torch.tensor(out, dtype=cmd.dtype, device=cmd.device)
        return torch.stack([cmd[:, 0], v_new], 1)

    # -- metrics -----------------------------------------------------------------
    def metrics(self, prefix="controller/") -> dict:
        """Drain the arm's counters. In env-transitions, like the rest of `grip_runtime`."""
        if self.steps == 0:
            return {}
        n = float(self.steps * self.B)
        d = {f"{prefix}tcs_active_frac": self.active_steps / n,
             f"{prefix}tcs_locks": float(self.locks),
             f"{prefix}tcs_spins": float(self.spins),
             f"{prefix}tcs_release_max": self.d_up,
             f"{prefix}tcs_cap_max": self.d_dn}
        self.locks = self.spins = self.active_steps = self.steps = 0
        self.d_up = self.d_dn = 0.0
        return d

    def meta(self) -> dict:
        """What a checkpoint has to carry to reproduce this arm: every threshold it ran under."""
        return {"params": {k: getattr(self.params, k)
                           for k in TractionParams.__dataclass_fields__}}
