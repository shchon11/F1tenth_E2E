"""A trained v1 policy as the DAgger teacher of a v2 student (2026-10-22).

Why: the v2 student races slowly because it follows instead of passing. In the selection races (Korea 2026, rep floor)
v2KF_s2341 u2048 drove 3.42 m/s with 1.33 passes/km; the v1 Korea pick drove 4.45 m/s with 2.72 passes/km. Without
props v2 still drove 3.44 m/s at 0.45 passes/km. Its PPO variants (coarse / fine update granularity, the lagrange
multiplier adaptive or held) bought safety, never passing. This teacher is the policy that passes: it drives every row
off its OWN observation contract with its own recurrent state and its dial held. Its linear plan, refitted to the
student's wp4 (`train_v2.linear_to_wp4`), is the label (`dagger_v2.Wp4Labels`, `--wp4-label policy`).

The one thing a league seat's builder (`seat_obs.SeatObsV1`) does not do right for a teacher: it records each row's
EXECUTED action as the previous action, and on a student-driven row that is a wp4 plan the v1 policy never issued.
`TeacherSeatObsV1` records the teacher's own last plan on every row -- the history it would have had, driving -- and
starts a fresh episode from "hold this speed" in the linear layout, as a v1 env does.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

from ..mpc import PlanSpec, _LAYOUTS, contract_of, contract_spec
from . import conditioning as cond_mod
from .memory import runtime_for
from .model import load_checkpoint
from .obs import flatten_obs, spec_from_dict
from .seat_obs import SeatObsV1, adopt


class TeacherSeatObsV1(SeatObsV1):
    """`SeatObsV1` whose action history is the teacher's own plans, not the executed ones."""

    def __init__(self, spec, env, teacher: "PolicyTeacher"):
        super().__init__(spec, env)
        self.teacher = teacher

    def reset_actions(self, ids: torch.Tensor, prev_action: torch.Tensor) -> None:
        """A fresh episode starts from "hold this speed" in the LINEAR layout: the env's `prev_action` holds it in the
        layout of whoever drives the row (wp4 for the student), which the v1 policy would misread."""
        v = self.env.sim.odom.state[ids, 3]                       # the spawn speed `_reset_envs_core` just set
        held = torch.zeros(ids.numel(), self.spec.act_dim, device=self.act_hist.device, dtype=self.act_hist.dtype)
        held[:, _LAYOUTS["linear"].speed] = (v / float(self.env.ecfg.v_max_policy) * 2 - 1).to(held.dtype)[:, None]
        self.act_hist[ids] = held[:, None, :]
        self.teacher.reset_rows(ids, held)

    def advance(self, r, a: torch.Tensor) -> None:
        super().advance(r, self.teacher.last_action.to(a.dtype))


class PolicyTeacher:
    """A v1 checkpoint driving every row of `env` off its own observation, deterministically, its dial held.

    `act()` is called once per env step, before the step: it advances the recurrent state and remembers the plan, which
    the builder then records as the previous action. `lin` is the plan format its actions are read in."""

    def __init__(self, env, path: str, device, dial: Optional[float] = None):
        model, extra = load_checkpoint(path, device, allow_conditional=True)
        if (model.meta or {}).get("arch") == "v2":
            raise ValueError(f"{os.path.basename(path)} is a v2 checkpoint; the policy teacher is a v1 policy")
        if not (extra or {}).get("spec"):
            raise ValueError(f"{os.path.basename(path)} records no observation spec (extra['spec'])")
        self.path, self.model, self.extra, self.env = str(path), model, extra, env
        self.lin = contract_spec(contract_of(extra), base=PlanSpec()) or PlanSpec()
        if self.lin.speed_mode != "linear":
            raise ValueError(f"{os.path.basename(path)} plans in {self.lin.speed_mode!r}; the refit reads linear plans")
        spec = spec_from_dict(extra["spec"])
        if int(spec.act_dim) != int(model.meta.get("act_dim", spec.act_dim)):
            raise ValueError(f"{os.path.basename(path)}: its network's actions are not its spec's")
        self.runtime = runtime_for(model, env.B, device)
        self.hidden = None
        cdim = int((model.meta.get("cond") or {}).get("dim", 0) or 0)
        held = dial if dial is not None else (extra.get("experiment") or {}).get("dial_constant")
        if cdim and held is None:
            raise ValueError(f"{os.path.basename(path)} reads a dial and none is held: pass dial=")
        self.dial = None if not cdim else float(held)
        self.cspec = cond_mod.CondSpec.from_meta(model.meta["cond"]) if cdim else None
        self.last_action = torch.zeros(env.B, int(spec.act_dim), device=env.device)
        self.seat = TeacherSeatObsV1(spec, env, self)
        env.extra_seat_builders = tuple(getattr(env, "extra_seat_builders", ())) + (self.seat,)
        adopt(self.seat, env)                                    # a no-op before the env's first reset

    def describe(self) -> str:
        return f"policy {os.path.basename(self.path)} (v1, dial {self.dial}, its own observation)"

    def reset_rows(self, ids: torch.Tensor, held: torch.Tensor) -> None:
        self.last_action[ids] = held.to(self.last_action.dtype)
        if ids.numel() == 0:
            return
        done = torch.zeros(self.env.B, dtype=torch.bool, device=self.env.device)
        done[ids] = True
        if self.runtime.stateful:
            self.runtime.reset(done)
        if self.hidden is not None:
            self.hidden = self.hidden.reset(done)

    @torch.no_grad()
    def act(self) -> torch.Tensor:
        """(B, act_dim) normalised linear plan of every row, from the observation of the latest step."""
        obs = self.seat.obs
        if obs is None:
            raise RuntimeError("the policy teacher has not seen a step: reset the env first")
        scan, proprio = flatten_obs(obs)
        c = None
        if self.dial is not None:
            mu = torch.full((scan.shape[0],), self.dial, device=scan.device)
            c = cond_mod.mu_to_c(mu, self.cspec).to(scan.dtype)
        action, _logp, self.hidden = self.model.act(self.runtime.observe(scan, proprio), proprio, deterministic=True,
                                                    c=c, h=self.hidden)
        self.last_action = action.to(self.last_action.dtype).clone()
        return self.last_action
