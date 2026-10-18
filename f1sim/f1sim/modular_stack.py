"""The modular baselines' own view of the race: localisation + perception, no ground truth (2026-10-18).

The user: "모듈형은 왜 상대 위치랑 자기 위치를 gt로 써? 제정신이여? Localization Perception
perdection은 개나줬냐?" Until this module the planners driving the ego car (raceline pure pursuit,
ForzaETH's spliner and predictive spliner, the lane switcher) were handed the simulator's state:
their own pose and speed exactly, every other car's position and speed exactly, every crate exactly,
and this episode's actuator latency, servo lag and calibration errors. None of that exists on a car.

`ModularStack` is what a car has instead, per ego row:

* its pose from the particle filter (`localization.ParticleFilter`) on its own odometry and scan;
* its speed and yaw rate from that odometry (the VESC's wheel speed, the gyro);
* the obstacles -- cars and crates alike, as upstream's detector reports obstacles -- from detection
  and tracking on its own scan (`perception.Perception`);
* the NOMINAL actuator model (`ActuatorParams` defaults: 15 ms command delay, 80 ms servo, no trim
  or gain error), which is what a stack is calibrated to, not this episode's draw.

What it keeps is the friction: the speed profile is still chosen for the floor's grip, because our
policy is told the same thing through its grip dial (an operator setting, `evaluate` feeds it the
true value). Same information on both sides.

The planner reads all of it through `observer` (`FrenetOpponentPlanner._opponents` and friends), and
the ego's trailing gap is held behind the nearest moving track rather than the true car ahead.
"""
from __future__ import annotations

from typing import Optional

import torch

from .localization import ParticleFilter, PFParams
from .perception import DetectParams, Perception, TrackParams

#: What one step rebinds, per object. Each is kept in a fixed buffer and written back in place, so the
#: whole step can be one CUDA graph: a graph replays addresses, and state rebound to a new tensor would
#: be read from the old one on every replay.
_PF_STATE = ("particles", "weights", "estimate", "_odom_prev", "_ready")
_PER_STATE = ("x", "P", "active", "hits", "misses", "ext", "det_xy", "det_valid", "det_ext")
#: Eager steps before the capture (kernels compiled, allocator warm).
_EAGER_BEFORE_CAPTURE = 3


class ModularStack:
    def __init__(self, env, rows: torch.Tensor, base, *, pf: Optional[PFParams] = None,
                 det: Optional[DetectParams] = None, trk: Optional[TrackParams] = None):
        self.env = env
        self.base = base
        dev = env.sim.state.device
        self.rows = rows.to(dev).long()
        lid = env.cfg.lidar
        mount = (float(lid.mount_x), float(lid.mount_y), float(lid.mount_z))
        self.pf = ParticleFilter(env.sim, self.rows, pf, mount=mount)
        self.per = Perception(env.sim, base, self.rows, det, trk, mount=mount)
        B = env.sim.state.shape[0]
        T = self.per.T
        # Whole-batch views the planner indexes by env row (only `rows` are ever filled).
        self.obs_s = torch.zeros(B, T, device=dev)
        self.obs_d = torch.zeros(B, T, device=dev)
        self.obs_vs = torch.zeros(B, T, device=dev)
        self.obs_ok = torch.zeros(B, T, dtype=torch.bool, device=dev)
        self.obs_xy = torch.zeros(B, T, 2, device=dev)
        self.obs_vd = torch.zeros(B, T, device=dev)
        #: half extents behind / ahead / left / right of each track's centre [m] (`Perception.ext`)
        self.obs_ext = torch.zeros(B, T, 4, device=dev)
        self.est_state = env.sim.state.clone()
        self._nearest = torch.zeros(B, dtype=torch.long, device=dev)
        act = env.cfg.actuator
        self._nominal = {"cmd_delay": float(act.cmd_delay), "servo_tau": float(act.servo_tau),
                         "steer_bias": float(act.steer_bias), "steer_gain": float(act.steer_gain),
                         "speed_gain": float(act.speed_gain)}
        # One step's inputs, in fixed buffers (the graph's), and the graph once captured.
        R = self.rows.numel()
        self._in_odom = torch.zeros(R, 3, device=dev)
        self._in_scan = torch.zeros(R, int(lid.n_beams), device=dev)
        self._in_pose = torch.zeros(R, 3, device=dev)
        self._in_fresh = torch.zeros(R, dtype=torch.bool, device=dev)
        self._in_tid = torch.zeros(R, dtype=torch.long, device=dev)
        self._bufs = None
        self._graph = None
        self._eager_steps = 0
        #: A CUDA graph of the step (localisation + perception) after a few eager steps: the step is a
        #: few hundred small kernels, ~16 ms launched one by one for one car, which kept the console's
        #: modular baselines below real time (2026-10-19). Off on the CPU, with F1SIM_STACK_GRAPH=0, or
        #: after a capture fails (then the step stays eager, and says so in `graph_note`).
        import os as _os
        self.use_graph = dev.type == "cuda" and _os.environ.get("F1SIM_STACK_GRAPH", "1") != "0"
        self.graph_note = ""

    # ------------------------------------------------------------------ one control step
    @torch.no_grad()
    def update(self, env) -> torch.Tensor:
        """Advance localisation and perception on this step's sensors; returns the estimated state
        (B, 8): the stack's rows replaced by what the car believes, the rest untouched (and unread)."""
        lr = env.last_result
        rows = self.rows
        # A new episode: the operator puts the car on the grid and gives the filter its pose (`fresh`,
        # masked every step rather than behind an `any()` that would synchronise with the GPU).
        self._in_odom.copy_(lr.odom[rows, :3])
        self._in_scan.copy_(lr.scan[rows])
        self._in_pose.copy_(env.sim.state[rows, :3])
        self._in_fresh.copy_(env.ep_step[rows] == 0)
        self._in_tid.copy_(env.sim.tid[rows])
        self._advance()
        st = env.sim.state.clone()
        est = torch.zeros(rows.numel(), st.shape[1], device=st.device, dtype=st.dtype)
        est[:, :3] = self.pf.estimate
        est[:, 3] = lr.odom[rows, 3]                    # wheel speed
        est[:, 5] = lr.odom[rows, 4]                    # gyro yaw rate
        st[rows] = est
        self.est_state = st
        return st

    def _advance(self) -> None:
        """One step on what the input buffers hold: replayed once captured, eager before."""
        if self._graph is not None:
            self._graph.replay()
            return
        self._core()
        self._eager_steps += 1
        if self.use_graph and self._eager_steps >= _EAGER_BEFORE_CAPTURE \
                and not torch.cuda.is_current_stream_capturing():
            self._capture()

    def _core(self) -> None:
        """One step of localisation and perception on the input buffers, every piece of state written
        back into its own buffer -- eager, or replayed as one CUDA graph (`_capture`)."""
        self.pf.reset(self._in_fresh, self._in_pose, self._in_odom)
        pose = self.pf.step(self._in_odom, self._in_scan, self._in_tid)
        self.per.reset(self._in_fresh)
        self.per.step(pose, self._in_scan, self._in_tid)
        self._persist()
        rows, tid = self.rows, self._in_tid
        ok = self.per.confirmed
        self.obs_ok[rows] = ok
        self.obs_s[rows] = self.per.x[..., 0]
        self.obs_d[rows] = self.per.x[..., 1]
        self.obs_vs[rows] = self.per.x[..., 2]
        self.obs_vd[rows] = self.per.x[..., 3]
        self.obs_ext[rows] = self.per.ext
        self.obs_xy[rows] = self.per.to_world(self.per.x[..., 0], self.per.x[..., 1], tid[:, None].expand_as(ok))

    def _persist(self) -> None:
        if self._bufs is None:
            self._bufs = {(id(o), n): getattr(o, n).clone() for o, names in ((self.pf, _PF_STATE), (self.per, _PER_STATE))
                          for n in names}
        for o, names in ((self.pf, _PF_STATE), (self.per, _PER_STATE)):
            for n in names:
                buf, v = self._bufs[(id(o), n)], getattr(o, n)
                if v is not buf:
                    buf.copy_(v)
                    setattr(o, n, buf)

    def _capture(self) -> None:
        """Record `_core` as one CUDA graph and replay it from now on. Warmed up on a side stream with
        the state saved and put back, so the capture does not advance the filter; captured on that
        stream, named, on this device (`graph_fastpath.GraphedCallable` says why it must be named)."""
        dev = self.rows.device
        saved = {k: v.clone() for k, v in self._bufs.items()}
        # The warm-up draws random numbers; put the generator back so the first replay draws what the
        # next eager step would have (the capture itself does not advance it). Same as
        # `GraphedCallable`, and what makes the graph drive exactly like the eager stack.
        rng = torch.cuda.get_rng_state(dev)
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        try:
            with torch.cuda.stream(side):
                for _ in range(2):
                    self._core()
            torch.cuda.current_stream(dev).wait_stream(side)
            torch.cuda.synchronize(dev)
            for k, v in saved.items():
                self._bufs[k].copy_(v)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=side):
                self._core()
        except Exception as exc:                       # noqa: BLE001 -- stays eager, said once
            self.use_graph = False
            self.graph_note = f"stack graph off: {type(exc).__name__}: {exc}"
            for k, v in saved.items():
                self._bufs[k].copy_(v)
            try:
                torch.cuda.set_rng_state(rng, dev)
            except RuntimeError:                       # a failed capture can leave it unsettable
                pass
            return
        # This call's step already ran eagerly (`update`) and its state is what `saved` holds -- the
        # capture recorded the work without running it -- so the graph takes over from the next step.
        for k, v in saved.items():
            self._bufs[k].copy_(v)
        torch.cuda.set_rng_state(rng, dev)
        self._graph = g

    def nominal_P(self, env) -> dict:
        """The simulator's parameter dict with the actuator model at its calibrated nominal values;
        the friction (the grip dial's information) stays."""
        P = dict(env.sim.P)
        B = env.sim.state.shape[0]
        for k, v in self._nominal.items():
            if k in P:
                P[k] = torch.full((B,), v, device=env.sim.state.device, dtype=P[k].dtype)
        return P

    # ------------------------------------------------------------------ the planner's observer
    def _gaps(self, planner, state, tid):
        """(B, T) signed arc from each car's estimated station to each track (+ ahead), and the
        car's raceline index."""
        b = planner.base
        idx = b.project(state[:, :2], tid)[0]
        s_e = idx.to(state.dtype) * b.ds[tid]
        L = b.length[tid]
        g = torch.remainder(self.obs_s - s_e[:, None] + 0.5 * L[:, None], L[:, None]) - 0.5 * L[:, None]
        return torch.where(self.obs_ok, g, torch.full_like(g, float("inf"))), idx

    def opponents(self, planner, state, tid, rear):
        """`FrenetOpponentPlanner._opponents` from the tracks: the nearest obstacle not behind `rear`."""
        B = state.shape[0]
        dev, dt = state.device, state.dtype
        tid_b = torch.zeros(B, dtype=torch.long, device=dev) if tid is None else tid
        gaps, _ = self._gaps(planner, state, tid_b)
        rank = torch.where(gaps > rear[:, None], gaps, torch.full_like(gaps, float("inf")))
        j = rank.argmin(1)
        ar = torch.arange(B, device=dev)
        g = torch.where(torch.isfinite(rank[ar, j]), gaps[ar, j], torch.full((B,), float("inf"), device=dev, dtype=dt))
        b = planner.base
        idx_opp = torch.remainder((self.obs_s[ar, j] / b.ds[tid_b]).round().long(), b.N)
        ego_d = planner._lateral(state[:, :2], tid_b)[0]
        self._nearest = j
        return g, self.obs_d[ar, j].to(dt), idx_opp, tid_b, ego_d

    def obstacle_list(self, planner, state, tid):
        """Every track of each row's car as upstream's `ObstacleArray` carries it, relative to the car:
        (gap to the centre [m, + ahead], d centre, vs, vd, half extents behind / ahead / left / right,
        valid), each (B, T). What `SplinerTeacher` in upstream mode plans from."""
        gaps, _ = self._gaps(planner, state, tid)
        ok = self.obs_ok & torch.isfinite(gaps)
        g = torch.where(ok, gaps, torch.zeros_like(gaps))
        e = self.obs_ext
        return {"gap": g, "d": self.obs_d, "vs": self.obs_vs, "vd": self.obs_vd, "back": e[..., 0],
                "front": e[..., 1], "left": e[..., 2], "right": e[..., 3], "ok": ok}

    def opponent_speed(self, planner, state):
        ar = torch.arange(state.shape[0], device=state.device)
        return self.obs_vs[ar, self._nearest].clamp_min(0.0).to(state.dtype)

    def obstacle_clearance(self, q: torch.Tensor) -> torch.Tensor:
        """Distance from points q (B, ..., 2) to the nearest STATIC track of their row's car, minus
        its disc; a car being raced is not in it, as the simulator's prop clearance has no cars."""
        static = self.obs_ok & (self.obs_vs.abs() < self.per.tp.static_speed)
        B = q.shape[0]
        flat = q.reshape(B, -1, 2)
        radius = torch.maximum(self.obs_ext.amax(-1), torch.full_like(self.obs_ext[..., 0], self.per.tp.obs_radius))
        d = (flat[:, :, None, :] - self.obs_xy[:, None, :, :]).norm(dim=-1) - radius[:, None, :]
        d = torch.where(static[:, None, :], d, torch.full_like(d, float("inf")))
        return d.amin(-1).view(q.shape[:-1])

    def follow_cap(self, env, state):
        """`F1VecEnv.follow_cap` from the tracks: the nearest MOVING track ahead, its tracked speed."""
        e = env.ecfg
        tid = env.sim.tid
        b = self.base
        idx = b.project(state[:, :2], tid)[0]
        s_e = idx.to(state.dtype) * b.ds[tid]
        L = b.length[tid]
        g = torch.remainder(self.obs_s - s_e[:, None] + 0.5 * L[:, None], L[:, None]) - 0.5 * L[:, None]
        moving = self.obs_ok & (self.obs_vs.abs() >= self.per.tp.static_speed)
        ahead = torch.where(moving & (g > 0), g, torch.full_like(g, 1e9))
        gap, j = ahead.min(1)
        v_ahead = self.obs_vs.gather(1, j[:, None])[:, 0].clamp_min(0.0)
        closing = (state[:, 3] - v_ahead).clamp_min(0.0)
        trigger = e.opp_follow_gap + closing * closing / (2.0 * e.opp_follow_decel)
        return gap < trigger, (v_ahead * e.opp_follow_ratio).clamp_min(0.5)

    # ------------------------------------------------------------------ for the viewer
    def localisation_error(self, env) -> torch.Tensor:
        return (self.pf.estimate[:, :2] - env.sim.state[self.rows, :2]).norm(dim=1)
