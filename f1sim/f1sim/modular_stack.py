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

    # ------------------------------------------------------------------ one control step
    @torch.no_grad()
    def update(self, env) -> torch.Tensor:
        """Advance localisation and perception on this step's sensors; returns the estimated state
        (B, 8): the stack's rows replaced by what the car believes, the rest untouched (and unread)."""
        lr = env.last_result
        rows = self.rows
        # A new episode: the operator puts the car on the grid and gives the filter its pose. Masked,
        # every step, rather than behind an `any()` that would synchronise with the GPU each time.
        fresh = env.ep_step[rows] == 0
        self.pf.reset(fresh, env.sim.state[rows, :3].clone(), lr.odom[rows, :3].clone())
        self.per.reset(fresh)
        tid = env.sim.tid[rows]
        pose = self.pf.step(lr.odom[rows, :3], lr.scan[rows], tid)
        self.per.step(pose, lr.scan[rows], tid)
        st = env.sim.state.clone()
        est = torch.zeros(rows.numel(), st.shape[1], device=st.device, dtype=st.dtype)
        est[:, :3] = pose
        est[:, 3] = lr.odom[rows, 3]                    # wheel speed
        est[:, 5] = lr.odom[rows, 4]                    # gyro yaw rate
        st[rows] = est
        self.est_state = st
        ok = self.per.confirmed
        self.obs_ok[rows] = ok
        self.obs_s[rows] = self.per.x[..., 0]
        self.obs_d[rows] = self.per.x[..., 1]
        self.obs_vs[rows] = self.per.x[..., 2]
        self.obs_vd[rows] = self.per.x[..., 3]
        self.obs_ext[rows] = self.per.ext
        self.obs_xy[rows] = self.per.to_world(self.per.x[..., 0], self.per.x[..., 1], tid[:, None].expand_as(ok))
        return st

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
