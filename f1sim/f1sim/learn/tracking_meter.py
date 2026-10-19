"""How far the car ends up from the trajectory its controller was handed (2026-10-20).

The user, on the controller tuning: "제어기 튜닝을 속도나 충돌 기준으로 보면 안되지. trajectory와의 rmse로
봐야지". A controller -- the iLQR tracker, ForzaETH's MAP -- is judged by how closely the car follows the
trajectory it is given. Lap time and contacts also measure the planner, and a controller tuned on them
buys speed by leaving the trajectory: cutting the corners the plan drew, slowing where it did not.

Every control step, each ego row that follows a reference has that reference taken as a TIME-INDEXED
trajectory in the world frame:

* a plan policy (iLQR or MAP following): the tracker's own time-indexed reference of the plan,
  `PlanTracker.last_ref` -- the plan's path walked from the measured speed towards its speed profile
  within the tracker's acceleration bounds (`mpc.reference`). `MAPPlanTracker` runs the iLQR for every
  row, so both followers are measured against the same trajectory;
* a planner ego on MAP (`learn.common.PlannerEgoDriver`, the planner baselines): the waypoints the
  planner handed MAP that step, from the car's projection on them, walked the same way.

h control steps later the car's TRUE position is compared with that reference. The measure a controller
is tuned on is the LATERAL one (the user, 2026-10-20: "세로 오차를 왜재 가로를 재야지"): the car's distance
from the reference's PATH -- `xtrack_rmse_m`, at issue for a waypoint reference (its cross-track error)
and per horizon for both kinds (for a plan, which starts at the car, only a horizon means anything).
The longitudinal error is mostly the motor's lag and a controller's steering cannot remove it, so it is
reported, never optimised: per horizon also the RMSE of the error to the time-indexed point split into
lateral and longitudinal parts, and of the speed error. A reference is dropped when its row was reset
before the horizon came.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .. import dynamics as dyn

HORIZONS_S = (0.1, 0.2, 0.3, 0.5)
#: Samples from the first seconds after a car's (re)spawn are left out: cars are placed about 0.2 m off the line,
#: and on Korea 2026 the convergence carried 62-93 % of the squared error of a whole run (2026-10-20), so an
#: unfiltered RMSE measured how fast a car joins its line, not how it follows it.
SETTLE_S = 2.0
#: The horizon of the time-indexed position RMSE (reported, not optimised since 2026-10-20).
OBJECTIVE_HORIZON_S = 0.5
#: A plan's lateral objective: the car's distance from the plan's path this long after it was issued.
PLAN_XTRACK_HORIZON_S = 0.2


def path_distance(path: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """(R,) distance of points `p` (R, 2) from polylines `path` (R, n, 2), the end segments extended (a car
    a little ahead of a short reference, or behind its start, is measured against the path's line)."""
    seg = path[:, 1:] - path[:, :-1]
    L2 = (seg * seg).sum(-1).clamp_min(1e-12)
    u = ((p[:, None, :] - path[:, :-1]) * seg).sum(-1) / L2                    # (R, n-1)
    lo = torch.zeros_like(u); hi = torch.ones_like(u)
    lo[:, 0] = -float("inf"); hi[:, -1] = float("inf")
    u = torch.maximum(torch.minimum(u, hi), lo)
    foot = path[:, :-1] + u[..., None] * seg
    return (p[:, None, :] - foot).norm(dim=-1).min(1).values


def walk_path(xy: torch.Tensor, v_path: torch.Tensor, start_xy: torch.Tensor, v_now: torch.Tensor,
              dt: float, steps: int, a_max: float, a_brake: float, floor: float = 0.3):
    """A waypoint path (R, n, 2) with speeds (R, n) as a time-indexed trajectory (R, steps+1, 4) =
    x, y, heading, speed target: from the projection of `start_xy` on the path, walked from `v_now`
    towards the path's speed within `a_max` / `a_brake` (the walk of `mpc.reference`), straight on past
    the end. Also returns the cross-track distance of `start_xy` from the path (R,)."""
    seg = xy[:, 1:] - xy[:, :-1]                                              # (R, n-1, 2)
    L = seg.norm(dim=-1).clamp_min(1e-9)
    s = torch.cat([torch.zeros_like(L[:, :1]), torch.cumsum(L, 1)], 1)       # (R, n)
    rel = start_xy[:, None, :] - xy[:, :-1]
    u = ((rel * seg).sum(-1) / (L * L)).clamp(0.0, 1.0)                       # (R, n-1)
    foot = xy[:, :-1] + u[..., None] * seg
    d = (start_xy[:, None, :] - foot).norm(dim=-1)
    j = d.argmin(1)
    ar = torch.arange(xy.shape[0], device=xy.device)
    xtrack = d[ar, j]
    s0 = s[ar, j] + u[ar, j] * L[ar, j]
    S = s[:, -1]
    head = torch.atan2(seg[..., 1], seg[..., 0])                              # (R, n-1)

    def at(sq):                                                                # (R,) arc -> x, y, heading, v
        sc = torch.minimum(sq, S)
        i = torch.searchsorted(s.contiguous(), sc[:, None].contiguous())[:, 0].clamp(1, s.shape[1] - 1)
        w = ((sc - s[ar, i - 1]) / (s[ar, i] - s[ar, i - 1]).clamp_min(1e-9)).clamp(0.0, 1.0)
        p = xy[ar, i - 1] * (1 - w)[:, None] + xy[ar, i] * w[:, None]
        h = head[ar, (i - 1).clamp(max=head.shape[1] - 1)]
        over = (sq - S).clamp_min(0.0)
        p = p + over[:, None] * torch.stack([torch.cos(h), torch.sin(h)], 1)
        v = v_path[ar, i - 1] * (1 - w) + v_path[ar, i] * w
        return p, h, v

    out = []
    sq = s0.clone()
    vw = v_now.abs().clamp_min(floor)
    for t in range(steps + 1):
        p, h, v = at(sq)
        out.append(torch.cat([p, h[:, None], v[:, None]], 1))
        if t < steps:
            dv = (v - vw).clamp(-a_brake * dt, a_max * dt)
            vw = (vw + dv).clamp_min(floor)
            sq = sq + vw * dt
    return torch.stack(out, 1), xtrack


def resample(ref: torch.Tensor, dt_ref: float, dt: float, K: int) -> torch.Tensor:
    """(R, T, 4) on the grid 0, dt_ref, ... -> (R, K, 4) on 0, dt, ..., (K-1) dt (held at the last)."""
    T = ref.shape[1]
    tq = torch.arange(K, device=ref.device, dtype=ref.dtype) * (dt / dt_ref)
    i0 = tq.floor().clamp(max=T - 2).long()
    w = (tq - i0.to(ref.dtype)).clamp(0.0, 1.0)
    a, b = ref[:, i0], ref[:, i0 + 1]
    out = a * (1 - w)[None, :, None] + b * w[None, :, None]
    # heading: interpolate the angle difference, not the raw angles
    dpsi = torch.atan2(torch.sin(b[..., 2] - a[..., 2]), torch.cos(b[..., 2] - a[..., 2]))
    out[..., 2] = a[..., 2] + dpsi * w[None, :]
    return out


class TrackingMeter:
    """Wraps `sim.step` (like `severity_meter.ContactSeverityMeter`): the references are read when the
    step is entered -- after the controller has run on the pre-step state -- and the car is compared
    with the references issued h steps before once the step has run."""

    def __init__(self, env):
        self.env = env
        sim = env.sim
        self.dt = float(sim.control_dt)
        self.hs = [max(1, int(round(h / self.dt))) for h in HORIZONS_S]
        self.K = max(self.hs) + 1
        B = int(sim.state.shape[0])
        dev = sim.state.device
        self.ref = torch.zeros(self.K, B, self.K, 4, device=dev)
        self.have = torch.zeros(self.K, B, dtype=torch.bool, device=dev)
        self.epi_at = torch.zeros(self.K, B, dtype=torch.long, device=dev)
        self.issued = torch.full((self.K,), -1, dtype=torch.long)
        self.epi = torch.zeros(B, dtype=torch.long, device=dev)
        self.settle_steps = int(round(SETTLE_S / self.dt))
        self.age_ok = torch.zeros(self.K, B, dtype=torch.bool, device=dev)
        self.t = 0
        z = lambda: torch.zeros((), dtype=torch.float64, device=dev)
        self.acc = {h: {"n": z(), "e2": z(), "lat2": z(), "lon2": z(), "dv2": z(), "xt2": z()} for h in self.hs}
        self.xt = {"n": z(), "e2": z()}
        # per car, the lateral objective's samples (cross-track at issue / a plan's path distance at its horizon):
        # an RMSE over a fleet is carried by its worst cars, so their spread is reported too
        self.row_n = torch.zeros(B, dtype=torch.float64, device=dev)
        self.row_e2 = torch.zeros(B, dtype=torch.float64, device=dev)
        self.h_plan = max(1, int(round(PLAN_XTRACK_HORIZON_S / self.dt)))
        self.source: Optional[str] = None
        self._orig_step = None
        self._orig_map_cmd = None
        self._planner_cls = None
        #: (rows, path_xy, path_v) handed to MAP by a planner ego this step (set by the wrapper below)
        self._planner_ref = None

    # -- attach / detach ------------------------------------------------------------------------
    def __enter__(self):
        sim = self.env.sim
        self._orig_step = sim.step

        def step_spy(*a, **kw):
            self.epi += (sim.steps == 0).long()
            self._issue()
            r = self._orig_step(*a, **kw)
            self._compare()
            self.t += 1
            return r

        sim.step = step_spy
        from . import common
        self._planner_cls = common.PlannerEgoDriver
        orig = self._planner_cls._map_command
        self._orig_map_cmd = orig
        meter = self

        def map_command(drv, env, rows, st, P):
            out = orig(drv, env, rows, st, P)
            last = getattr(drv.tracker, "last", None) or {}
            if env is meter.env and "path_xy" in last:
                meter._planner_ref = (rows, last["path_xy"], last["path_v"])
            return out

        self._planner_cls._map_command = map_command
        return self

    def __exit__(self, *exc):
        if self._orig_step is not None:
            self.env.sim.step = self._orig_step
            self._orig_step = None
        if self._orig_map_cmd is not None:
            self._planner_cls._map_command = self._orig_map_cmd
            self._orig_map_cmd = None
        return False

    # -- one step ----------------------------------------------------------------------------------
    def _issue(self) -> None:
        env, sim = self.env, self.env.sim
        slot = self.t % self.K
        self.issued[slot] = self.t
        self.have[slot] = False
        st = sim.state
        tr = getattr(env, "tracker", None)
        spec = getattr(tr, "spec", None)
        settled = sim.steps >= self.settle_steps                 # past the spawn's convergence
        self.age_ok[slot] = settled
        if self._planner_ref is not None:
            rows, xy, v = self._planner_ref
            self._planner_ref = None
            a_max = float(getattr(spec, "a_max", 6.0)); a_brk = float(getattr(spec, "a_brake", 5.0))
            T = int(math.ceil((self.K - 1) * self.dt / 0.05)) + 1
            walked, xtrack = walk_path(xy.to(st.dtype), v.to(st.dtype), st[rows, :2], st[rows, dyn.IVX],
                                       0.05, T - 1, a_max, a_brk)
            self.ref[slot, rows] = resample(walked, 0.05, self.dt, self.K)
            self.have[slot, rows] = True
            w = settled[rows].double()
            self.xt["n"] += float(w.sum())
            self.xt["e2"] += (w * xtrack.double() ** 2).sum()
            self.row_n.index_add_(0, rows, w)
            self.row_e2.index_add_(0, rows, w * xtrack.double() ** 2)
            self.source = self.source or "planner waypoints (MAP)"
        elif (getattr(env, "act_dim", 2) != 2 and tr is not None and getattr(tr, "last_ref", None) is not None
              and not getattr(env, "_all_external", False) and getattr(env, "_plan_pose", None) is not None):
            rows = env.learner & ~getattr(env, "external_driven", torch.zeros_like(env.learner))
            lr = tr.last_ref.to(st.dtype)                                      # (B, N+1, 4) body frame
            body = resample(lr, float(spec.dt), self.dt, self.K)
            pose = env._plan_pose
            c, s_ = torch.cos(pose[:, 2])[:, None], torch.sin(pose[:, 2])[:, None]
            w = body.clone()
            w[..., 0] = pose[:, 0:1] + c * body[..., 0] - s_ * body[..., 1]
            w[..., 1] = pose[:, 1:2] + s_ * body[..., 0] + c * body[..., 1]
            w[..., 2] = pose[:, 2:3] + body[..., 2]
            self.ref[slot] = torch.where(rows[:, None, None], w, self.ref[slot])
            self.have[slot] = rows
            self.source = self.source or "plan (tracker reference)"
        self.epi_at[slot] = self.epi

    def _compare(self) -> None:
        st = self.env.sim.state
        pos, v = st[:, :2], st[:, dyn.IVX]
        now = self.t + 1                                     # the post-step state is one step after issue t
        for h in self.hs:
            i = now - h
            if i < 0:
                continue
            slot = i % self.K
            if int(self.issued[slot]) != i:
                continue
            ok = self.have[slot] & (self.epi_at[slot] == self.epi) & self.age_ok[slot]
            if not bool(ok.any()):
                continue
            r = self.ref[slot][:, h]                          # (B, 4)
            e = pos - r[:, :2]
            c, s_ = torch.cos(r[:, 2]), torch.sin(r[:, 2])
            lon = e[:, 0] * c + e[:, 1] * s_
            lat = -e[:, 0] * s_ + e[:, 1] * c
            dv = v - r[:, 3]
            xt = path_distance(self.ref[slot][..., :2], pos)    # the distance from the path itself
            m = ok.double()
            a = self.acc[h]
            a["n"] += m.sum()
            a["xt2"] += (m * xt.double() ** 2).sum()
            if h == self.h_plan and self.source and self.source.startswith("plan"):
                self.row_n += m
                self.row_e2 += m * xt.double() ** 2
            a["e2"] += (m * (lat.double() ** 2 + lon.double() ** 2)).sum()
            a["lat2"] += (m * lat.double() ** 2).sum()
            a["lon2"] += (m * lon.double() ** 2).sum()
            a["dv2"] += (m * dv.double() ** 2).sum()

    # -- result ----------------------------------------------------------------------------------
    def report(self) -> dict:
        per = {}
        for hs, h in zip(HORIZONS_S, self.hs):
            a = self.acc[h]
            n = float(a["n"].item())
            if n <= 0:
                continue
            per[f"{hs:.1f}"] = {"n": int(n), "xtrack_rmse_m": math.sqrt(float(a["xt2"].item()) / n),
                                "pos_rmse_m": math.sqrt(float(a["e2"].item()) / n),
                                "lat_rmse_m": math.sqrt(float(a["lat2"].item()) / n),
                                "lon_rmse_m": math.sqrt(float(a["lon2"].item()) / n),
                                "speed_rmse_mps": math.sqrt(float(a["dv2"].item()) / n)}
        n_xt = float(self.xt["n"].item())
        obj = per.get(f"{OBJECTIVE_HORIZON_S:.1f}", {}).get("pos_rmse_m")
        xt0 = math.sqrt(float(self.xt["e2"].item()) / n_xt) if n_xt > 0 else None
        # the lateral objective: a waypoint reference's cross-track error at issue; a plan's distance from its path
        # PLAN_XTRACK_HORIZON_S after issue (at issue it is zero by construction)
        lat_obj = xt0 if xt0 is not None else per.get(f"{PLAN_XTRACK_HORIZON_S:.1f}", {}).get("xtrack_rmse_m")
        cars = None
        have = self.row_n > 0
        if bool(have.any()):
            per_car = (self.row_e2[have] / self.row_n[have]).sqrt().sort().values
            k = per_car.numel()
            share = self.row_e2[have].sort(descending=True).values
            top = max(1, k // 10)
            cars = {"n": int(k), "median_m": float(per_car[k // 2]), "p90_m": float(per_car[min(k - 1, int(0.9 * k))]),
                    "max_m": float(per_car[-1]), "min_m": float(per_car[0]),
                    "worst10pct_share_of_sq_error": float(share[:top].sum() / share.sum().clamp_min(1e-12))}
        return {"tracking": {
            "source": self.source, "horizons": per,
            "xtrack_rmse_m": xt0, "objective_xtrack_m": lat_obj, "objective_per_car": cars,
            "objective_pos_rmse_m": obj, "objective_horizon_s": OBJECTIVE_HORIZON_S,
            "settle_s": SETTLE_S,
            "note": ("the car's TRUE position h control steps after a reference was issued, against where that "
                     "time-indexed reference put it; references issued in the first settle_s after a (re)spawn are "
                     "left out (f1sim.learn.tracking_meter)")}}
