"""Obstacle detection and tracking for the modular baselines (2026-10-18).

The user: "Localization Perception prediction은 개나줬냐?" A modular stack sees the other cars and the
crates only through its own LiDAR, from where it believes it is. This is ForzaETH's perception
(`race_stack/perception`: `detect.py` + `tracking.py`, arXiv:2403.11784 section IV), batched:

* **Detection** -- the scan put in the map frame from the localisation estimate; points within
  `map_margin` of the static map are the map (their `boundaries_inflation`); the rest are split into
  clusters by the Adaptive Breakpoint Detector (Borges & Aldon: consecutive points break when their
  distance exceeds `r * sin(dphi) / sin(lambda - dphi) + 3 sigma`, their lambda 10 deg, sigma 0.03 m);
  a cluster with fewer than `min_points` points (their `min_obs_size` 10) or wider than `max_size` is
  dropped, and so is one whose centre lies outside the lane. Upstream fits a rectangle to each
  cluster; here the centre is the cluster's mean pushed `center_shift` away from the scanner -- the
  visible face is the near side of the object.
* **Tracking** -- a constant-velocity Kalman filter per track in the raceline's Frenet frame
  (s, d, vs, vd), nearest-neighbour association inside a gate, a track confirmed after `confirm` hits
  and dropped after `max_misses` scans unseen. A track slower than `static_speed` is static -- a crate,
  or a stopped car -- and the planners treat it as such.

Prediction is the planners' own: the spliner aims at a tracked obstacle where it is, the predictive
spliner where its tracked speed puts it `FIXED_PRED_TIME` from now (constant velocity -- upstream's
learned opponent model is not reproduced, and this is the simpler predictor, not the better one).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DetectParams:
    lam_deg: float = 10.0          # ABD lambda
    sigma: float = 0.03            # ABD range noise [m]
    min_points: int = 10           # min_obs_size
    max_size: float = 0.8          # [m] widest cluster kept (upstream 0.5 for their 0.5 m cars; ours read longer)
    max_dist: float = 9.0          # max_viewing_distance [m]
    map_margin: float = 0.15       # [m] a point this close to the static map is the map
    lane_margin: float = 0.05      # [m] a centre must be this far inside the lane
    center_shift: float = 0.12     # [m] visible face -> centre
    max_obs: int = 16              # clusters kept per car, nearest first
    merge_dist: float = 0.45       # [m] centres closer than this are one object (a car's corner splits it)


@dataclass
class TrackParams:
    slots: int = 6
    gate: float = 1.0              # [m] association distance
    sigma_meas: float = 0.10       # [m] measurement noise, s and d
    sigma_acc: float = 4.0         # [m/s^2] process noise of the constant-velocity model
    confirm: int = 3
    max_misses: int = 10           # 0.25 s at 40 Hz
    static_speed: float = 0.5      # [m/s]
    obs_radius: float = 0.25       # [m] the disc an obstacle occupies, for clearance


class Perception:
    """Detection + tracking for env `rows`, against `base` (a `RacelineTeacher`: the Frenet frame)."""

    def __init__(self, sim, base, rows: torch.Tensor, det: Optional[DetectParams] = None,
                 trk: Optional[TrackParams] = None, mount=(0.297, 0.0, 0.110)):
        self.sim, self.base = sim, base
        self.dp, self.tp = det or DetectParams(), trk or TrackParams()
        self.device = sim.state.device
        self.rows = rows.to(self.device).long()
        R, T = self.rows.numel(), int(self.tp.slots)
        self.R, self.T = R, T
        lid = sim.cfg.lidar
        self.n = int(lid.n_beams)
        self.angles = torch.linspace(-0.5 * float(lid.fov), 0.5 * float(lid.fov), self.n, device=self.device)
        self.dphi = float(lid.fov) / (self.n - 1)
        self.mount = tuple(float(m) for m in mount)
        self.dt = float(sim.control_dt)
        self.lane_left, self.lane_right = self._lane_widths()
        self.x = torch.zeros(R, T, 4, device=self.device)             # s, d, vs, vd
        self.P = torch.zeros(R, T, 4, 4, device=self.device)
        self.active = torch.zeros(R, T, dtype=torch.bool, device=self.device)
        self.hits = torch.zeros(R, T, dtype=torch.long, device=self.device)
        self.misses = torch.zeros(R, T, dtype=torch.long, device=self.device)
        C = int(self.dp.max_obs)
        self.det_xy = torch.zeros(R, C, 2, device=self.device)          # this scan's detections (viewer)
        self.det_valid = torch.zeros(R, C, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ the lane
    def _lane_widths(self):
        """(T, N) metres from each raceline point to the static map, to its left and to its right."""
        b, tr = self.base, self.sim.track
        xy, tan = b.xy.to(self.device), b.tan.to(self.device)
        Tn, N, _ = xy.shape
        tid = torch.arange(Tn, device=self.device)[:, None].expand(Tn, N)
        out = []
        for sign in (1.0, -1.0):
            nrm = torch.stack([-tan[..., 1], tan[..., 0]], -1) * sign
            ok = torch.ones(Tn, N, dtype=torch.bool, device=self.device)
            w = torch.zeros(Tn, N, device=self.device)
            for k in range(1, 61):                                    # out to 3 m in 5 cm steps
                o = 0.05 * k
                ok = ok & (tr.sample_edt(xy + nrm * o, tid) > 0.03)
                w = torch.where(ok, torch.full_like(w, o), w)
            out.append(w)
        return out[0], out[1]

    def frenet(self, xy: torch.Tensor, tid: torch.Tensor):
        """(s [m], d [m left], raceline index) of points xy (..., 2) on track tid (...)."""
        b = self.base
        flat, tflat = xy.reshape(-1, 2), tid.reshape(-1)
        idx = b.project(flat, tflat)[0]
        p0, t0 = b.xy[tflat, idx], b.tan[tflat, idx]
        rel = flat - p0
        along = (rel * t0).sum(-1)
        d = rel[:, 1] * t0[:, 0] - rel[:, 0] * t0[:, 1]
        L = b.length[tflat]
        s = torch.remainder(idx.to(xy.dtype) * b.ds[tflat] + along, L)
        sh = xy.shape[:-1]
        return s.view(sh), d.view(sh), idx.view(sh)

    def to_world(self, s: torch.Tensor, d: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """Frenet (s, d) -> world xy, on the raceline's own lattice."""
        b = self.base
        idx = torch.remainder((s / b.ds[tid]).round().long(), b.N)
        t0 = b.tan[tid, idx]
        return b.xy[tid, idx] + d[..., None] * torch.stack([-t0[..., 1], t0[..., 0]], -1)

    # ------------------------------------------------------------------ lifecycle
    def reset(self, which: torch.Tensor) -> None:
        self.active = self.active & ~which[:, None]
        self.hits = torch.where(which[:, None], torch.zeros_like(self.hits), self.hits)
        self.misses = torch.where(which[:, None], torch.zeros_like(self.misses), self.misses)

    # ------------------------------------------------------------------ detection
    @torch.no_grad()
    def detect(self, pose: torch.Tensor, scan: torch.Tensor, tid: torch.Tensor):
        """(centres (R, C, 2) world, valid (R, C)) from one scan at the estimated `pose` (R, 3)."""
        dp = self.dp
        R, n = scan.shape
        r = torch.nan_to_num(scan.float(), nan=1e3, posinf=1e3, neginf=0.0)
        valid = (r > 0.05) & (r < dp.max_dist)
        mx, my, _ = self.mount
        c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        ox = pose[:, 0] + mx * c - my * s
        oy = pose[:, 1] + mx * s + my * c
        a = pose[:, 2:3] + self.angles[None]
        rr = torch.where(valid, r, torch.zeros_like(r))
        pts = torch.stack([ox[:, None] + rr * torch.cos(a), oy[:, None] + rr * torch.sin(a)], -1)   # (R, n, 2)
        edt = self.sim.track.sample_edt(pts, tid[:, None].expand(R, n))
        cand = valid & (edt > dp.map_margin)
        lam = math.radians(dp.lam_deg)
        dmax = torch.minimum(rr[:, 1:], rr[:, :-1]) * (math.sin(self.dphi) / math.sin(lam - self.dphi)) + 3 * dp.sigma
        gap = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)
        link = cand[:, 1:] & cand[:, :-1] & (gap <= dmax)
        false_col = torch.zeros(R, 1, dtype=torch.bool, device=self.device)
        start = cand & ~torch.cat([false_col, link], 1)
        end = cand & ~torch.cat([link, false_col], 1)
        cid = torch.cumsum(start.long(), 1) * cand.long()                       # 0 = not a cluster
        M = n + 1
        flat = (torch.arange(R, device=self.device)[:, None] * M + cid).reshape(-1)
        cnt = torch.zeros(R * M, device=self.device).index_add_(0, flat, cand.reshape(-1).float())
        sx = torch.zeros(R * M, device=self.device).index_add_(0, flat, (pts[..., 0] * cand).reshape(-1))
        sy = torch.zeros(R * M, device=self.device).index_add_(0, flat, (pts[..., 1] * cand).reshape(-1))
        first = torch.zeros(R * M, 2, device=self.device).index_add_(0, flat, (pts * start[..., None]).reshape(-1, 2))
        last = torch.zeros(R * M, 2, device=self.device).index_add_(0, flat, (pts * end[..., None]).reshape(-1, 2))
        cnt, sx, sy = cnt.view(R, M), sx.view(R, M), sy.view(R, M)
        first, last = first.view(R, M, 2), last.view(R, M, 2)
        size = (last - first).norm(dim=-1)
        ok = (cnt >= dp.min_points) & (size <= dp.max_size)
        ok[:, 0] = False
        mean = torch.stack([sx, sy], -1) / cnt.clamp_min(1.0)[..., None]
        org = torch.stack([ox, oy], -1)[:, None, :]
        ray = mean - org
        dist = ray.norm(dim=-1)
        centre = mean + ray / dist.clamp_min(1e-3)[..., None] * dp.center_shift
        C = min(int(dp.max_obs), M)
        key = torch.where(ok, dist, torch.full_like(dist, float("inf")))
        near, j = torch.topk(key, C, dim=1, largest=False)
        cen = torch.gather(centre, 1, j[..., None].expand(R, C, 2))
        w = torch.gather(cnt, 1, j)
        keep = torch.isfinite(near)
        # One object, one detection: the breakpoint detector cuts a car at its corner into two
        # clusters a few centimetres apart, and two tracks for one car is a car the planner sees
        # twice. A cluster within `merge_dist` of a nearer kept one is folded into it (count-weighted).
        for c in range(1, C):
            dist_c = (cen[:, :c] - cen[:, c:c + 1]).norm(dim=-1)                        # (R, c)
            close = keep[:, :c] & (dist_c < dp.merge_dist) & keep[:, c:c + 1]
            if not bool(close.any()):
                continue
            tgt = torch.where(close, dist_c, torch.full_like(dist_c, float("inf"))).argmin(1)
            fold = close.any(1)
            ar = torch.arange(R, device=self.device)
            wt, wc = w[ar, tgt], w[:, c]
            merged = (cen[ar, tgt] * wt[:, None] + cen[:, c] * wc[:, None]) / (wt + wc).clamp_min(1.0)[:, None]
            cen[ar, tgt] = torch.where(fold[:, None], merged, cen[ar, tgt])
            w[ar, tgt] = torch.where(fold, wt + wc, wt)
            keep[:, c] = keep[:, c] & ~fold
        # inside the lane: the raceline's own left/right room at the centre's station
        s_c, d_c, i_c = self.frenet(cen, tid[:, None].expand(R, C))
        tt = tid[:, None].expand(R, C)
        inside = (d_c < self.lane_left[tt, i_c] - dp.lane_margin) & (d_c > -self.lane_right[tt, i_c] + dp.lane_margin)
        keep = keep & inside
        self.det_xy, self.det_valid = cen, keep
        return cen, keep, s_c, d_c

    # ------------------------------------------------------------------ tracking
    @torch.no_grad()
    def step(self, pose: torch.Tensor, scan: torch.Tensor, tid: torch.Tensor) -> None:
        cen, keep, s_c, d_c = self.detect(pose, scan, tid)
        tp, dt = self.tp, self.dt
        R, T, C = self.R, self.T, cen.shape[1]
        L = self.base.length[tid]                                                  # (R,)
        # predict
        F = torch.eye(4, device=self.device)
        F[0, 2] = F[1, 3] = dt
        G = torch.tensor([0.5 * dt * dt, 0.5 * dt * dt, dt, dt], device=self.device)
        Q = torch.diag(G * G) * tp.sigma_acc ** 2
        Q[0, 2] = Q[2, 0] = Q[1, 3] = Q[3, 1] = 0.5 * dt ** 3 * tp.sigma_acc ** 2
        self.x = self.x @ F.T
        self.x[..., 0] = torch.remainder(self.x[..., 0], L[:, None])
        self.P = F @ self.P @ F.T + Q
        # associate (greedy nearest, slot order)
        ds = torch.remainder(s_c[:, None, :] - self.x[..., 0:1] + 0.5 * L[:, None, None], L[:, None, None]) - 0.5 * L[:, None, None]
        dd = d_c[:, None, :] - self.x[..., 1:2]
        cost = torch.sqrt(ds * ds + dd * dd)
        cost = torch.where(self.active[..., None] & keep[:, None, :], cost, torch.full_like(cost, float("inf")))
        used = torch.zeros(R, C, dtype=torch.bool, device=self.device)
        assign = torch.full((R, T), -1, dtype=torch.long, device=self.device)
        ar = torch.arange(R, device=self.device)
        for t in range(T):
            c = cost[:, t, :].masked_fill(used, float("inf"))
            j = c.argmin(1)
            ok = c[ar, j] < tp.gate
            assign[:, t] = torch.where(ok, j, assign[:, t])
            used[ar, j] = used[ar, j] | ok
        got = assign >= 0
        jj = assign.clamp_min(0)
        z_s = torch.gather(s_c, 1, jj)
        z_d = torch.gather(d_c, 1, jj)
        # update
        H = torch.zeros(2, 4, device=self.device)
        H[0, 0] = H[1, 1] = 1.0
        Rm = torch.eye(2, device=self.device) * tp.sigma_meas ** 2
        y = torch.stack([torch.remainder(z_s - self.x[..., 0] + 0.5 * L[:, None], L[:, None]) - 0.5 * L[:, None],
                         z_d - self.x[..., 1]], -1)                                     # (R, T, 2)
        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ torch.linalg.inv(S)                                          # (R, T, 4, 2)
        x_new = self.x + (K @ y[..., None])[..., 0]
        P_new = (torch.eye(4, device=self.device) - K @ H) @ self.P
        self.x = torch.where(got[..., None], x_new, self.x)
        self.P = torch.where(got[..., None, None], P_new, self.P)
        self.x[..., 0] = torch.remainder(self.x[..., 0], L[:, None])
        self.hits = torch.where(got, self.hits + 1, self.hits)
        self.misses = torch.where(got, torch.zeros_like(self.misses), self.misses + self.active.long())
        self.active = self.active & (self.misses <= tp.max_misses)
        # spawn: each unassociated detection takes the first free slot
        P0 = torch.diag(torch.tensor([tp.sigma_meas ** 2, tp.sigma_meas ** 2, 9.0, 1.0], device=self.device))
        for c in range(C):
            new = keep[:, c] & ~used[:, c]
            free = ~self.active
            has = free.any(1) & new
            slot = free.float().argmax(1)
            m = torch.zeros(R, T, dtype=torch.bool, device=self.device)
            m[ar, slot] = has
            init = torch.stack([s_c[:, c], d_c[:, c], torch.zeros(R, device=self.device),
                                torch.zeros(R, device=self.device)], -1)[:, None, :].expand(R, T, 4)
            self.x = torch.where(m[..., None], init, self.x)
            self.P = torch.where(m[..., None, None], P0.expand(R, T, 4, 4), self.P)
            self.hits = torch.where(m, torch.ones_like(self.hits), self.hits)
            self.misses = torch.where(m, torch.zeros_like(self.misses), self.misses)
            self.active = self.active | m

    @property
    def confirmed(self) -> torch.Tensor:
        return self.active & (self.hits >= self.tp.confirm)
