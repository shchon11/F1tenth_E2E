"""A minimum-time racing line for each procedural obstacle layout, built on the GPU, and the teacher
that drives it.

Why (2026-09-24). The raceline teacher drives the empty track at its limit and touches nothing,
because its line is optimised offline and then only tracked. A procedural layout is static for an
episode -- a prop moves only when something hits it -- so the same holds for obstacles if each
layout gets its own line. Measured on eight ICCAS layouts with lines from `Raceline.build` on the
layout's burned-in grid: no prop and no wall touched in 37.5 s per car. But that solver took 1-4
minutes per layout on this machine's four cores and did not converge on four of the eight, and a
real-time planner (`lattice`, tried first) never got past the mismatch between what it planned and
what the car then did: it re-planned every 25 ms and the car ended up 7 cm (median) and 19 cm (p90)
from its plan in half a second.

`LayoutLines` builds the lines for many layouts at once, in seconds:

1. The layout's props are rasterised the way `Track.for_planning` does a map's own, and distance
   fields are taken to the walls and to the props separately.
2. Two seeds per layout: the empty track's minimum-time line routed round every blocked stretch
   (`planning_seed.obstacle_aware_seed`, shortest detour), and a whole-lap lattice DP over lateral
   offset and slope that prices offset, slope, the lateral acceleration a change of slope asks for
   at the empty line's speed there, and clearance. The router picks the shortest way round a prop;
   the DP the one that keeps pace -- on one layout the router threaded a gap between two crates
   (10.5 s) where the DP went round them (7.3 s).
3. Near the props only (`NEAR_PROP_M`, `FREE_M`), each seed, smoothed, is the reference for
   offsets along its normals: a periodic cubic spline over control points every `CTRL_M`, faded in
   and out so the line joins the empty track's line smoothly and is that line everywhere else. They are optimised with Adam for lap time -- a closed-form
   quasi-steady profile (grip-limited cornering speed, forward and backward acceleration passes as
   running minima) -- under exact penalties that keep every point `PROP_MARGIN` from the props and,
   from the walls, the empty line's own margin (0.40 m, or what the empty line had where it had
   less). A distance field is the constraint, not a lane measured along normals: detours that bend
   the normals made those corridors contradict each other.
4. The winner per layout is the feasible line with the shorter lap under `raceline.speed_profile`,
   the profile the raceline teacher itself drives; feasibility is `check_line` (the body rectangle
   against every prop polygon, the distance field at its corners).

`LayoutLineTeacher` is the raceline teacher with one line per env row (its track id is the row),
rebuilt for the rows whose layout changed since the last call.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .planning_seed import obstacle_aware_seed
from .prop_math import _polygon_vertices, _slot_is_live, _to_world, section_halfplanes
from .raceline import Raceline, normals, resample_closed, speed_profile
from .track import Track

#: Clearance kept from a prop by the car's body [m]. The raceline teacher runs up to ~0.15 m inside
#: its line at speed (the reason the empty line keeps 0.40 m from the walls); with 0.20 m a car 0.14 m
#: off its line for a whole second touched a crate its line cleared by 0.22 m. So 0.15 of that
#: offset plus 0.15.
PROP_MARGIN = 0.30
#: Where the line moves round a prop, the least it may keep from a wall [m past half the body]: the
#: same tracking offset and a little more.
WALL_FLOOR = 0.25
#: The empty line's wall margin: half the body plus this, as `Raceline.build` keeps it.
WALL_MARGIN = 0.40
#: Spline control spacing [m] of the optimised offsets.
CTRL_M = 0.4
#: Only the line near the props moves: points within `NEAR_PROP_M` of one or where a seed departs
#: from the empty line, and `FREE_M` either side of those. Everywhere else the empty track's
#: minimum-time line stands -- the quasi-steady model below is not the profile the teacher drives,
#: and left to optimise an empty track it made the line slower (6.56 s -> 6.82 s).
NEAR_PROP_M = 1.2
FREE_M = 2.0
#: Curvature stencil of the optimiser's speed model, in line points (4 cm apart on ICCAS).
STENCIL = 3
#: Adam steps (cosine-decayed). Few on purpose: the optimiser's quasi-steady model is not the profile
#: the teacher drives, and the longer it is fitted the further the two part. Measured on 32 layouts,
#: mean profile lap: 150 steps 6.98 s, 300 6.94-6.96, 600 6.92-6.96, 2000 7.02 (4.4 s against 11 s).
ITERS = 600
#: When during the optimisation a line is kept as a candidate, as fractions of `ITERS`; each is
#: judged by the teacher's own profile. Earlier snapshots bought nothing at 600 steps.
SNAPSHOTS = (1.0,)


def env_polygons(env, rows: Sequence[int], body_height: float = 0.27) -> List[List[np.ndarray]]:
    """Each row's live props as world polygons (V, 2), the ones a car body can touch."""
    poses, pn, pd, zlo, zhi = env.procedural.slots()
    out = []
    for b in rows:
        live = _slot_is_live(pn[b], zlo[b], zhi[b]) & (zlo[b] < body_height) & (zhi[b] > 0)
        polys = []
        for c in torch.nonzero(live).flatten().tolist():
            nw, dw = _to_world(pn[b, c][None], pd[b, c][None], poses[b, c][None])
            valid = pn[b, c].pow(2).sum(-1) > 0.5
            polys.append(_polygon_vertices(nw, dw, valid[None])[0][valid].cpu().numpy().astype(float))
        out.append(polys)
    return out


def burn_layout(track: Track, polygons: List[np.ndarray]) -> Track:
    """`track` with every cell intersecting one of `polygons` (each (V, 2), world) occupied."""
    occ = track.occupancy.copy()
    res, origin = float(track.resolution), np.asarray(track.origin, dtype=float)
    H, W = occ.shape
    for v in polygons:
        lo = np.maximum(np.floor((v.min(0) - origin) / res).astype(int), 0)
        hi = np.minimum(np.floor((v.max(0) - origin) / res).astype(int), [W - 1, H - 1])
        if np.any(lo > hi):
            continue
        rows, cols = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        centers = np.stack([cols + 0.5, rows + 0.5], -1) * res + origin
        n_, o_ = section_halfplanes(v, len(v))
        cell = 0.5 * res * np.abs(n_).sum(1)
        occ[rows, cols] |= np.all(centers @ n_.T <= o_ + cell + 1e-12, -1)
    return Track.from_occupancy(occ, res, track.origin, track.centerline, track.name)


def check_line(xy: np.ndarray, polygons: List[np.ndarray], track: Track,
               half_length: float = 0.29, half_width: float = 0.155) -> Tuple[float, float]:
    """(least prop clearance, least wall clearance) [m] of a body driven along `xy` with its
    heading on the line: separating-axis gap to every polygon, and the distance field at the body's
    corners minus half a cell."""
    t = np.roll(xy, -1, 0) - np.roll(xy, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)
    corners = np.stack([xy + s1 * half_length * t + s2 * half_width * nrm
                        for s1 in (1, -1) for s2 in (1, -1)], 1)                # (N, 4, 2)
    prop = np.full(len(xy), np.inf)
    for v in polygons:
        n_, o_ = section_halfplanes(v, len(v))
        face = (np.einsum("nck,fk->ncf", corners, n_).min(1) - o_).max(-1)
        rel = v[None] - xy[:, None]
        vx = (rel * t[:, None]).sum(-1)
        vy = (rel * nrm[:, None]).sum(-1)
        gx = np.maximum(vx.min(1) - half_length, -half_length - vx.max(1))
        gy = np.maximum(vy.min(1) - half_width, -half_width - vy.max(1))
        prop = np.minimum(prop, np.maximum(face, np.maximum(gx, gy)))
    res, origin, edt = float(track.resolution), np.asarray(track.origin), track.edt
    ij = np.floor((corners.reshape(-1, 2) - origin) / res).astype(int)
    ij[:, 0] = np.clip(ij[:, 0], 0, edt.shape[1] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, edt.shape[0] - 1)
    wall = edt[ij[:, 1], ij[:, 0]].reshape(-1, 4).min(1) - 0.5 * res
    return float(prop.min()), float(wall.min())


def profile_lap(xy: np.ndarray, profile_kw: dict) -> float:
    """Lap time [s] of `xy` under `raceline.speed_profile`, the profile the raceline teacher drives."""
    v = speed_profile(xy, **profile_kw)
    ds = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)
    return float((2 * ds / np.maximum(v + np.roll(v, -1), 1e-9)).sum())


def _smooth_closed(p: np.ndarray, sigma_pts: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d
    return np.stack([gaussian_filter1d(p[:, k], sigma_pts, mode="wrap") for k in range(2)], 1)


class LayoutLines:
    """Builds minimum-time lines round obstacle layouts on one track, many layouts per call."""

    def __init__(self, track: Track, base: Raceline, device, *, veh_width: float = 0.31,
                 a_lat: float = 7.0, a_acc: float = 6.5, a_brake: float = 4.0, v_max: float = 10.0,
                 vehicle=None, mu: float = 1.0489, iters: int = ITERS):
        from .params import VehicleParams
        self.track, self.device = track, torch.device(device)
        self.base = np.asarray(base.xy, dtype=float)
        self.N = len(self.base)
        self.n0 = normals(self.base)
        self.veh = float(veh_width)
        self.a_lat, self.a_acc, self.a_brake, self.v_max = a_lat, a_acc, a_brake, v_max
        self.vehicle = vehicle or VehicleParams()
        self.profile_kw = dict(v_max=v_max, a_lat=a_lat, a_acc=a_acc, a_brake=a_brake, mu=mu, vehicle=self.vehicle)
        self.iters = int(iters)
        self.kmax = math.tan(self.vehicle.s_max) / (self.vehicle.lf + self.vehicle.lr)
        res = float(track.resolution)
        self.res, self.origin = res, np.asarray(track.origin, dtype=float)
        self.H, self.W = track.occupancy.shape
        from scipy import ndimage
        self.wall_edt = torch.tensor(ndimage.distance_transform_edt(~track.occupancy) * res,
                                     dtype=torch.float32, device=self.device)
        # periodic cubic B-spline basis over the line's points
        L = float(np.linalg.norm(np.roll(self.base, -1, 0) - self.base, axis=1).sum())
        C = max(8, int(round(L / CTRL_M)))
        u = np.arange(self.N) * C / self.N
        j = np.floor(u).astype(int); t = u - j
        w = np.stack([(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6,
                      (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6], 1)
        Bm = np.zeros((self.N, C))
        for k in range(4):
            np.add.at(Bm, (np.arange(self.N), (j - 1 + k) % C), w[:, k])
        self.Bt = torch.tensor(Bm, dtype=torch.float32, device=self.device)
        self.v_base = torch.tensor(speed_profile(self.base, **self.profile_kw), dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------ distance fields
    def _grids(self, burned: List[Track]) -> torch.Tensor:
        from scipy import ndimage
        props = np.stack([ndimage.distance_transform_edt(~(T.occupancy & ~self.track.occupancy)) * self.res
                          for T in burned])
        B = len(burned)
        return torch.cat([self.wall_edt[None, None].expand(B, 1, -1, -1),
                          torch.tensor(props, dtype=torch.float32, device=self.device)[:, None]], 1)

    def _sample(self, grids: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """(B, 2, n) wall and prop distances at world points `xy` (B, n, 2), bilinear."""
        gx = (xy[..., 0] - self.origin[0]) / self.res - 0.5
        gy = (xy[..., 1] - self.origin[1]) / self.res - 0.5
        g = torch.stack([gx / (self.W - 1) * 2 - 1, gy / (self.H - 1) * 2 - 1], -1)[:, None]
        return F.grid_sample(grids, g, align_corners=True, padding_mode="border")[:, :, 0]

    # ------------------------------------------------------------------ seeds
    def _route_seeds(self, burned: List[Track]) -> List[np.ndarray]:
        """The router's seed per layout, or the empty line where it finds no route at this clearance
        (it raises then; the DP seed of the same layout still competes, and `check_line` judges)."""
        out = []
        for T in burned:
            try:
                out.append(resample_closed(obstacle_aware_seed(T, self.base, clearance=self.veh / 2 + PROP_MARGIN + 0.02), self.N))
            except ValueError:
                out.append(self.base.copy())
        return out

    def _dp_seeds(self, grids: torch.Tensor, w_off=0.3, w_slope=0.5, w_lat=1.0, w_clr=2.0) -> List[np.ndarray]:
        dev, B, N = self.device, grids.shape[0], self.N
        st = max(1, int(round(0.25 / float(np.linalg.norm(np.diff(self.base, axis=0), axis=1).mean()))))
        idx = torch.arange(0, N, st, device=dev); NSt = idx.numel()
        lat = torch.arange(-30, 31, device=dev, dtype=torch.float32) * 0.05
        ND, KS = lat.numel(), 6
        mid = ND // 2
        steps = torch.arange(-KS, KS + 1, device=dev); NK = steps.numel()
        dsw = st * float(np.linalg.norm(np.diff(self.base, axis=0), axis=1).mean())
        slopes = steps.float() * 0.05 / dsw
        pb = torch.tensor(self.base, dtype=torch.float32, device=dev)[idx]
        nb = torch.tensor(self.n0, dtype=torch.float32, device=dev)[idx]
        nodes = pb[None, :, None] + lat[None, None, :, None] * nb[None, :, None]
        dist = self._sample(grids, nodes.expand(B, -1, -1, -1).reshape(B, -1, 2)).view(B, 2, NSt, ND)
        swing = 0.29 * slopes.abs()
        bw = self._sample(grids, pb[None].expand(B, -1, -1))[:, 0]
        needw = torch.minimum(torch.full_like(bw, self.veh / 2 + WALL_MARGIN), bw - 0.02)[..., None, None] + swing
        needp = self.veh / 2 + PROP_MARGIN + swing
        cw, cp = dist[:, 0][..., None], dist[:, 1][..., None]
        viol = ((needp - cp) > 0) | ((needw - cw) > 0)
        node = dsw * (w_off * lat[None, None, :, None] ** 2 + w_slope * slopes ** 2
                      + w_clr * ((needp + 0.10 - cp).clamp_min(0) / 0.10) ** 2) + 1e4 * viol.float()
        dk = (steps[:, None] - steps[None, :]).float() * 0.05 / dsw ** 2
        trans = dsw * w_lat * ((self.v_base[idx] ** 2 / self.a_lat)[:, None, None] * dk[None]) ** 2
        rows = torch.arange(B, device=dev)
        s0 = (~viol[:, :, mid, KS]).float().argmax(1)                  # a station where the line is clear
        pred = torch.arange(ND, device=dev)[None, :] - steps[:, None] + KS
        Cc = torch.full((B, ND, NK), 1e9, device=dev); Cc[rows, mid, KS] = 0
        back = torch.empty((B, NSt + 1, ND, NK), dtype=torch.long, device=dev)
        pad = torch.full((B, KS, NK), 1e9, device=dev)
        order = (s0[:, None] + torch.arange(1, NSt + 1, device=dev)[None]) % NSt
        for i in range(NSt):
            si = order[:, i]
            prev = torch.cat([pad, Cc, pad], 1)[:, pred]
            best, arg = (prev + trans[si].transpose(1, 2)[:, :, None, :]).min(-1)
            Cc = best.transpose(1, 2) + node[rows, si]
            back[:, i + 1] = arg.transpose(1, 2)
        g = torch.full((B,), mid, device=dev, dtype=torch.long)
        k = Cc[rows, mid].argmin(1)                                     # back on the line where it began
        gs = []
        for i in range(NSt, 0, -1):
            gs.append(g)
            kp = back[rows, i, g, k]; g = (g - (k - KS)).clamp(0, ND - 1); k = kp
        gs = torch.stack(gs[::-1], 1)
        dd = torch.zeros(B, NSt, device=dev); dd.scatter_(1, order, lat[gs])
        ix = np.r_[idx.cpu().numpy(), N]
        out = []
        for b in range(B):
            d_st = dd[b].cpu().numpy()
            out.append(self.base + np.interp(np.arange(N), ix, np.r_[d_st, d_st[0]])[:, None] * self.n0)
        return out

    # ------------------------------------------------------------------ the optimisation
    def _free_weights(self, seeds: List[np.ndarray], grids: torch.Tensor) -> np.ndarray:
        """(K, N) in [0, 1]: 1 where the line may move, fading to 0 where it is the empty line's."""
        from scipy.ndimage import binary_dilation, gaussian_filter1d
        ds = float(np.linalg.norm(np.diff(self.base, axis=0), axis=1).mean())
        base_t = torch.tensor(self.base, dtype=torch.float32, device=self.device)
        prop_d = self._sample(grids, base_t[None].expand(grids.shape[0], -1, -1))[:, 1].cpu().numpy()
        k = int(np.ceil(FREE_M / ds)); sig = max(1.0, 0.5 / ds)
        out = []
        for i, sd in enumerate(seeds):
            free = (np.linalg.norm(sd - self.base, axis=1) > 0.01) | (prop_d[i] < NEAR_PROP_M)
            free = binary_dilation(np.r_[free, free, free], iterations=k)[len(free): 2 * len(free)]
            out.append(np.clip(2.0 * gaussian_filter1d(free.astype(float), sig, mode="wrap"), 0.0, 1.0))
        return np.stack(out)

    def _optimise(self, seeds: List[np.ndarray], grids: torch.Tensor) -> np.ndarray:
        # callers (evaluation, the env's teacher label) run under no_grad; this is an optimisation
        with torch.enable_grad():
            return self._optimise_inner(seeds, grids)

    def _optimise_inner(self, seeds: List[np.ndarray], grids: torch.Tensor) -> np.ndarray:
        dev, N = self.device, self.N
        wts = self._free_weights(seeds, grids)                                   # (K, N)
        sm = [self.base + w[:, None] * (_smooth_closed(s, 8) - self.base) for s, w in zip(seeds, wts)]
        W = torch.tensor(wts, dtype=torch.float32, device=dev)
        base = torch.tensor(np.stack(sm), dtype=torch.float32, device=dev)
        nrm = torch.tensor(np.stack([normals(s) for s in sm]), dtype=torch.float32, device=dev)
        off0 = ((torch.tensor(np.stack(seeds), dtype=torch.float32, device=dev) - base) * nrm).sum(-1)
        need_w = torch.minimum(torch.full((len(seeds), N), self.veh / 2 + WALL_MARGIN, device=dev),
                               (self._sample(grids, base)[:, 0] - 0.02).clamp_min(self.veh / 2 + WALL_FLOOR))
        need_p = self.veh / 2 + PROP_MARGIN
        alpha = torch.linalg.lstsq(self.Bt, off0.T).solution.T.clone().requires_grad_(True)
        opt = torch.optim.Adam([alpha], lr=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, self.iters))
        H = STENCIL
        snaps = []
        keep = {int(self.iters * f) for f in SNAPSHOTS}
        for it in range(self.iters + 1):
            if it in keep:
                with torch.no_grad():
                    snaps.append((base + (W * (alpha @ self.Bt.T))[..., None] * nrm).cpu().numpy())
            if it == self.iters:
                break
            d = W * (alpha @ self.Bt.T)
            xy = base + d[..., None] * nrm
            x1 = (torch.roll(xy, -H, 1) - torch.roll(xy, H, 1)) * 0.5
            x2 = torch.roll(xy, -H, 1) - 2 * xy + torch.roll(xy, H, 1)
            kap = (x1[..., 0] * x2[..., 1] - x1[..., 1] * x2[..., 0]) / x1.norm(dim=-1).clamp_min(1e-5) ** 3
            ds = (torch.roll(xy, -1, 1) - xy).norm(dim=-1).clamp_min(1e-4)
            cap = torch.sqrt(self.a_lat / kap.abs().clamp_min(1e-3)).clamp(max=self.v_max)
            c2 = torch.cat([cap, cap], 1).square()
            s = torch.cat([torch.zeros_like(ds[:, :1]), torch.cat([ds, ds], 1).cumsum(1)[:, :-1]], 1)
            fwd = torch.cummin(c2 - 2 * self.a_acc * s, 1).values + 2 * self.a_acc * s
            bwd = torch.flip(torch.cummin(torch.flip(c2 + 2 * self.a_brake * s, [1]), 1).values, [1]) - 2 * self.a_brake * s
            v2 = torch.roll(torch.minimum(fwd, bwd)[:, N // 2: N // 2 + N].clamp_min(0.05), N // 2, 1)
            lap = (ds / v2.sqrt()).sum(1)
            dist = self._sample(grids, xy)
            viol = torch.maximum((need_w - dist[:, 0]).clamp_min(0), (need_p - dist[:, 1]).clamp_min(0))
            pen = ((kap.abs() - 0.9 * self.kmax).clamp_min(0).square()).sum(1) + 1e3 * viol.sum(1) + 1e4 * viol.square().sum(1)
            loss = (lap + pen).sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        return np.stack(snaps, 1)                                                  # (K, S, N, 2)

    @torch.no_grad()
    def _no_grad_grids(self, burned):
        return self._grids(burned)

    def build(self, layouts: List[List[np.ndarray]]):
        """(lines (B, N, 2), laps (B,), clearances (B, 2), ok (B,)) for each layout's polygons."""
        burned = [burn_layout(self.track, P) for P in layouts]
        grids = self._no_grad_grids(burned)
        seeds = self._route_seeds(burned)
        with torch.no_grad():
            seeds_dp = self._dp_seeds(grids)
        B = len(layouts)
        cand = self._optimise(seeds + seeds_dp, torch.cat([grids, grids], 0))    # (2B, S, N, 2)
        lines = np.empty((B, self.N, 2)); laps = np.full(B, np.inf); clr = np.full((B, 2), -np.inf)
        ok = np.zeros(B, dtype=bool)
        for b in range(B):
            # every snapshot of both seeds' runs, judged by the profile the teacher drives
            for xy in list(cand[b]) + list(cand[B + b]):
                cp, cw = check_line(xy, layouts[b], burned[b])
                feasible = cp >= PROP_MARGIN - 0.03 and cw >= WALL_FLOOR - 0.05
                lp = profile_lap(xy, self.profile_kw)
                better = (feasible and not ok[b]) or (feasible == ok[b] and lp < laps[b])
                if better:
                    lines[b], laps[b], clr[b], ok[b] = xy, lp, (cp, cw), feasible
        return lines, laps, clr, ok


REJOIN_M = 8.0
#: The rejoin's curvature limits: a share of full lock (the grip limit at the speed braking leaves
#: applies on top), and the share of that over the first 0.3 m while the steering servo comes round.
REJOIN_LOCK = 0.9
REJOIN_FIRST = 0.4


def _rejoin_dp(builder: "LayoutLines", clean: np.ndarray, pose: np.ndarray, grid: torch.Tensor):
    """(line (N, 2), merge index, ok) -- `clean` with a stretch that starts where the car is and
    joins `clean` within `REJOIN_M`, found by DP over (lateral node, slope) along it."""
    dev, N = builder.device, len(clean)
    n = normals(clean)
    p = pose[:2]
    j0 = int(np.linalg.norm(clean - p, axis=1).argmin())
    d0 = float((p - clean[j0]) @ n[j0])
    t = clean[(j0 + 1) % N] - clean[j0 - 1]
    e = (pose[2] - np.arctan2(t[1], t[0]) + np.pi) % (2 * np.pi) - np.pi
    s0 = float(np.tan(np.clip(e, -1.2, 1.2)))
    ds = float(np.linalg.norm(np.diff(clean, axis=0), axis=1).mean())
    st = max(1, int(round(0.24 / ds))); dsw = st * ds
    NSt = int(np.ceil(REJOIN_M / dsw)) + 1
    idx = (j0 + np.arange(NSt) * st) % N
    # 1.25 cm across and 0.24 m along: one node of slope change is 0.22 1/m. At 2.5 cm it was 0.43,
    # more than the first 0.3 m allows, and a car spawned askew could not start turning at all.
    dd, KS = 0.0125, 16
    lat = torch.arange(-120, 121, device=dev, dtype=torch.float32) * dd
    ND, mid = lat.numel(), 120
    steps = torch.arange(-KS, KS + 1, device=dev); NK = steps.numel()
    slopes = steps.float() * dd / dsw
    pb = torch.tensor(clean[idx], dtype=torch.float32, device=dev)
    nb = torch.tensor(n[idx], dtype=torch.float32, device=dev)
    nodes = pb[:, None] + lat[None, :, None] * nb[:, None]                       # (NSt, ND, 2)
    dist = builder._sample(grid[None], nodes.reshape(1, -1, 2))[0].view(2, NSt, ND)
    car = builder._sample(grid[None], torch.tensor(p[None, None], dtype=torch.float32, device=dev))[0, :, 0]
    sig = torch.arange(NSt, device=dev, dtype=torch.float32) * dsw
    # no closer than the car is now at the start, the full margins half a metre on
    full_p, full_w = builder.veh / 2 + PROP_MARGIN, builder.veh / 2 + WALL_FLOOR
    need_p = torch.minimum(torch.full_like(sig, full_p), car[1] - 0.02 + sig)[:, None, None] + 0.29 * slopes.abs()
    need_w = torch.minimum(torch.full_like(sig, full_w), car[0] - 0.02 + sig)[:, None, None] + 0.29 * slopes.abs()
    cw, cp = dist[0][..., None], dist[1][..., None]
    viol = ((need_p - cp) > 0) | ((need_w - cw) > 0)
    node = dsw * 0.2 * lat[None, :, None] ** 2 + 1e4 * viol.float() + 1e4 * ((need_p - cp).clamp_min(0) + (need_w - cw).clamp_min(0))
    # Curvature the car can actually drive from where it is: the steering's own limit and the grip
    # at the speed it has, and much less over the first 0.3 m, while the servo comes round from the
    # straight wheels it was spawned with. Without it the lattice asked for up to 7 1/m; a car
    # spawned beside its line at 2.3 m/s turned as hard as it could (1.7 rad/s) and still fell
    # behind the line, into a crate the line went round.
    dk = (steps[:, None] - steps[None, :]).float() * dd / dsw ** 2                  # (NKp, NKn)
    # The grip limit at the speed the car can be doing there if it brakes (the profile the teacher
    # then drives on this line brakes for the curvature, and did not have a way round otherwise:
    # at 2.3 m/s a crate 1 m ahead with the line 1.1 m to the left had none).
    v0 = max(float(pose[3]) if len(pose) > 3 else 0.0, 0.5)
    v_at = torch.sqrt(torch.clamp(v0 ** 2 - 2 * 0.8 * builder.a_brake * sig, min=0.5 ** 2))
    k_lim = torch.minimum(torch.full_like(sig, REJOIN_LOCK * builder.kmax), 0.8 * builder.a_lat / v_at ** 2)
    first = sig < 0.3 + 1e-6                                                         # (NSt,)
    lim = torch.where(first, REJOIN_FIRST * k_lim, k_lim)
    trans = dsw * 0.05 * dk ** 2 + 1e4 * (dk.abs()[None] > lim[:, None, None]).float()   # (NSt, NKp, NKn)
    g0 = int(np.clip(round(d0 / dd), -mid, mid)) + mid
    k0 = int(np.clip(round(s0 * dsw / dd), -KS, KS)) + KS
    C = torch.full((ND, NK), 1e9, device=dev); C[g0, k0] = 0.0
    pred = torch.arange(ND, device=dev)[None, :] - steps[:, None] + KS
    pad = torch.full((KS, NK), 1e9, device=dev)
    back = torch.empty((NSt, ND, NK), dtype=torch.long, device=dev)
    for i in range(1, NSt):
        prev = torch.cat([pad, C, pad], 0)[pred]                                     # (NKn, ND, NKp)
        best, arg = (prev + trans[i].T[:, None, :]).min(-1)
        C = best.T + node[i]
        back[i] = arg.T
    C = C + 1e3 * lat[:, None] ** 2 + 1e3 * slopes[None] ** 2                       # on the line, along it
    g, k = divmod(int(C.argmin()), NK)
    gs = [g]
    for i in range(NSt - 1, 0, -1):
        kp = int(back[i, g, k]); g = int(np.clip(g - (k - KS), 0, ND - 1)); k = kp
        gs.append(g)
    d_st = lat[torch.tensor(gs[::-1], device=dev)].cpu().numpy()
    d_st[0] = d0
    fine = np.arange(NSt - 1) * st
    d_f = np.interp(np.arange((NSt - 1) * st + 1), np.r_[fine, (NSt - 1) * st], np.r_[d_st[:-1], d_st[-1]])
    from scipy.ndimage import gaussian_filter1d
    d_f = gaussian_filter1d(d_f, max(1.0, 0.08 / ds), mode="nearest"); d_f[0] = d0
    off = np.zeros(N)
    span = (j0 + np.arange(len(d_f))) % N
    off[span] = d_f
    off[(j0 - np.arange(1, int(0.5 / ds))) % N] = d0                              # just behind the car
    line = clean + off[:, None] * n
    return line, int((j0 + (NSt - 1) * st) % N), float(C.min()) < 1e3


class LayoutLineTeacher:
    """The raceline teacher with one line per env row, each built for that row's obstacle layout.

    `GRAPH_STATE`: which rows are still rejoining, updated in place inside the call.

    `plan_action` has the raceline teacher's signature; `tid` is replaced by the env row, which is
    what indexes this teacher's lines. Driving the learner (`plan_action`, evaluation and DAgger)
    it checks every call for a row whose layout changed and rebuilds that line first -- a host-side
    check. As an opponent (`plan_rows`) it does nothing of the kind inside the call, so the
    console's teacher graph can capture it: the env announces every reset (`reset_rows`, after the
    new layout is drawn) and the lines are rebuilt there, in place, where the graph reads them. The
    eager call was 35 ms, most of a 49 ms console step. A row whose layout no line could be proved
    clear for keeps its best line and is flagged in `label_valid`. Without procedural obstacles it
    is the raceline teacher.

    After a spawn (or any rebuild) a car that is not on its line gets a rejoin line: the same line
    with a stretch, found by DP from the car's own offset and heading, that joins it within
    `REJOIN_M` and keeps the prop and wall margins (no closer than the car already is, at first).
    The raceline teacher's pure-pursuit rejoin was the one contact left in 3676 encounters: a car
    spawned 1.1 m off its line cut back across to it and brushed a crate. The switch back to the
    row's own line happens on the device, when the car passes the merge point.

    `a_lat`: the lateral limit the lines are *driven* at, when it should differ from the one they
    were optimised for -- the minimum-time solve for the empty track's line stops converging above
    7 m/s^2 on ICCAS, while the geometry changes little and the profile is recomputed anyway.
    """

    GRAPH_STATE = ("_joining",)

    def __init__(self, base, env, *, iters: int = ITERS, a_lat: Optional[float] = None):
        from .teacher import RacelineTeacher
        if not isinstance(base, RacelineTeacher):
            raise TypeError("LayoutLineTeacher wraps a RacelineTeacher: the grip, the limits and the "
                            "tracking are all its own")
        if env is None:
            raise ValueError("LayoutLineTeacher needs the env whose layouts it drives")
        if int(env.sim.track.T) != 1:
            raise ValueError("LayoutLineTeacher builds lines on one track")
        self.base, self.env, self.device = base, env, base.device
        track = env.sim.track.tracks[0]
        rl = Raceline.from_xy(base.xy[0].cpu().numpy().astype(float), base.v[0].cpu().numpy().astype(float))
        self.builder = LayoutLines(track, rl, self.device, veh_width=float(env.cfg.vehicle.width),
                                   a_lat=base.a_lat, a_acc=base.a_acc, a_brake=base.a_brake,
                                   vehicle=env.cfg.vehicle, mu=base.mu_nom, iters=iters)
        self._kw = dict(wheelbase=base.L, device=self.device, vehicle=base.vehicle, mu_nominal=base.mu_nom,
                        mu_f_scale_nominal=base.mu_f_nom, a_lat=base.a_lat if a_lat is None else float(a_lat),
                        a_acc=base.a_acc, a_brake=base.a_brake)
        B = env.B
        # one line's profiles, repeated per row: every row starts on the empty track's line. Rows
        # [0, B) hold each row's line, [B, 2B) the line it rejoins that one by after a spawn.
        self.lines = RacelineTeacher([rl], **self._kw)
        for name in ("xy", "v_grip", "kappa", "tan", "length", "ds"):
            t = getattr(self.lines, name)
            setattr(self.lines, name, t.expand(2 * B, *t.shape[1:]).clone())
        self.lines.v = self.lines.v_grip[:, self.lines.nominal_grip_index]
        #: (B,) rows still on their rejoin line, and where that line meets the row's own line.
        self._joining = torch.zeros(B, dtype=torch.bool, device=self.device)
        self._merge = torch.zeros(B, dtype=torch.long, device=self.device)
        self.rejoins = 0
        self.lines.label_grip = base.label_grip
        self.lines.speed_scale = base.speed_scale
        self.label_valid = torch.zeros(B, dtype=torch.bool, device=self.device)
        self.last_laps = np.full(B, np.nan)
        self._poses = None
        self.rebuilds = 0

    # The per-car settings live on the base teacher, as for every other teacher kind: a slot's
    # speed band and grip label are written there (`opponent_slots`), and read from there each call.
    @property
    def speed_scale(self):
        return self.base.speed_scale

    @speed_scale.setter
    def speed_scale(self, v):
        self.base.speed_scale = v

    @property
    def label_grip(self) -> str:
        return self.base.label_grip

    @label_grip.setter
    def label_grip(self, v: str):
        self.base.label_grip = v

    @property
    def offset_limit(self):
        return self.base.offset_limit

    @offset_limit.setter
    def offset_limit(self, v):
        self.base.offset_limit = v

    def _sync(self) -> None:
        b, l_ = self.base, self.lines
        l_.speed_scale, l_.label_grip, l_.label_grip_codes = b.speed_scale, b.label_grip, b.label_grip_codes
        l_.offset_limit = None          # an offset limit is per raceline point of the base line

    def project(self, xy, tid=None):
        return self.lines.project(xy, torch.arange(xy.shape[0], device=xy.device))  # the rows' own lines

    def reset_rows(self, rows) -> None:
        """The env has drawn these rows' new layouts: rebuild their lines now, in place.

        A prop shoved later in the race is not noticed on this path (the learner path's `refresh`
        does notice it); this teacher does not shove props."""
        if getattr(self.env, "procedural", None) is None or self._poses is None:
            return
        self.refresh(rows=torch.as_tensor(rows).flatten().tolist())

    def refresh(self, rows: Optional[List[int]] = None) -> None:
        """Rebuild the lines of `rows`, or of every row whose layout moved since the last rebuild."""
        if getattr(self.env, "procedural", None) is None:
            return
        poses = self.env.procedural.p_poses
        if rows is None:
            changed = (torch.ones(poses.shape[0], dtype=torch.bool, device=poses.device) if self._poses is None
                       else ((poses - self._poses).abs().amax((1, 2)) > 0.02))
            rows = torch.nonzero(changed).flatten().tolist()
        if not rows:
            return
        layouts = env_polygons(self.env, rows)
        lines, laps, clr, ok = self.builder.build(layouts)
        r = torch.tensor(rows, device=self.device)
        gb = self.base.grip_bin(self.env.sim.P, self.env.B, self.device)[r].tolist()
        self._write_rows(r, lines, gb)
        # rejoin lines for the rows whose car is not on its new line
        cars = self.env.sim.state[r, :4].cpu().numpy().astype(float)                 # x, y, yaw, vx
        join, merge, joining = lines.copy(), np.zeros(len(rows), dtype=int), np.zeros(len(rows), dtype=bool)
        burned = [burn_layout(self.builder.track, P) for P in layouts]
        grids = self.builder._grids(burned)
        for i, (xy, pose) in enumerate(zip(lines, cars)):
            nrm = normals(xy)
            j = int(np.linalg.norm(xy - pose[:2], axis=1).argmin())
            off = abs(float((pose[:2] - xy[j]) @ nrm[j]))
            t = xy[(j + 1) % len(xy)] - xy[j - 1]
            e = abs((pose[2] - np.arctan2(t[1], t[0]) + np.pi) % (2 * np.pi) - np.pi)
            if off < 0.10 and e < 0.15:
                continue
            with torch.no_grad():
                cand, m, found = _rejoin_dp(self.builder, xy, pose, grids[i])
            if found:
                cp, cw = check_line(cand, layouts[i], burned[i])
                # the car itself may start closer than the margins; judged from 0.5 m on it must not
                if cp >= min(PROP_MARGIN - 0.03, 0.10) and cw >= min(WALL_FLOOR - 0.05, 0.08):
                    join[i], merge[i], joining[i] = cand, m, True
        self._write_rows(r + self.env.B, join, gb)
        self._merge[r] = torch.tensor(merge, device=self.device)
        self._joining[r] = torch.tensor(joining, device=self.device)
        self.rejoins += int(joining.sum())
        self.label_valid[r] = torch.tensor(ok, device=self.device)
        self.last_laps[rows] = laps
        if self._poses is None:
            self._poses = poses.clone()
        else:
            self._poses[r] = poses[r]
        self.rebuilds += len(rows)

    def _write_rows(self, r: torch.Tensor, lines: np.ndarray, bins: List[int]) -> None:
        """Write lines (R, N, 2) into rows `r` of the per-row raceline teacher, in place.

        The raceline teacher computes a profile at each of its 14 grip levels, 68 ms apiece -- 7.6 s
        for eight lines, most of a console reset. Only the level each row is driving at (and the
        nominal one) is computed exactly; the others are the nearest exact one scaled by the square
        root of the grip ratio, which is what `RacelineTeacher.speed_mode = "sqrt"` does. They are
        read only if a row's friction changes mid-episode (the console's friction setting)."""
        from .raceline import curvature
        L_ = self.lines
        N, G, nom = L_.N, np.asarray(L_.grip_levels, dtype=float), L_.nominal_grip_index
        kw = self._kw
        xs, ks, vs, lens = [], [], [], []
        for xy, b in zip(lines, bins):
            xr = resample_closed(np.asarray(xy, dtype=float), N)
            exact = sorted({int(b), nom})
            prof = {k: speed_profile(xr, 10.0, kw["a_lat"], kw["a_acc"], kw["a_brake"],
                                     mu=kw["mu_nominal"] * G[k], vehicle=kw["vehicle"]) for k in exact}
            near = [min(exact, key=lambda j: abs(G[j] - G[k])) for k in range(len(G))]
            vs.append(np.stack([prof[k] if k in prof else prof[near[k]] * np.sqrt(G[k] / G[near[k]])
                                for k in range(len(G))]))
            xs.append(xr); ks.append(curvature(xr))
            lens.append(float(np.linalg.norm(np.roll(xr, -1, 0) - xr, axis=1).sum()))
        f = lambda a: torch.tensor(np.stack(a) if isinstance(a, list) else a, dtype=torch.float32, device=self.device)
        xy_t = f(xs)
        tan = torch.roll(xy_t, -1, 1) - torch.roll(xy_t, 1, 1)
        L_.xy[r] = xy_t
        L_.tan[r] = tan / tan.norm(dim=2, keepdim=True).clamp_min(1e-9)
        L_.kappa[r] = f(ks)
        L_.v_grip[r] = f(vs)
        L_.length[r] = f(np.asarray(lens))
        L_.ds[r] = L_.length[r] / N

    def _line_ids(self, rows: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Which line each of `rows` drives: its rejoin line until it has passed the merge point,
        then its own. Device-only, so the console's teacher graph can hold it."""
        B = self.env.B
        j = self.lines.project(xy, rows)[0]
        N = self.lines.N
        passed = torch.remainder(j - self._merge[rows], N) < N // 2
        still = self._joining[rows] & ~passed
        self._joining[rows] = still
        return rows + B * still.long()

    @torch.no_grad()
    def plan_action(self, state, P=None, tid=None, v_max: float = 8.0, spec=None, iters: int = 6,
                    offset=None, idx=None, plan_speed=None):
        self.refresh()
        self._sync()
        rows = torch.arange(state.shape[0], device=state.device)
        self.last_label_valid = self.label_valid.clone()
        return self.lines.plan_action(state, P, self._line_ids(rows, state[:, :2]), v_max, spec, iters=iters,
                                      offset=offset, plan_speed=plan_speed)

    @torch.no_grad()
    def plan_rows(self, rows, state, P=None, tid=None, v_max: float = 8.0, spec=None, offset=None,
                  plan_speed=None):
        """`plan_action` for `rows` of the batch only, (R, ACT_DIM) -- the opponent teachers' call
        (`row_planning`). The per-car tensors are cut to the rows for the call and put back. No
        host sync here after the first call (see the class note)."""
        if self._poses is None:
            self.refresh()
        self._sync()
        B, l_ = state.shape[0], self.lines
        cut = lambda t: t[rows] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == B else t
        held = (l_.speed_scale, l_.label_grip_codes)
        l_.speed_scale, l_.label_grip_codes = cut(held[0]), cut(held[1])
        try:
            return l_.plan_action(state[rows], None if P is None else {k: cut(v) for k, v in P.items()},
                                  self._line_ids(rows, state[rows, :2]), v_max, spec, offset=cut(offset),
                                  plan_speed=cut(plan_speed))
        finally:
            l_.speed_scale, l_.label_grip_codes = held

    @torch.no_grad()
    def project_rows(self, action, rows, state, P, v_max: float, spec, plan_speed=None):
        """The raceline teacher's `project_plan_action` for an (R, ACT_DIM) `action` of `rows`,
        after the env has scaled its speeds (`row_planning.RowPlanning.project_rows`)."""
        self._sync()
        B, l_ = state.shape[0], self.lines
        cut = lambda t: t[rows] if torch.is_tensor(t) and t.dim() > 0 and t.shape[0] == B else t
        held = (l_.speed_scale, l_.label_grip_codes)
        l_.speed_scale, l_.label_grip_codes = cut(held[0]), cut(held[1])
        try:
            return l_.project_plan_action(action, state[rows], None if P is None else {k: cut(v) for k, v in P.items()},
                                          v_max, spec, plan_speed=cut(plan_speed))
        finally:
            l_.speed_scale, l_.label_grip_codes = held

    @torch.no_grad()
    def __call__(self, state, P=None, tid=None, offset=None):
        self.refresh()
        self._sync()
        return self.lines(state, P, torch.arange(state.shape[0], device=state.device), offset=offset)
