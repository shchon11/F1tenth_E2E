"""Global raceline: minimum-curvature optimization + friction-limited speed profile.

Formulation follows Heilmeier et al. 2019 (TUM) but solved as an iterated bounded
least-squares problem (scipy.optimize.lsq_linear), no external QP solver:
    p_i = c_i + n_i * alpha_i        (lateral offset alpha along the centerline normal, left +)
    kappa(alpha) ~= kappa_0 + J alpha (linearized about the current line)
    min ||kappa_0 + J d||^2 + lam ||D d||^2   s.t.  lo <= alpha + d <= hi
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse as sp
from scipy.optimize import lsq_linear

from .track import Track, resample_closed


# --------------------------------------------------------------------------- geometry
def _circ_diff_matrices(n: int, ds: float):
    """Central first/second difference operators on a closed curve (dense, n x n)."""
    D1 = np.zeros((n, n)); D2 = np.zeros((n, n))
    for i in range(n):
        D1[i, (i + 1) % n] += 0.5 / ds; D1[i, (i - 1) % n] -= 0.5 / ds
        D2[i, (i + 1) % n] += 1 / ds ** 2; D2[i, i] -= 2 / ds ** 2; D2[i, (i - 1) % n] += 1 / ds ** 2
    return D1, D2


def curvature(pts: np.ndarray) -> np.ndarray:
    d = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    dd = np.roll(pts, -1, 0) - 2 * pts + np.roll(pts, 1, 0)
    ds = np.linalg.norm(d, axis=1) / 2
    d = d / (2 * ds[:, None]); dd = dd / (ds[:, None] ** 2)
    return d[:, 0] * dd[:, 1] - d[:, 1] * dd[:, 0]


def normals(pts: np.ndarray) -> np.ndarray:
    t = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], 1)     # left of travel direction


def track_widths(track: Track, pts: np.ndarray, max_w: float = 10.0):
    """Distance from each centerline point to the wall on the left and right, by marching along
    the normal on the occupancy grid. Returns (w_left, w_right) in meters."""
    n = normals(pts)
    step = track.resolution * 0.5
    H, W = track.occupancy.shape
    out = []
    for sign in (1.0, -1.0):
        w = np.full(len(pts), max_w)
        for k in range(1, int(max_w / step)):
            q = pts + sign * n * (k * step)
            col = np.round((q[:, 0] - track.origin[0]) / track.resolution).astype(int)
            row = np.round((q[:, 1] - track.origin[1]) / track.resolution).astype(int)
            inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
            occ = np.ones(len(pts), dtype=bool)
            occ[inside] = track.occupancy[row[inside], col[inside]]
            hit = occ & (w == max_w)
            w[hit] = k * step
            if (w < max_w).all():
                break
        out.append(w)
    return out[0], out[1]


# --------------------------------------------------------------------------- optimization
def _diff_ops(N: int, ds: float):
    """Sparse central first/second difference operators on a closed, uniformly sampled curve."""
    i = np.arange(N); ip = (i + 1) % N; im = (i - 1) % N
    D1 = sp.csr_matrix((np.r_[np.full(N, 0.5 / ds), np.full(N, -0.5 / ds)], (np.r_[i, i], np.r_[ip, im])), shape=(N, N))
    D2 = sp.csr_matrix((np.r_[np.full(N, 1 / ds ** 2), np.full(N, -2 / ds ** 2), np.full(N, 1 / ds ** 2)],
                        (np.r_[i, i, i], np.r_[ip, i, im])), shape=(N, N))
    return D1, D2


def _box_qp(A, b, lo, hi, x0=None, maxiter: int = 3000):
    """min ||A x - b||^2  s.t. lo <= x <= hi, by L-BFGS-B on the normal equations (A sparse).
    scipy's lsq_linear('trf') stops far from the optimum on these systems (cost 232 vs 86 on a
    test track after its 500 iterations); bvls is exact but ~20x slower."""
    from scipy.optimize import minimize
    AtA = (A.T @ A).tocsr(); Atb = A.T @ b
    def f(x):
        g = AtA @ x - Atb
        return 0.5 * float(x @ (g - Atb)), g
    x0 = np.clip(np.zeros_like(lo) if x0 is None else x0, lo, hi)
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=np.stack([lo, hi], 1),
                 options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-14, gtol=1e-9))
    return r.x


def min_curvature_raceline(center: np.ndarray, w_left: Optional[np.ndarray] = None, w_right: Optional[np.ndarray] = None,
                           veh_width: float = 0.31, margin: float = 0.25, iters: int = 30,
                           smooth: float = 0.5, n_points: Optional[int] = None, track: Optional[Track] = None,
                           step_max: float = 0.4, width_cap: Optional[float] = None, tol: float = 1e-3,
                           margin_narrow_ratio: float = 0.25, margin_min: float = 0.12,
                           kappa_max: Optional[float] = 1.1) -> np.ndarray:
    """Minimum-curvature line by iterated bounded least squares (Gauss-Newton on the exact curvature).

    Each iteration linearizes kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5 about the current line
    (derivatives w.r.t. the *current* line's own arclength, denominator included), solves for
    lateral offsets d along its normals within the remaining track width (|d| <= step_max as a
    trust region), backtracks on the true cost sum(kappa^2), then re-parametrizes: the new line
    is resampled uniformly and becomes the next reference. Without the exact denominator and the
    re-parametrization the discrete curvature is under-estimated wherever points bunch up (inside
    of corners) and the optimizer drives the line into V-shaped apexes. When `track` is given the
    widths are re-measured on the map every iteration (Heilmeier et al. 2019 style); otherwise
    the given widths stay fixed and the reference is not re-parametrized.
    margin shrinks on narrow lanes (margin_narrow_ratio * (lane - car), at least margin_min) so a
    1.3 m corridor still leaves the car room to turn; kappa_max is a soft cap (iteratively
    re-weighted rows) so the line never asks for less than the car's minimum turning radius
    (~0.74 m at full lock) where the geometry allows."""
    c = resample_closed(center, n_points or len(center))
    N = len(c)
    def free_space(wl, wr):
        m = np.minimum(margin, np.maximum(margin_min, margin_narrow_ratio * (wl + wr - veh_width)))
        return veh_width / 2 + m
    free = veh_width / 2 + margin
    w_k = np.ones(N)
    reparam = track is not None
    if not reparam:
        if w_left is None or w_right is None:
            raise ValueError("give a track or explicit widths")
        n0 = normals(c); alpha = np.zeros(N)
    cost = lambda xy: float(np.sum(curvature(xy) ** 2))
    line = lambda a: c if reparam else c + n0 * a[:, None]
    cur = cost(line(None) if reparam else line(alpha))
    for it in range(iters):
        ds = float(np.linalg.norm(np.roll(c, -1, 0) - c, axis=1).mean())
        if reparam:
            c = _detour_obstacles(track, c, veh_width / 2 + margin_min)   # an obstacle on the line: go round it first,
            p = c; n = normals(p)                                          # else both widths collapse and the line sits in it
            wl, wr = track_widths(track, p)
            if width_cap is not None:
                wl = np.minimum(wl, width_cap); wr = np.minimum(wr, width_cap)
            fr = free_space(wl, wr)
            lo = -(wr - fr); hi = wl - fr
        else:
            p = c + n0 * alpha[:, None]; n = n0
            fr = free_space(w_left, w_right)
            lo = -(w_right - fr) - alpha; hi = (w_left - fr) - alpha
        bad = lo > hi                                   # narrower than car + margins: stay centred
        mid = 0.5 * (lo[bad] + hi[bad]); lo[bad] = mid - 1e-6; hi[bad] = mid + 1e-6
        D1, D2 = _diff_ops(N, ds)
        x1, y1 = D1 @ p[:, 0], D1 @ p[:, 1]
        x2, y2 = D2 @ p[:, 0], D2 @ p[:, 1]
        num = x1 * y2 - y1 * x2
        den = x1 ** 2 + y1 ** 2
        k0 = num / den ** 1.5
        Nx, Ny = sp.diags(n[:, 0]), sp.diags(n[:, 1])
        J_num = sp.diags(y2) @ D1 @ Nx + sp.diags(x1) @ D2 @ Ny - sp.diags(x2) @ D1 @ Ny - sp.diags(y1) @ D2 @ Nx
        J_den = 2.0 * (sp.diags(x1) @ D1 @ Nx + sp.diags(y1) @ D1 @ Ny)
        J = sp.diags(den ** -1.5) @ J_num - sp.diags(1.5 * num * den ** -2.5) @ J_den
        R = np.sqrt(smooth) * D2 * ds ** 2                # smooth offsets (damping)
        if kappa_max is not None:                         # soft minimax: over-weight rows above the cap
            w_k = 1.0 + 8.0 * np.clip(np.abs(k0) / kappa_max - 0.8, 0.0, None) ** 2 * (np.abs(k0) > 0.8 * kappa_max)
        A = sp.vstack([sp.diags(w_k) @ J, R]).tocsr(); b = np.concatenate([-w_k * k0, np.zeros(N)])
        lo_d = np.maximum(lo, -step_max); hi_d = np.minimum(hi, step_max)
        fix = lo_d > hi_d                               # trust region outside the feasible band: jump to it
        lo_d[fix] = np.clip(lo[fix], -step_max * 4, None); hi_d[fix] = np.clip(hi[fix], None, step_max * 4)
        d = _box_qp(A, b, lo_d, hi_d)
        improved = False
        for scale in (1.0, 0.5, 0.25, 0.125):           # backtracking on the true objective
            if reparam:
                cand = resample_closed(p + n * (scale * d)[:, None], N); val = cost(cand)
            else:
                cand = alpha + scale * d; val = cost(c + n0 * cand[:, None])
            if val < cur:
                improved = True; break
        if not improved:
            break
        if reparam:
            c = cand
        else:
            alpha = cand
        done = (cur - val) < tol * cur
        cur = val
        if done and it > 1:
            break
    race = c if reparam else c + n0 * alpha[:, None]
    race = resample_closed(race, N)
    if track is not None:                                # clearance repair: corner cutting past a convex obstacle
        wl, wr = track_widths(track, race)
        race = _push_clear(track, race, float(np.min(free_space(wl, wr))))
    return race


def _signed_field(track: Track):
    """(signed distance, d/dx, d/dy) on the map grid: +distance to the nearest obstacle outside them,
    -distance to the nearest free cell inside them. The plain EDT is flat zero inside an obstacle, so a
    point that ended up *in* one (an obstacle dropped on the reference line) has no gradient to escape
    along; with the signed field it does. Cached on the track."""
    from scipy import ndimage
    c = getattr(track, "_sdf_cache", None)
    if c is None:
        inside = ndimage.distance_transform_edt(track.occupancy).astype(np.float32) * track.resolution
        sd = (track.edt - inside).astype(np.float32)
        gy, gx = np.gradient(sd, track.resolution)
        c = (sd, gx.astype(np.float32), gy.astype(np.float32))
        try:
            track._sdf_cache = c
        except Exception:
            pass
    return c


def _push_clear(track: Track, pts: np.ndarray, clearance: float, iters: int = 30, tol: float = 0.02) -> np.ndarray:
    """Nudge points closer than `clearance` (minus tol) to any obstacle outwards along the signed
    distance gradient, smoothing the neighbourhood of moved points so the nudge does not leave a kink
    (a 1 cm step between 8 cm samples is a curvature of ~1.5 1/m). Points inside an obstacle are pushed
    out to the nearest free side, which is how a line with a box dropped on it becomes a line around it."""
    from scipy import ndimage
    sd, gx, gy = _signed_field(track)
    xy = pts.copy(); N = len(xy); moved = np.zeros(N, bool)
    def clear(xy):
        rc = np.stack([(xy[:, 1] - track.origin[1]) / track.resolution, (xy[:, 0] - track.origin[0]) / track.resolution])
        return rc, ndimage.map_coordinates(sd, rc, order=1, mode="nearest")
    for it in range(iters):
        rc, d = clear(xy)
        bad = d < clearance - tol
        if not bad.any():
            break
        g = np.stack([ndimage.map_coordinates(gx, rc, order=1, mode="nearest"),
                      ndimage.map_coordinates(gy, rc, order=1, mode="nearest")], 1)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-6)
        xy[bad] += g[bad] * (clearance - d[bad])[:, None]
        moved |= bad
        w = ndimage.binary_dilation(moved, iterations=4)            # wrap-around neighbourhood
        w |= np.roll(moved, 4) | np.roll(moved, -4)
        w &= ~bad                                                   # smooth around the detour, never the escaping
        sm = 0.5 * xy + 0.25 * (np.roll(xy, 1, 0) + np.roll(xy, -1, 0))   # points themselves: averaging them back
        xy[w] = sm[w]                                               # into the obstacle is how this used to oscillate
    return resample_closed(xy, N) if moved.any() else pts


def _smooth_keep_clear(track: Track, pts: np.ndarray, clearance: float, iters: int = 400) -> np.ndarray:
    """Laplacian smoothing that only accepts a move while the point keeps `clearance`. The fallback
    reference is a raw map centerline plus detour bumps: collision-free but as noisy as the SLAM map,
    and a line the tracker cannot follow (curvature spikes) is as useless as one through a wall."""
    from scipy import ndimage
    sd, _, _ = _signed_field(track)
    def clr(xy):
        rc = np.stack([(xy[:, 1] - track.origin[1]) / track.resolution, (xy[:, 0] - track.origin[0]) / track.resolution])
        return ndimage.map_coordinates(sd, rc, order=1, mode="nearest")
    xy = pts.copy()
    for _ in range(iters):
        sm = 0.5 * xy + 0.25 * (np.roll(xy, 1, 0) + np.roll(xy, -1, 0))
        ok = clr(sm) >= clearance
        if not ok.any():
            break
        xy[ok] = sm[ok]
    return xy


def _detour_obstacles(track: Track, pts: np.ndarray, clearance: float, window_m: float = 2.5) -> np.ndarray:
    """Route a closed line around whatever blocks it: for each blocked stretch, offset the line sideways
    by a raised-cosine bump over a window, growing the amplitude until the stretch clears and taking the
    side with more room. Gradient nudging (_push_clear) can only escape a wall it is *near*; a box
    dropped on the line needs a detour with a shape, or the smoothing pulls the line back into it."""
    from scipy import ndimage
    sd, _, _ = _signed_field(track)
    def clr(xy):
        rc = np.stack([(xy[:, 1] - track.origin[1]) / track.resolution, (xy[:, 0] - track.origin[0]) / track.resolution])
        return ndimage.map_coordinates(sd, rc, order=1, mode="nearest")
    N = len(pts)
    d = clr(pts)
    bad = d < clearance - 0.02
    if not bad.any() or bad.all():
        return pts
    ds = float(np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1).mean())
    W = max(4, int(round(window_m / max(ds, 1e-6))))
    idx = np.nonzero(bad)[0]
    runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    if len(runs) > 1 and bad[0] and bad[-1]:                       # wrap-around run
        runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
    xy = pts.copy()
    for run in runs:
        n = normals(xy)
        c0 = int(run[len(run) // 2])
        half = len(run) // 2 + W
        win = (np.arange(c0 - half, c0 + half + 1)) % N
        u = np.linspace(-np.pi, np.pi, len(win))
        bump = 0.5 * (1.0 + np.cos(u))                             # 1 in the middle, 0 and flat at the ends
        probe = 0.6
        left = clr(xy[run] + n[run] * probe).mean(); right = clr(xy[run] - n[run] * probe).mean()
        best = None
        steps = np.linspace(0.0, 1.0, 26)[None, :, None]                # for the crossing test
        for side in ((1.0, -1.0) if left >= right else (-1.0, 1.0)):
            for A in np.linspace(0.15, 1.8, 23):
                cand = xy.copy()
                cand[win] = xy[win] + n[win] * (side * A * bump)[:, None]
                # the offset must sweep through free space the whole way: a bump that merely lands in the
                # open floor beyond a duct hose has fine clearance at both ends and crosses the boundary
                sweep = (xy[win][:, None, :] + (cand[win] - xy[win])[:, None, :] * steps).reshape(-1, 2)
                m = min(float(clr(cand[win]).min()), float(clr(sweep).min()) + 1e-9)
                if best is None or m > best[0]:
                    best = (m, cand)
                if m >= clearance:
                    break
            if best is not None and best[0] >= clearance:
                break
        if best is not None:
            xy = best[1]
    return resample_closed(xy, N)


def speed_profile(pts: np.ndarray, v_max: float = 10.0, a_lat: float = 6.0, a_acc: float = 4.0,
                  a_brake: float = 3.0, v_min: float = 1.0) -> np.ndarray:
    """Friction-ellipse limited speed along a closed path: lateral limit, then forward
    (acceleration) and backward (braking) passes, repeated so the loop closes.
    Defaults are conservative on purpose: the VESC brakes the rear axle only, so usable
    braking is ~mu*g*(rear load share) minus load transfer, i.e. ~3 m/s^2, and the total
    lateral limit is kept below mu*g so trail-braking does not saturate the rear."""
    kap = np.abs(curvature(pts)) + 1e-6
    ds = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
    v = np.minimum(v_max, np.sqrt(a_lat / kap))
    N = len(v)
    for _ in range(3):
        for i in range(N):                                  # forward: accel limit
            j = (i + 1) % N
            ax = a_acc * np.sqrt(max(0.0, 1 - (v[i] ** 2 * kap[i] / a_lat) ** 2))
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * ax * ds[i]))
        for i in range(N - 1, -1, -1):                      # backward: brake limit
            j = (i + 1) % N
            ax = a_brake * np.sqrt(max(0.0, 1 - (v[j] ** 2 * kap[j] / a_lat) ** 2))
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * ax * ds[i]))
    return np.maximum(v, v_min)


# --------------------------------------------------------------------------- container
@dataclass
class Raceline:
    xy: np.ndarray        # (N,2)
    v: np.ndarray         # (N,) target speed
    kappa: np.ndarray     # (N,)
    s: np.ndarray         # (N,) arclength at each point
    length: float
    lap_time: float       # sum ds / v (kinematic estimate)

    @staticmethod
    def build(track: Track, veh_width: float = 0.31, margin: float = 0.40, v_max: float = 10.0,
              a_lat: float = 6.0, a_acc: float = 4.0, a_brake: float = 3.0, iters: int = 30,
              smooth: float = 0.5, width_cap_ratio: float = 0.8) -> "Raceline":
        """margin: free space kept between the car's side and the boundary (0.40 m: the pure-pursuit
        teacher cuts inside the line by up to ~0.15 m at speed, and duct hoses are soft targets anyway).
        width_cap_ratio: a side is never wider than this fraction of the median total lane width,
        so openings into side rooms / pit areas of SLAM maps do not pull the line off the lane."""
        if track.centerline is None:
            raise ValueError("track needs a centerline")
        c = resample_closed(track.centerline, len(track.centerline))
        c = _detour_obstacles(track, c, veh_width / 2 + 0.12)   # obstacles dropped on the centerline (competition boxes)
        wl, wr = track_widths(track, c)
        cap = width_cap_ratio * float(np.median(wl + wr)) if width_cap_ratio else None
        xy = min_curvature_raceline(c, veh_width=veh_width, margin=margin, iters=iters, smooth=smooth,
                                    track=track, width_cap=cap)
        # Never hand back a line that hits something. The corridor model (one interval along the
        # reference normal) cannot always route around an obstacle in a narrow lane -- a 0.5 m box in a
        # 1.7 m lane leaves no feasible band -- and then the optimizer returns a spike through it. The
        # cleared reference itself is collision-free by construction, so fall back to it.
        from scipy import ndimage as _ndi
        def _min_clear(pts):
            rc = np.stack([(pts[:, 1] - track.origin[1]) / track.resolution, (pts[:, 0] - track.origin[0]) / track.resolution])
            return float(_ndi.map_coordinates(track.edt, rc, order=1, mode="nearest").min())
        if _min_clear(xy) < veh_width / 2 or float(np.abs(curvature(xy)).max()) > 4.0:
            safe = _smooth_keep_clear(track, _detour_obstacles(track, c, veh_width / 2 + 0.10), veh_width / 2 + 0.08)
            safe = resample_closed(_push_clear(track, safe, veh_width / 2 + 0.05), len(safe))
            # the first solve started from a reference that ran through the obstacles, so its corridor was
            # degenerate; from a cleared reference the same optimizer has a well-posed problem
            retry = min_curvature_raceline(safe, veh_width=veh_width, margin=margin, iters=iters, smooth=smooth,
                                           track=track, width_cap=cap)
            cands = [xy, safe, retry]
            clear_ok = [p for p in cands if _min_clear(p) >= veh_width / 2]
            xy = (min(clear_ok, key=lambda p: float(np.abs(curvature(p)).max())) if clear_ok
                  else max(cands, key=_min_clear))      # nothing is clean: take the one that hits least
        v = speed_profile(xy, v_max, a_lat, a_acc, a_brake)
        return Raceline.from_xy(xy, v)

    @staticmethod
    def build_cached(track: Track, cache_dir: Optional[str] = None, **kw) -> "Raceline":
        """Raceline.build with an on-disk cache keyed by the track's occupancy + parameters."""
        if getattr(track, "base", None) is not None:                          # carved variants (pockets): the base's line
            return Raceline.build_cached(track.base, cache_dir, **kw)
        import hashlib, os
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "f1sim", "racelines")
        os.makedirs(cache_dir, exist_ok=True)
        import inspect
        params = {k: v.default for k, v in inspect.signature(Raceline.build).parameters.items() if k != "track"}
        params.update(kw)                                  # key includes the *effective* parameters, defaults too
        cl = b"" if track.centerline is None else np.asarray(track.centerline, dtype=np.float32).tobytes()
        h = hashlib.md5(np.packbits(track.occupancy).tobytes() + cl + repr(sorted(params.items())).encode() + b"rl12").hexdigest()[:12]
        path = os.path.join(cache_dir, f"{track.name}_{h}.csv")
        if os.path.exists(path):
            return Raceline.load(path)
        rl = Raceline.build(track, **kw)
        rl.save(path)
        return rl

    @staticmethod
    def from_xy(xy: np.ndarray, v: np.ndarray) -> "Raceline":
        ds = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
        return Raceline(xy, v, curvature(xy), s, float(ds.sum()), float((ds / v).sum()))

    def save(self, path: str):
        np.savetxt(path, np.column_stack([self.xy, self.v, self.kappa]), delimiter=",",
                   header="x_m,y_m,v_mps,kappa", comments="# ")

    @staticmethod
    def load(path: str) -> "Raceline":
        d = np.genfromtxt(path, delimiter=",", comments="#")
        return Raceline.from_xy(d[:, :2], d[:, 2])
