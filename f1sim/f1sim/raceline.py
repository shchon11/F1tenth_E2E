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

from .track import Track, _limit_curvature, resample_closed


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
            p = c; n = normals(p)
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
        clearance = float(np.min(free_space(wl, wr)))
        race = _push_clear(track, race, clearance)
        if kappa_max is not None:
            race = _enforce_turn_radius(race, kappa_max, track, clearance)
    elif kappa_max is not None:
        race = _enforce_turn_radius(race, kappa_max)
    return race


def _enforce_turn_radius(pts: np.ndarray, kappa_max: float, track: Optional[Track] = None,
                         clearance: Optional[float] = None, rounds: int = 12, smooth_iters: int = 400) -> np.ndarray:
    """Make the line's curvature something the car can actually steer.

    The solve treats `kappa_max` as a preference (re-weighted rows), not a constraint, and on some
    layouts it converges to a line that violates it badly: of 118 catalog tracks two came out at
    R = 0.27 m and R = 0.45 m against the car's 0.74 m full-lock radius, both ~10 % longer than the
    same track driven the other way. Those two were also the privileged teacher's worst tracks by a
    wide margin (12.1 and 5.5 collisions/km against 0.18 over the set) -- an unfollowable line makes
    an unfollowable label. Laplacian-smooth the offending stretches, push the result back off the
    boundary, and repeat until the cap holds."""
    n = len(pts)
    peak = lambda p: float(np.abs(curvature(p)).max())
    best = pts
    for _ in range(rounds):
        if peak(best) <= kappa_max:
            break
        pts = resample_closed(_limit_curvature(pts, 1.0 / kappa_max, iters=smooth_iters), n)
        if peak(pts) < peak(best):
            best = pts
        if track is None or clearance is None:
            continue
        # pushing back off the boundary re-introduces curvature at the point it moves, so keep the
        # pushed line only while it is still the flatter of the two; otherwise smooth again from it
        pushed = _push_clear(track, pts, clearance)
        if peak(pushed) < peak(best):
            best = pushed
        pts = pushed
    return best


def _push_clear(track: Track, pts: np.ndarray, clearance: float, iters: int = 30, tol: float = 0.02) -> np.ndarray:
    """Nudge points closer than `clearance` (minus tol) to any obstacle outwards along the distance
    gradient, smoothing the neighbourhood of moved points so the nudge does not leave a kink
    (a 1 cm step between 8 cm samples is a curvature of ~1.5 1/m)."""
    from scipy import ndimage
    gy, gx = np.gradient(track.edt, track.resolution)
    xy = pts.copy(); N = len(xy); moved = np.zeros(N, bool)
    def clear(xy):
        rc = np.stack([(xy[:, 1] - track.origin[1]) / track.resolution, (xy[:, 0] - track.origin[0]) / track.resolution])
        return rc, ndimage.map_coordinates(track.edt, rc, order=1, mode="nearest")
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
        sm = 0.5 * xy + 0.25 * (np.roll(xy, 1, 0) + np.roll(xy, -1, 0))
        xy[w] = sm[w]
    return resample_closed(xy, N) if moved.any() else pts


def speed_profile(pts: np.ndarray, v_max: float = 10.0, a_lat: float = 6.0, a_acc: float = 6.0,
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


# --------------------------------------------------------------------------- minimum time
def _abs_curvature(xy: np.ndarray) -> np.ndarray:
    """|curvature()| for M closed lines at once: xy (M, N, 2) -> (M, N)."""
    d = np.roll(xy, -1, 1) - np.roll(xy, 1, 1)
    dd = np.roll(xy, -1, 1) - 2 * xy + np.roll(xy, 1, 1)
    h = np.linalg.norm(d, axis=2) / 2
    d = d / (2 * h[..., None]); dd = dd / (h[..., None] ** 2)
    return np.abs(d[..., 0] * dd[..., 1] - d[..., 1] * dd[..., 0])


def _lap_times(xy: np.ndarray, v_max: float, a_lat: float, a_acc: float, a_brake: float,
               v_min: float = 1.0, passes: int = 3) -> np.ndarray:
    """`speed_profile` and sum(ds / v) for M closed lines at once: xy (M, N, 2) -> (M,).

    The same recursion, instruction for instruction, with the candidate axis vectorised; the
    passes are sequential in arclength and stay a Python loop. The optimiser below needs a
    finite-difference gradient, i.e. one lap time per degree of freedom per step, and N Python
    iterations per *batch* is what makes that affordable."""
    kap = _abs_curvature(xy) + 1e-6
    ds = np.linalg.norm(np.roll(xy, -1, 1) - xy, axis=2)
    v = np.minimum(v_max, np.sqrt(a_lat / kap))
    N = v.shape[1]
    g = kap / a_lat
    for _ in range(passes):
        for i in range(N):
            j = (i + 1) % N
            ax = a_acc * np.sqrt(np.maximum(0.0, 1 - (v[:, i] ** 2 * g[:, i]) ** 2))
            v[:, j] = np.minimum(v[:, j], np.sqrt(v[:, i] ** 2 + 2 * ax * ds[:, i]))
        for i in range(N - 1, -1, -1):
            j = (i + 1) % N
            ax = a_brake * np.sqrt(np.maximum(0.0, 1 - (v[:, j] ** 2 * g[:, j]) ** 2))
            v[:, i] = np.minimum(v[:, i], np.sqrt(v[:, j] ** 2 + 2 * ax * ds[:, i]))
    return (ds / np.maximum(v, v_min)).sum(1)


def _periodic_bspline_basis(N: int, K: int) -> np.ndarray:
    """(N, K) uniform periodic cubic B-spline basis: non-negative, rows sum to one."""
    u = np.arange(N)[:, None] * (K / N) - np.arange(K)[None, :]
    a = np.abs((u + K / 2) % K - K / 2)
    return np.where(a < 1, 2 / 3 - a ** 2 + a ** 3 / 2, np.where(a < 2, (2 - a) ** 3 / 6, 0.0))


def min_time_raceline(track: Track, start: np.ndarray, veh_width: float = 0.31, margin: float = 0.40,
                      v_max: float = 10.0, a_lat: float = 6.0, a_acc: float = 6.0, a_brake: float = 3.0,
                      width_cap: Optional[float] = None, iters: int = 8, knot_spacing=(2.0, 1.0, 0.5),
                      step_max: float = 0.12, fd_step: float = 0.002, inner_iters: int = 40,
                      margin_narrow_ratio: float = 0.25, margin_min: float = 0.12,
                      kappa_max: Optional[float] = 1.1, kappa_weight: float = 20.0) -> np.ndarray:
    """Descend the lap time of `speed_profile` itself, starting from `start` (the minimum-curvature line).

    Minimum curvature maximises corner speed and ignores everything else a lap is made of: path
    length, and where along the corner the speed is actually wanted (a late apex onto a long
    straight, a short line through a corner that leads nowhere). Those only show up in the time, so
    the time is what is minimised -- the same point-mass, friction-ellipse profile the teacher
    drives, at the same limits, with no surrogate in between. At a_lat 9 / a_acc 7 / a_brake 5 the
    result is 13.6 % quicker than the minimum-curvature line on `real:korea_2025_iccas` (a 3.9 m
    lane, where that line runs 46.2 m against 39.3 m), 4.3 % on `real:map12x16`, 3.5 % on
    `gen:competition:0`.

    Lateral offsets along the current line's normals are a periodic cubic B-spline: smooth by
    construction, so the descent cannot buy time with a kink the finite-difference curvature does
    not see. The basis is non-negative and sums to one, so boxing each coefficient by the tightest
    lane bound under its support keeps the whole offset inside the lane. Each outer iteration moves
    at most `step_max`, re-parametrises and re-measures the lane, exactly as the minimum-curvature
    solve does.

    `knot_spacing` is a coarse-to-fine schedule [m], and the coarse stages are where the time is:
    a lone 0.5 m knot can only trade a little path for a lot of local curvature, so descent on the
    fine basis alone crawls (1.5 % after the same budget that gives 13.6 % with the schedule).
    Moving a whole corner takes a basis function as long as the corner.

    The gradient is a *central* difference over the coefficients, one batched profile per
    evaluation: a bump of height h on a 0.5 m knot changes the local curvature by ~8 h, so a
    one-sided difference measures the second-order cost of the bump (always positive, 4 s/m at
    h = 1 cm on the first track tried) instead of the slope.

    Never returns a slower line than it was given: the result goes through the same clearance and
    turn-radius repair as the minimum-curvature line and is kept only if it is still the quicker of
    the two on the full-resolution profile."""
    from scipy.optimize import minimize
    # At the resolution of the line it was given, never a coarser one: resampling is linear, so a
    # decimated copy carries curvature noise the descent learns to cancel -- measured, a line that
    # gained 1.6 % on a 434-point grid was 19 % *slower* on the 800 points the teacher drives.
    N = len(start)
    length = float(np.linalg.norm(np.roll(start, -1, 0) - start, axis=1).sum())
    lim = (v_max, a_lat, a_acc, a_brake)
    free_space = lambda wl, wr: veh_width / 2 + np.minimum(margin, np.maximum(margin_min, margin_narrow_ratio * (wl + wr - veh_width)))

    def objective(lines):                                  # (M, N, 2) -> (M,)
        t = _lap_times(lines, *lim)
        if kappa_max is None:
            return t
        ds = np.linalg.norm(np.roll(lines, -1, 1) - lines, axis=2)
        return t + kappa_weight * (np.clip(_abs_curvature(lines) - kappa_max, 0.0, None) ** 2 * ds).sum(1)

    c = start
    cur = float(objective(c[None])[0])
    for spacing in np.atleast_1d(knot_spacing):
        K = max(8, int(round(length / spacing)))
        B = _periodic_bspline_basis(N, K)
        support = B > 0
        step = step_max
        for _ in range(iters):
            n = normals(c)
            wl, wr = track_widths(track, c)
            if width_cap is not None:
                wl = np.minimum(wl, width_cap); wr = np.minimum(wr, width_cap)
            fr = free_space(wl, wr)
            lo = -(wr - fr); hi = wl - fr
            bad = lo > hi                                   # narrower than car + margins: stay centred
            mid = 0.5 * (lo[bad] + hi[bad]); lo[bad] = mid - 1e-6; hi[bad] = mid + 1e-6
            lo_k = np.array([lo[support[:, k]].max() for k in range(K)])
            hi_k = np.array([hi[support[:, k]].min() for k in range(K)])
            bad = lo_k > hi_k
            mid = 0.5 * (lo_k[bad] + hi_k[bad]); lo_k[bad] = mid - 1e-6; hi_k[bad] = mid + 1e-6
            # trust region, unless the lane itself has moved out from under the line
            lo_t = np.where(lo_k > step, lo_k, np.maximum(lo_k, -step))
            hi_t = np.where(hi_k < -step, hi_k, np.minimum(hi_k, step))
            lo_t = np.minimum(lo_t, hi_t)

            def f(theta):
                th = np.concatenate([theta[None], theta[None] + fd_step * np.eye(K), theta[None] - fd_step * np.eye(K)])
                val = objective(c[None] + n[None] * (th @ B.T)[..., None])
                return float(val[0]), (val[1:K + 1] - val[K + 1:]) / (2 * fd_step)
            r = minimize(f, np.clip(np.zeros(K), lo_t, hi_t), jac=True, method="L-BFGS-B",
                         bounds=np.stack([lo_t, hi_t], 1), options=dict(maxiter=inner_iters, maxfun=2 * inner_iters))
            cand = resample_closed(c + n * (B @ r.x)[:, None], N)
            val = float(objective(cand[None])[0])
            if val >= cur - 1e-4:
                if step < 0.03:
                    break
                step *= 0.5
                continue
            c, cur = cand, val
    out = c
    wl, wr = track_widths(track, out)
    clearance = float(np.min(free_space(wl, wr)))
    out = _push_clear(track, out, clearance)
    if kappa_max is not None:
        out = _enforce_turn_radius(out, kappa_max, track, clearance)
    lap = lambda p: float((np.linalg.norm(np.roll(p, -1, 0) - p, axis=1) / speed_profile(p, *lim)).sum())
    return out if lap(out) < lap(start) else start


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
              a_lat: float = 6.0, a_acc: float = 6.0, a_brake: float = 3.0, iters: int = 30,
              smooth: float = 0.5, width_cap_ratio: float = 0.8, objective: str = "min_curvature") -> "Raceline":
        """margin: free space kept between the car's side and the boundary (0.40 m: the pure-pursuit
        teacher cuts inside the line by up to ~0.15 m at speed, and duct hoses are soft targets anyway).
        width_cap_ratio: a side is never wider than this fraction of the median total lane width,
        so openings into side rooms / pit areas of SLAM maps do not pull the line off the lane.
        objective: "min_curvature" (the default, and the line every recorded run, obstacle placement
        and frozen suite was built on) or "min_time", which refines that line by descending the lap
        time of the speed profile at these same limits (`min_time_raceline`)."""
        if objective not in ("min_curvature", "min_time"):
            raise ValueError(f"unknown raceline objective {objective!r}")
        if track.centerline is None:
            raise ValueError("track needs a centerline")
        c = resample_closed(track.centerline, len(track.centerline))
        wl, wr = track_widths(track, c)
        cap = width_cap_ratio * float(np.median(wl + wr)) if width_cap_ratio else None
        xy = min_curvature_raceline(c, veh_width=veh_width, margin=margin, iters=iters, smooth=smooth,
                                    track=track, width_cap=cap)
        if objective == "min_time":
            xy = min_time_raceline(track, xy, veh_width=veh_width, margin=margin, v_max=v_max, a_lat=a_lat,
                                   a_acc=a_acc, a_brake=a_brake, width_cap=cap)
        v = speed_profile(xy, v_max, a_lat, a_acc, a_brake)
        return Raceline.from_xy(xy, v)

    @staticmethod
    def build_cached(track: Track, cache_dir: Optional[str] = None, **kw) -> "Raceline":
        """Raceline.build with an on-disk cache keyed by the track's occupancy + parameters."""
        import hashlib, os
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "f1sim", "racelines")
        os.makedirs(cache_dir, exist_ok=True)
        import inspect
        params = {k: v.default for k, v in inspect.signature(Raceline.build).parameters.items() if k != "track"}
        params.update(kw)                                  # key includes the *effective* parameters, defaults too
        if params.get("objective") == "min_curvature":
            del params["objective"]                        # the key every line cached before the option existed has
        cl = b"" if track.centerline is None else np.asarray(track.centerline, dtype=np.float32).tobytes()
        h = hashlib.md5(np.packbits(track.occupancy).tobytes() + cl + repr(sorted(params.items())).encode() + b"rl5").hexdigest()[:12]
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
