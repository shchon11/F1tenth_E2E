"""Periodic minimum-time racing lines under explicit axle and actuator constraints.

Minimum curvature supplies an initialization only. The default solve jointly
optimizes path and speed, requires convergence and dense feasibility, and persists
its diagnostics. Its quasi-steady local optimum is not a full dynamic lap guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sys

import numpy as np
from scipy import sparse as sp

from .track import Track, _limit_curvature, resample_closed


# --------------------------------------------------------------------------- progress
#: Where a long build says how it is getting on. Default: one line on stderr.
#:
#: This exists because the minimum-time solve is twenty to seventy minutes of silence, and the two
#: places that wait on it cannot see stderr at all -- the console GUI shows "레이싱 라인 준비 중"
#: and nothing else for the whole of it, which is indistinguishable from a hang and was reported
#: as one. A hook rather than a return value because the caller that needs the news
#: (`viewer.sim_worker`) is four frames above the loop that has it.
#:
#: Install with `raceline.progress_to(fn)`; `fn(text)` must be cheap and must not raise.
_PROGRESS = None
_PROGRESS_LAST = 0.0
#: Seconds between reports. The solver's inner loop runs far faster than anyone can read.
PROGRESS_EVERY = 5.0


def progress_to(fn):
    """Install a progress sink and return the previous one, for `try/finally` restoration."""
    global _PROGRESS
    prev, _PROGRESS = _PROGRESS, fn
    return prev


def report(text: str, *, force: bool = False) -> None:
    """Say how a long build is getting on, at most every `PROGRESS_EVERY` seconds."""
    global _PROGRESS_LAST
    import time as _t
    now = _t.monotonic()
    if not force and now - _PROGRESS_LAST < PROGRESS_EVERY:
        return
    _PROGRESS_LAST = now
    if _PROGRESS is None:
        print(f"[raceline] {text}", file=sys.stderr, flush=True)
        return
    try:
        _PROGRESS(text)
    except Exception:
        pass                       # a progress sink that fails must not fail the build


# --------------------------------------------------------------------------- geometry
def _circ_diff_matrices(n: int, ds: float):
    """Central first/second difference operators on a closed curve (dense, n x n)."""
    D1 = np.zeros((n, n)); D2 = np.zeros((n, n))
    for i in range(n):
        D1[i, (i + 1) % n] += 0.5 / ds; D1[i, (i - 1) % n] -= 0.5 / ds
        D2[i, (i + 1) % n] += 1 / ds ** 2; D2[i, i] -= 2 / ds ** 2; D2[i, (i - 1) % n] += 1 / ds ** 2
    return D1, D2


def curvature(pts: np.ndarray) -> np.ndarray:
    """Signed three-point circle curvature, valid on nonuniform arc samples."""
    before = pts - np.roll(pts, 1, axis=0)
    after = np.roll(pts, -1, axis=0) - pts
    denominator = (np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
                   * np.linalg.norm(before + after, axis=1))
    return 2 * (before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]) / np.maximum(denominator, 1e-15)


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


def _box_qp(A, b, lo, hi, x0=None, maxiter: int = 3000, deadline: Optional[float] = None):
    """min ||A x - b||^2  s.t. lo <= x <= hi, by L-BFGS-B on the normal equations (A sparse).
    scipy's lsq_linear('trf') stops far from the optimum on these systems (cost 232 vs 86 on a
    test track after its 500 iterations); bvls is exact but ~20x slower."""
    from scipy.optimize import minimize
    import time
    AtA = (A.T @ A).tocsr(); Atb = A.T @ b
    def f(x):
        g = AtA @ x - Atb
        return 0.5 * float(x @ (g - Atb)), g
    x0 = np.clip(np.zeros_like(lo) if x0 is None else x0, lo, hi)
    def callback(x):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("raceline seed deadline exceeded; partial geometry discarded")
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=np.stack([lo, hi], 1),
                 callback=callback if deadline is not None else None,
                 options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-14, gtol=1e-9))
    return r.x


def min_curvature_raceline(center: np.ndarray, w_left: Optional[np.ndarray] = None, w_right: Optional[np.ndarray] = None,
                           veh_width: float = 0.31, margin: float = 0.25, iters: int = 30,
                           smooth: float = 0.5, n_points: Optional[int] = None, track: Optional[Track] = None,
                           step_max: float = 0.4, width_cap: Optional[float] = None, tol: float = 1e-3,
                           margin_narrow_ratio: float = 0.25, margin_min: float = 0.12,
                           kappa_max: Optional[float] = 1.1,
                           _max_seconds: Optional[float] = None) -> np.ndarray:
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
    (~0.74 m at full lock) where the geometry allows. Iteration bounds determine the
    result; an optional wall deadline aborts instead of returning load-dependent geometry."""
    import time
    started = time.monotonic()
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
        if _max_seconds is not None and time.monotonic() - started >= _max_seconds:
            raise TimeoutError("raceline seed deadline exceeded; partial geometry discarded")
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
        d = _box_qp(A, b, lo_d, hi_d,
                    deadline=None if _max_seconds is None else started + _max_seconds)
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


def _legacy_speed_profile(pts: np.ndarray, v_max: float = 10.0, a_lat: float = 6.0, a_acc: float = 6.0,
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


def _first_positive_root(a, b, c):
    """First boundary of a quadratic feasible at zero, including linear/concave cases."""
    disc = b * b - 4 * a * c
    den = b + np.sqrt(np.maximum(disc, 0.))
    root = np.full_like(c, np.inf)
    np.divide(-2 * c, den, out=root, where=(disc >= 0) & (den > 1e-14))
    return np.where(c >= 0, 0., root)


def speed_profile(pts: np.ndarray, v_max: float = 10.0, a_lat: Optional[float] = None,
                  a_acc: Optional[float] = None, a_brake: Optional[float] = None,
                  v_min: float = 1.0, *, mu: Optional[float] = 1.0489,
                  mu_front: Optional[float] = None, mu_rear: Optional[float] = None,
                  vehicle=None, speed_ceiling: Optional[np.ndarray] = None) -> np.ndarray:
    """Closed-lap profile with motor, power and *both axle* friction constraints.

    ``mu_front`` and ``mu_rear`` are absolute coefficients; absent values use ``mu``
    times the vehicle's axle scales. Defaults use the full VehicleParams axle limits. Explicit positional acceleration
    values remain upper bounds. ``v_min`` is retained for source compatibility, but
    never raises a speed above its physical limit. The original algorithm is available
    as ``_legacy_speed_profile`` for reproducing historical experiments.

    On each edge, ax=(v[j]^2-v[i]^2)/(2 ds). Tire longitudinal demand also includes
    the vehicle's rolling/aero resistance c_roll+c_drag*v^2. Quasi-steady yaw balance allocates
    lateral force/mass as ay*lr/L at the front and ay*lf/L at the rear. Normal loads
    include longitudinal transfer, g*lr/L-h*ax/L and g*lf/L+h*ax/L. Each axle must
    satisfy Fx^2+Fy^2 <= (mu*Fz)^2 at *both* edge endpoints. Drive/brake force uses
    the configured drivetrain split (4WD by default), never an assumed rear-only
    drive. Motor/current limits are separate caps, not ellipse multipliers.

    The quadratic edge boundaries are solved analytically in squared speed. Cyclic
    relaxation only decreases speeds until every edge is feasible, including the
    lap seam; it does not depend on an arbitrary three-pass cutoff or starting index.
    This is a spatial quasi-steady model, not a dynamic lap-time guarantee.
    """
    from .params import VehicleParams
    vehicle = vehicle or VehicleParams()
    mu = vehicle.mu if mu is None else float(mu)
    muf = mu * vehicle.mu_f_scale if mu_front is None else float(mu_front)
    mur = mu * vehicle.mu_r_scale if mu_rear is None else float(mu_rear)
    a_lat = 9.81 * min(muf, mur) if a_lat is None else float(a_lat)
    a_acc = vehicle.a_max if a_acc is None else float(a_acc)
    a_brake = vehicle.a_brake if a_brake is None else float(a_brake)
    limits = np.array([v_max, a_lat, a_acc, a_brake, muf, mur, vehicle.v_switch,
                       vehicle.v_max, vehicle.a_max, vehicle.a_brake], dtype=float)
    if not np.isfinite(limits).all() or np.min(limits) <= 0:
        raise ValueError("speed, acceleration and friction limits must be positive")
    v_max = min(vehicle.v_max, float(v_max))
    a_acc, a_brake = min(vehicle.a_max, a_acc), min(vehicle.a_brake, a_brake)
    xy = np.asarray(pts, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
        raise ValueError("profile needs at least three finite 2D path points")
    kap = np.abs(curvature(xy))
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    if np.min(ds) <= 1e-9 or not np.isfinite(kap).all():
        raise ValueError("profile needs distinct adjacent path points")
    lateral = min(a_lat, 9.81 * min(muf, mur))
    u = np.minimum(v_max ** 2, lateral / np.maximum(kap, 1e-12))
    length = vehicle.lf + vehicle.lr
    steer = np.arctan(length * curvature(xy))
    steering_speed = vehicle.sv_max * ds / np.maximum(np.abs(np.roll(steer, -1) - steer), 1e-12)
    # Limiting both endpoint speeds bounds the segment's average steering rate too.
    u = np.minimum(u, np.minimum(steering_speed, np.roll(steering_speed, 1)) ** 2)
    split = vehicle.drive_split_r if vehicle.wheel_model else 1.
    if not 0 <= split <= 1:
        raise ValueError("drive_split_r must be between zero and one")
    axles = [(1 - split, vehicle.lr / length, -vehicle.h / length, muf),
             (split, vehicle.lf / length, vehicle.h / length, mur)]
    roll, drag = vehicle.c_roll, vehicle.c_drag
    if roll < 0 or drag < 0 or roll >= a_acc:
        raise ValueError("profile requires nonnegative resistance below the motor limit")
    for share, load, _, grip in axles:
        # Constant-speed cornering still consumes longitudinal tire force to overcome drag.
        aa = (share * drag) ** 2 + (load * kap) ** 2
        bb = np.full_like(u, 2 * share ** 2 * roll * drag)
        cc = np.full_like(u, (share * roll) ** 2 - (grip * 9.81 * load) ** 2)
        u = np.minimum(u, _first_positive_root(aa, bb, cc))
    # The motor must at least sustain the speed against resistance on a long straight.
    top_low, top_high = 0., float(v_max ** 2)
    for _ in range(40):
        middle = .5 * (top_low + top_high)
        motor = a_acc * min(1., vehicle.v_switch / max(np.sqrt(middle), 1e-9))
        if roll + drag * middle <= motor:
            top_low = middle
        else:
            top_high = middle
    u = np.minimum(u, top_low)
    if speed_ceiling is not None:
        ceiling = np.asarray(speed_ceiling, dtype=float)
        if ceiling.shape != u.shape or not np.isfinite(ceiling).all() or np.min(ceiling) <= 0:
            raise ValueError("speed ceiling must be positive, finite and match the path")
        u = np.minimum(u, ceiling ** 2)
    if np.min(u) <= 0:
        raise ValueError("available tire grip cannot sustain forward motion against resistance")

    def edge_limit(low, k_low, k_high, sign):
        # Unknown high speed squared is low+d. Both tire conditions are quadratic
        # in d; the zero-speed-change state is feasible whenever this is an uphill
        # edge in squared speed. Otherwise returning low is already a harmless cap.
        h = 2 * ds
        resistance = roll + drag * low
        delta = (h * np.maximum(a_acc - resistance, 0.) / (1 + h * drag) if sign > 0
                 else h * (a_brake + resistance))
        if sign > 0:
            # The faster endpoint requires the most motor force and has least power authority.
            power = a_acc * vehicle.v_switch
            for _ in range(7):
                speed = np.sqrt(low + delta)
                force = delta / h + resistance + drag * delta
                step = (force * speed - power) / ((1 / h + drag) * speed + .5 * force / speed)
                delta = np.minimum(delta, np.maximum(0., delta - step))
        for share, load, transfer, grip in axles:
            for k, varies in [(k_low, False), (k_high, True)]:
                fx0 = share * resistance
                fx1 = share * (sign / h + (drag if varies else 0.))
                fy0 = load * k * low
                fy1 = load * k if varies else 0.
                fz1 = transfer * sign / h
                aa = fx1 ** 2 + fy1 ** 2 - (grip * fz1) ** 2
                bb = 2 * (fx0 * fx1 + fy0 * fy1 - grip ** 2 * 9.81 * load * fz1)
                cc = fx0 ** 2 + fy0 ** 2 - (grip * 9.81 * load) ** 2
                delta = np.minimum(delta, _first_positive_root(aa, bb, cc))
            if sign * transfer < 0:
                delta = np.minimum(delta, h * 9.81 * load / abs(transfer))
        return low + np.maximum(delta, 0.) * (1 - 1e-12)

    kp_next = np.roll(kap, -1)
    for _ in range(4 * len(u) + 32):
        previous = u.copy()
        u = np.minimum(u, np.roll(edge_limit(u, kap, kp_next, 1), 1))
        u = np.minimum(u, edge_limit(np.roll(u, -1), kp_next, kap, -1))
        if np.max(previous - u) < 1e-10:
            break
    else:
        raise RuntimeError("cyclic speed constraints did not converge")
    return np.sqrt(np.maximum(u, 0.))


def _refine_lap_time(seed: np.ndarray, track: Track, veh_width: float, margin: float,
                     profile_kw: dict, width_cap: Optional[float] = None,
                     kappa_max: Optional[float] = None, max_seconds: Optional[float] = None,
                     max_evaluations: int = 250, *, return_result: bool = False):
    """Joint path/speed minimum-time solve; no nonconverged seed fallback.

    ``kappa_max`` is retained for private-call compatibility; the solve enforces
    the actual vehicle steering bound instead of the former arbitrary 1.1 cap.
    ``max_evaluations`` is the solver iteration budget, not a feasible-iterate budget.
    """
    from .minimum_time import solve_minimum_time
    result = solve_minimum_time(seed, track, veh_width, margin, profile_kw, width_cap,
                               max_seconds=max_seconds, maxiter=max_evaluations)
    return result if return_result else result.xy


class RacelineCacheMiss(RuntimeError):
    """The line asked for is not in the cache and the caller asked not to build it.

    A `RuntimeError` because that is what the `F1SIM_RACELINE_CACHE_ONLY` path raised before this
    class existed, and a caller that caught that must keep catching it.
    """


#: `VehicleParams` fields added *after* the raceline cache key was defined, none of which the line
#: reads. They are kept out of the key so that adding one does not orphan every cached line.
#:
#: Why this has to exist: the key used to be `repr(VehicleParams())`, all of it, so the day a
#: suspension-noise parameter (`road_tilt_v`) was added every one of the 738 cached racelines stopped
#: matching and each track would have rebuilt -- twenty to seventy minutes apiece -- for a field
#: `Raceline.build` never touches. The line reads sixteen vehicle fields (mass properties, grip,
#: drag, the actuator limits); switching the key to exactly those would be the principled form, and
#: would itself invalidate all 738 once, which is why it is not done here.
#:
#: A new field goes here if and only if the raceline does not depend on it.
_ADDED_AFTER_LINE_KEY = frozenset({"road_tilt_v"})

#: `VehicleParams` fields the line does not read whose *default changed* after the key was defined,
#: each written into the key at the value it had then. Same reason as above: `road_tilt` went
#: 0.017 -> 0 on 2026-09-21 (the attitude comes from the dynamics alone now), and with it in the key
#: as-is all 738 cached lines would have rebuilt for a suspension-noise knob `Raceline.build` never
#: touches.
_FROZEN_IN_LINE_KEY = {"road_tilt": 0.017}


class _LineKeyVehicle:
    """Stands in for a `VehicleParams` inside the cache key, with the same `repr` it had when the
    key was defined -- dataclass format, declaration order -- minus `_ADDED_AFTER_LINE_KEY`."""

    __slots__ = ("_v",)

    def __init__(self, vehicle):
        self._v = vehicle

    def __repr__(self) -> str:
        import dataclasses
        v = self._v
        body = ", ".join(f"{fl.name}={_FROZEN_IN_LINE_KEY.get(fl.name, getattr(v, fl.name))!r}"
                         for fl in dataclasses.fields(v) if fl.name not in _ADDED_AFTER_LINE_KEY)
        return f"{type(v).__name__}({body})"

    def __lt__(self, other):                  # never compared: "vehicle" is a unique key
        return NotImplemented


# --------------------------------------------------------------------------- container
@dataclass
class Raceline:
    xy: np.ndarray        # (N,2)
    v: np.ndarray         # (N,) target speed
    kappa: np.ndarray     # (N,)
    s: np.ndarray         # (N,) arclength at each point
    length: float
    lap_time: float       # sum 2 ds / (v_i + v_j), constant-acceleration estimate
    optimization: Optional[dict] = None

    @staticmethod
    def build(track: Track, veh_width: float = 0.31, margin: float = 0.40, v_max: float = 10.0,
              a_lat: Optional[float] = None, a_acc: Optional[float] = None,
              a_brake: Optional[float] = None, iters: int = 30,
              smooth: float = 0.5, width_cap_ratio: float = 0.8, *,
              mu: float = 1.0489, mu_front: Optional[float] = None,
              mu_rear: Optional[float] = None, vehicle=None,
              optimize_lap_time: bool = True,
              objective: Optional[str] = None) -> "Raceline":
        """margin: free space kept between the car's side and the boundary (0.40 m: the pure-pursuit
        teacher cuts inside the line by up to ~0.15 m at speed, and duct hoses are soft targets anyway).
        width_cap_ratio: a side is never wider than this fraction of the median total lane width,
        so openings into side rooms / pit areas of SLAM maps do not pull the line off the lane.

        `objective` is the older spelling of the same choice, kept because it is what
        `--raceline-objective` passes and what evaluation scripts written against it send:
        "min_time" is `optimize_lap_time=True`, "min_curvature" is False. Without it those callers
        raised TypeError -- and only when the raceline cache missed, so training that reused a
        prebuilt line ran fine and the evaluation afterwards did not."""
        if objective is not None:
            if objective not in ("min_time", "min_curvature"):
                raise ValueError(f"raceline objective must be 'min_time' or 'min_curvature', "
                                 f"not {objective!r}")
            optimize_lap_time = (objective == "min_time")
        track = track.for_planning()
        if track.centerline is None:
            raise ValueError("track needs a centerline")
        from .params import VehicleParams
        vehicle = vehicle or VehicleParams()
        kappa_max = np.tan(vehicle.s_max) / (vehicle.lf + vehicle.lr)
        from .planning_seed import obstacle_aware_seed
        routed = obstacle_aware_seed(track, clearance=veh_width / 2)
        c = resample_closed(routed, len(track.centerline))
        wl, wr = track_widths(track, c)
        cap = width_cap_ratio * float(np.median(wl + wr)) if width_cap_ratio else None
        xy = min_curvature_raceline(c, veh_width=veh_width, margin=margin, iters=iters, smooth=smooth,
                                    track=track, width_cap=cap, kappa_max=kappa_max, margin_min=0.)
        profile_kw = dict(v_max=v_max, a_lat=a_lat, a_acc=a_acc, a_brake=a_brake,
                          mu=mu, mu_front=mu_front, mu_rear=mu_rear, vehicle=vehicle)
        if optimize_lap_time:
            result = _refine_lap_time(xy, track, veh_width, margin, profile_kw, width_cap=cap,
                                      return_result=True)
            line = Raceline.from_xy(result.xy, result.v)
            line.optimization = result.diagnostics
            return line
        v = speed_profile(xy, **profile_kw)
        line = Raceline.from_xy(xy, v)
        line.optimization = {"status": "initialization-only", "model": "minimum-curvature"}
        return line

    @staticmethod
    def build_cached(track: Track, cache_dir: Optional[str] = None, *, cache_only: bool = False,
                     **kw) -> "Raceline":
        """Raceline.build with an on-disk cache keyed by the track's occupancy + parameters.

        `cache_only`: raise `RacelineCacheMiss` on a miss instead of spending twenty minutes to an
        hour in the minimum-time solver. For a caller that only wants to *draw* the line, that is
        the right answer -- a line nobody drives on is not worth an hour. Keyword-only and outside
        `**kw` on purpose: `**kw` is the cache key, and a flag about how to look something up must
        not change what is being looked up.
        """
        import hashlib, os
        track = track.for_planning()
        cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "f1sim", "racelines")
        os.makedirs(cache_dir, exist_ok=True)
        import inspect
        params = {k: v.default for k, v in inspect.signature(Raceline.build).parameters.items() if k != "track"}
        params.update(kw)                                  # key includes the *effective* parameters, defaults too
        # `objective` is an alias `build` resolves into `optimize_lap_time` before doing anything,
        # so `None` is not a choice -- it is the absence of one, and the line it produces is the
        # line `optimize_lap_time` alone already keys. Keeping it in the key would make every
        # raceline ever cached miss on the day the parameter was restored, and rebuild each one
        # into a file byte-identical to the one beside it.
        if params.get("objective") is None:
            params.pop("objective", None)
        from .params import VehicleParams
        params["vehicle"] = _LineKeyVehicle(params["vehicle"] or VehicleParams())
        cl = b"" if track.centerline is None else np.asarray(track.centerline, dtype=np.float32).tobytes()
        geometry = repr((track.occupancy.shape, track.resolution, tuple(track.origin))).encode()
        h = hashlib.md5(np.packbits(track.occupancy).tobytes() + cl + geometry
                        + repr(sorted(params.items())).encode() + b"rl10-joint-minimum-time-swept-props-v1").hexdigest()[:12]
        path = os.path.join(cache_dir, f"{track.name}_{h}.csv")
        if os.path.exists(path):
            return Raceline.load(path)
        if cache_only:
            raise RacelineCacheMiss(f"raceline for {track.name!r} is not cached with "
                                    f"{ {k: v for k, v in kw.items()} or 'the defaults'} (key {h})")
        Raceline._announce_cache_miss(track.name, cache_dir, h, kw, params)
        import time as _time
        t0 = _time.monotonic()
        rl = Raceline.build(track, **kw)
        rl.save(path)
        report(f"{track.name}: {(_time.monotonic() - t0) / 60:.1f}분 만에 완성, 캐시에 저장",
               force=True)
        return rl

    #: Set `F1SIM_RACELINE_CACHE_ONLY=1` and a cache miss raises instead of spending twenty minutes
    #: to an hour in the minimum-time solver. For anything meant to be quick -- an evaluation, a
    #: test, a console session -- that is the behaviour you want: the answer "this needs a build"
    #: in a second beats the same answer after forty minutes of silence.
    CACHE_ONLY_ENV = "F1SIM_RACELINE_CACHE_ONLY"

    @staticmethod
    def _announce_cache_miss(name: str, cache_dir: str, h: str, kw: dict, params: dict) -> None:
        """Say that a build is starting, and say *why* this one and not the last one.

        The cache key is every parameter of `build`, so one different `--teacher-a-lat` or
        `--raceline-margin` is a fresh twenty-to-seventy-minute optimisation -- and until this
        existed it happened in total silence, which is why the same command felt instant one day
        and hung the next. The parameters that were passed are printed because they are the ones
        that moved; the variants already on disk are counted because "14 of this track, none of
        them yours" is the sentence that explains it.
        """
        import os
        try:
            have = sorted(f for f in os.listdir(cache_dir) if f.startswith(name + "_"))
        except OSError:
            have = []
        if os.environ.get(Raceline.CACHE_ONLY_ENV, "") not in ("", "0"):
            raise RacelineCacheMiss(
                f"raceline cache miss for {name!r} (key {h}) and {Raceline.CACHE_ONLY_ENV} is set. "
                f"{len(have)} variant(s) of this track are cached, none with these parameters: "
                f"{ {k: v for k, v in kw.items()} or 'the defaults'}. Build it once (it takes "
                f"20-70 min) or pass the parameters the run used -- the cache key is every "
                f"argument of Raceline.build, so --teacher-a-lat / --teacher-a-acc / "
                f"--teacher-a-brake / --raceline-margin / --raceline-objective all change it.")
        report(f"{name}: 캐시에 없어 최소시간 라인을 새로 만듭니다 (20~70분, 싱글스레드). "
               f"파라미터 { {k: v for k, v in kw.items()} or '기본값'} — 이 맵의 다른 변형 "
               f"{len(have)}개는 이미 캐시에 있으니, 즉시 끝날 줄 아셨다면 그 중 하나와 "
               f"파라미터가 다릅니다.", force=True)

    @staticmethod
    def from_xy(xy: np.ndarray, v: np.ndarray) -> "Raceline":
        ds = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)[:-1]])
        return Raceline(xy, v, curvature(xy), s, float(ds.sum()),
                        float((2 * ds / (v + np.roll(v, -1))).sum()))

    def save(self, path: str):
        import json
        header = "x_m,y_m,v_mps,kappa"
        if self.optimization is not None:
            header += "\noptimization: " + json.dumps(self.optimization, sort_keys=True, allow_nan=False)
        np.savetxt(path, np.column_stack([self.xy, self.v, self.kappa]), delimiter=",",
                   header=header, comments="# ")

    @staticmethod
    def load(path: str) -> "Raceline":
        import json
        d = np.genfromtxt(path, delimiter=",", comments="#")
        line = Raceline.from_xy(d[:, :2], d[:, 2])
        with open(path) as stream:
            for row in stream:
                if not row.startswith("#"):
                    break
                if row.startswith("# optimization: "):
                    line.optimization = json.loads(row[len("# optimization: "):])
        return line
