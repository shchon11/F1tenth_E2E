"""An occupancy map from a drive's LiDAR scans and wheel odometry, with no SLAM package installed.

The car's recordings carry `/scan` (40 Hz, 270 deg) and `/odom` (VESC wheel speed + steering
model), and nothing on this machine turns that into a map: no slam_toolbox, no cartographer. A
small F1TENTH venue does not need either. The hall is a few tens of metres across and the LiDAR
reaches 30 m, so almost every scan sees walls the map already holds, and scan-to-map matching
against a growing map (the Hector-SLAM idea) keeps the pose without a loop-closure back end.

How a scan is placed
--------------------
1. **Prior**: the previous corrected pose moved by the odometry delta between the two scans.
2. **De-skew**: a scan takes ~19 ms to sweep; at 9 m/s the car moves 0.17 m in that time, so each
   beam is moved back to the scan's stamp with the odometry speed and yaw rate.
3. **Match**: Gauss-Newton on the distance from every scan point to the nearest mapped surface
   (a truncated distance field of the cells hit so far, bilinear so it has a gradient), with a
   Cauchy weight so a passing car or a person is not pulled onto a wall.
4. **Fallback**: when fewer than a third of the points land on a surface after Gauss-Newton (wheel
   spin, a crash, a bad odometry step), a brute-force search over +-0.5 m / +-0.25 rad picks the best
   start and Gauss-Newton runs again from there.
5. **Grow**: every ~0.15 m of travel the scan's endpoints are added to the hit grid; the distance
   field is rebuilt once enough new cells have appeared (after the first lap almost none do).

The final grid is ray cast from the corrected keyframe poses: free along each beam, occupied at
its end, so a car or a person that stood somewhere for a moment is cleared by the views that saw
through that place later. Cells are written the way map_server reads them: 0 occupied, 254 free,
205 unknown, 0.05 m.

Pure numpy / scipy, no ROS: `bag_map.py` reads the bag and calls this.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------- inputs
@dataclass
class Scans:
    t: np.ndarray                 # (K,) stamp of each scan [s] (time of its first beam)
    ranges: np.ndarray            # (K, N) float32 [m]; inf / nan / 0 = no return
    angle_min: float
    angle_inc: float
    time_inc: float = 0.0         # [s] between consecutive beams; 0 = no de-skew
    range_min: float = 0.02
    range_max: float = 30.0
    laser: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # sensor pose on the car (x, y, yaw)


@dataclass
class Odom:
    t: np.ndarray                 # (M,) [s]
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    v: np.ndarray                 # forward speed [m/s]
    w: np.ndarray                 # yaw rate [rad/s]


# ---------------------------------------------------------------- 2-D pose algebra
def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def compose(a, b):
    """a then b: poses (..., 3) as (x, y, yaw)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([a[..., 0] + c * b[..., 0] - s * b[..., 1],
                     a[..., 1] + s * b[..., 0] + c * b[..., 1], wrap(a[..., 2] + b[..., 2])], -1)


def inverse(a):
    a = np.asarray(a, float)
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack([-c * a[..., 0] - s * a[..., 1], s * a[..., 0] - c * a[..., 1], wrap(-a[..., 2])], -1)


def transform(pose, pts, scale: float = 1.0):
    """Points (M, 2) from a frame at `pose` into the world."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    q = np.asarray(pts, float) * scale
    return np.stack([pose[0] + c * q[:, 0] - s * q[:, 1], pose[1] + s * q[:, 0] + c * q[:, 1]], 1)


def odom_at(odom: Odom, t: np.ndarray):
    """(poses (K, 3), v (K,), w (K,)) interpolated at times `t`."""
    yaw = np.unwrap(odom.yaw)
    pose = np.stack([np.interp(t, odom.t, odom.x), np.interp(t, odom.t, odom.y),
                     wrap(np.interp(t, odom.t, yaw))], 1)
    return pose, np.interp(t, odom.t, odom.v), np.interp(t, odom.t, odom.w)


def car_points(scans: Scans, k: int, beams: np.ndarray, v: float = 0.0, w: float = 0.0,
               max_range: float = 15.0) -> np.ndarray:
    """Scan k's returns in the car frame at the scan's stamp (M, 2), de-skewed with (v, w)."""
    r = scans.ranges[k, beams].astype(np.float64)
    ok = np.isfinite(r) & (r > max(scans.range_min, 0.05)) & (r < min(scans.range_max, max_range))
    b, r = beams[ok], r[ok]
    lx, ly, lyaw = scans.laser
    a = scans.angle_min + b * scans.angle_inc + lyaw
    px, py = lx + r * np.cos(a), ly + r * np.sin(a)
    if scans.time_inc > 0 and (abs(v) > 1e-3 or abs(w) > 1e-3):
        tau = b * scans.time_inc
        dth = w * tau
        dx, dy = v * tau * np.cos(0.5 * dth), v * tau * np.sin(0.5 * dth)
        c, s = np.cos(dth), np.sin(dth)
        px, py = dx + c * px - s * py, dy + s * px + c * py
    return np.stack([px, py], 1)


# ---------------------------------------------------------------- distance field
class Field:
    """Truncated distance [m] to the nearest surface cell, sampled bilinearly with its gradient."""

    def __init__(self, surface: np.ndarray, x0: float, y0: float, res: float, d_max: float = 1.0):
        self.res, self.d_max = float(res), float(d_max)
        rows, cols = np.nonzero(surface)
        pad = int(math.ceil(d_max / res)) + 2
        if rows.size == 0:
            self.D = np.full((2, 2), d_max, np.float32)
            self.x0, self.y0 = x0, y0
            return
        r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, surface.shape[0])
        c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, surface.shape[1])
        sub = surface[r0:r1, c0:c1]
        self.D = np.minimum(ndimage.distance_transform_edt(~sub) * res, d_max).astype(np.float32)
        self.x0, self.y0 = x0 + c0 * res, y0 + r0 * res

    def sample(self, x, y):
        H, W = self.D.shape
        u = (x - self.x0) / self.res - 0.5
        v = (y - self.y0) / self.res - 0.5
        j = np.floor(u).astype(np.int64)
        i = np.floor(v).astype(np.int64)
        fu, fv = u - j, v - i
        inside = (j >= 0) & (j < W - 1) & (i >= 0) & (i < H - 1)
        jc, ic = np.clip(j, 0, W - 2), np.clip(i, 0, H - 2)
        D = self.D
        d00, d01, d10, d11 = D[ic, jc], D[ic, jc + 1], D[ic + 1, jc], D[ic + 1, jc + 1]
        d = (1 - fv) * ((1 - fu) * d00 + fu * d01) + fv * ((1 - fu) * d10 + fu * d11)
        gx = ((1 - fv) * (d01 - d00) + fv * (d11 - d10)) / self.res
        gy = ((1 - fu) * (d10 - d00) + fu * (d11 - d01)) / self.res
        d = np.where(inside, d, self.d_max)
        gx, gy = np.where(inside, gx, 0.0), np.where(inside, gy, 0.0)
        return d, gx, gy

    def nearest(self, x, y):
        H, W = self.D.shape
        j = np.floor((x - self.x0) / self.res).astype(np.int64)
        i = np.floor((y - self.y0) / self.res).astype(np.int64)
        ok = (j >= 0) & (j < W) & (i >= 0) & (i < H)
        return np.where(ok, self.D[np.clip(i, 0, H - 1), np.clip(j, 0, W - 1)], self.d_max)


@dataclass
class MatchInfo:
    inliers: float        # share of the points within `INLIER_M` of a surface after the match
    rms: float            # [m] over the inliers
    iters: int
    scale: float = 1.0


INLIER_M = 0.10


def match(fld: Field, pts: np.ndarray, pose, iters: int = 15, cauchy: float = 0.15,
          fit_scale: bool = False, scale0: float = 1.0) -> Tuple[np.ndarray, MatchInfo]:
    """Gauss-Newton of a point set (car frame) onto a distance field. `fit_scale` also fits one
    factor on the points' distances from the car (the map's scale against the LiDAR's)."""
    x, y, th = float(pose[0]), float(pose[1]), float(pose[2])
    s = float(scale0)
    px, py = pts[:, 0], pts[:, 1]
    n = 4 if fit_scale else 3
    it = 0
    for it in range(1, iters + 1):
        c, sn = math.cos(th), math.sin(th)
        qx, qy = s * px, s * py
        wx, wy = x + c * qx - sn * qy, y + sn * qx + c * qy
        d, gx, gy = fld.sample(wx, wy)
        use = d < 0.95 * fld.d_max
        if use.sum() < 10:
            break
        wgt = use / (1.0 + (d / cauchy) ** 2)
        J = np.empty((len(d), n))
        J[:, 0], J[:, 1] = gx, gy
        J[:, 2] = gx * (-sn * qx - c * qy) + gy * (c * qx - sn * qy)
        if fit_scale:
            J[:, 3] = gx * (c * px - sn * py) + gy * (sn * px + c * py)
        JW = J * wgt[:, None]
        Hm = JW.T @ J
        g = JW.T @ d
        Hm += np.eye(n) * (1e-6 * np.trace(Hm) + 1e-9)
        try:
            delta = -np.linalg.solve(Hm, g)
        except np.linalg.LinAlgError:
            break
        step = math.hypot(delta[0], delta[1])
        if step > 0.3:
            delta[:2] *= 0.3 / step
        delta[2] = float(np.clip(delta[2], -0.15, 0.15))
        x, y, th = x + delta[0], y + delta[1], th + delta[2]
        if fit_scale:
            s = float(np.clip(s + np.clip(delta[3], -0.05, 0.05), 0.5, 1.5))
        if abs(delta[0]) + abs(delta[1]) < 1e-4 and abs(delta[2]) < 1e-4:
            break
    c, sn = math.cos(th), math.sin(th)
    qx, qy = s * px, s * py
    d, _, _ = fld.sample(x + c * qx - sn * qy, y + sn * qx + c * qy)
    inl = d < INLIER_M
    rms = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("nan")
    return np.array([x, y, float(wrap(th))]), MatchInfo(float(inl.mean()) if len(d) else 0.0, rms, it, s)


def search(fld: Field, pts: np.ndarray, pose, xy_win: float = 0.5, th_win: float = 0.25,
           th_step: float = 0.02, sigma: float = 0.08, stride: int = 1) -> np.ndarray:
    """Brute-force best pose around `pose` (every `stride` cells in x/y, `th_step` in yaw): the
    fallback for a scan Gauss-Newton could not place."""
    res = fld.res
    n = int(round(xy_win / res))
    sh = np.arange(-n, n + 1, max(1, int(stride)))
    best, best_pose = -1.0, np.asarray(pose, float)
    for dth in np.arange(-th_win, th_win + 1e-9, th_step):
        th = pose[2] + dth
        w = transform((pose[0], pose[1], th), pts)
        j = np.floor((w[:, 0] - fld.x0) / res).astype(np.int64)
        i = np.floor((w[:, 1] - fld.y0) / res).astype(np.int64)
        H, W = fld.D.shape
        jj = j[None, None, :] + sh[None, :, None]            # (1, S, M)
        ii = i[None, None, :] + sh[:, None, None]             # (S, 1, M)
        ok = (jj >= 0) & (jj < W) & (ii >= 0) & (ii < H)
        d = np.where(ok, fld.D[np.clip(ii, 0, H - 1), np.clip(jj, 0, W - 1)], fld.d_max)
        score = np.exp(-(d / sigma) ** 2).sum(-1)              # (S, S): rows = dy, cols = dx
        k = int(np.argmax(score))
        if score.flat[k] > best:
            best = float(score.flat[k])
            a, b = divmod(k, score.shape[1])
            best_pose = np.array([pose[0] + sh[b] * res, pose[1] + sh[a] * res, th])
    return best_pose


# ---------------------------------------------------------------- the mapper
@dataclass
class MapResult:
    poses: np.ndarray             # (K, 3) corrected car pose at every scan of the stretch used, map frame
    odom_poses: np.ndarray        # (K, 3) the odometry's own over the same scans
    t: np.ndarray                 # (K,) their stamps
    keyframes: np.ndarray         # indices (into the stretch) the final grid was cast from
    inliers: np.ndarray           # (K,) share of points on a surface after matching
    hit: np.ndarray               # (H, W) beam ends per cell
    free: np.ndarray              # (H, W) beams through per cell
    x0: float
    y0: float
    res: float
    seconds: float
    stats: Dict[str, float] = field(default_factory=dict)
    stretches: List[Dict[str, float]] = field(default_factory=list)

    def image(self, occ_ratio: float = 0.25, min_hits: int = 2, min_free: int = 2):
        """map_server greyscale (row 0 = the top): 0 occupied, 254 free, 205 unknown."""
        h, f = self.hit.astype(np.float64), self.free.astype(np.float64)
        occ = (h >= min_hits) & (h >= occ_ratio * (h + f))
        free = ~occ & (f >= min_free)
        img = np.full(self.hit.shape, 205, np.uint8)
        img[free] = 254
        img[occ] = 0
        return np.flipud(img)


def _blocked(pts: np.ndarray, n_beams: int) -> bool:
    """The scan sees almost nothing beyond a metre: the car is against something (a crash, a hand
    over the sensor). Nothing in it can be matched, and the wheels may still be spinning."""
    return len(pts) < 20 or (np.hypot(pts[:, 0], pts[:, 1]) > 1.0).sum() < 0.15 * n_beams


def _track(scans: Scans, op, ov, ow, k0: int, res: float, max_range: float, match_step: int,
           kf_dist: float, kf_ang: float, lost_scans: int, stop_laps: float, log, t_log) -> Dict:
    """Scan-to-map tracking from scan `k0` on a fresh map, until `stop_laps` laps are driven, the
    scans run out, or the car is lost (a blocked scan, or `lost_scans` scans in a row that do not
    match the map)."""
    K, N = scans.ranges.shape
    beams_m = np.arange(0, N, match_step)
    beams_all = np.arange(0, N, 1)
    slack = max_range + 8.0                      # the odometry's own extent, the LiDAR's reach and its drift
    x0 = float(op[k0:, 0].min() - slack); y0 = float(op[k0:, 1].min() - slack)
    W = int(math.ceil((op[k0:, 0].max() + slack - x0) / res)); H = int(math.ceil((op[k0:, 1].max() + slack - y0) / res))
    hits = np.zeros((H, W), np.uint16)

    def add_hits(world_pts):
        j = np.floor((world_pts[:, 0] - x0) / res).astype(np.int64)
        i = np.floor((world_pts[:, 1] - y0) / res).astype(np.int64)
        ok = (j >= 0) & (j < W) & (i >= 0) & (i < H)
        i, j = i[ok], j[ok]
        before = hits[i, j] >= 2
        np.add.at(hits, (i, j), 1)
        return int(((hits[i, j] >= 2) & ~before).sum())

    poses = np.zeros((K, 3)); inl = np.zeros(K)
    poses[k0] = op[k0]; inl[k0] = 1.0
    add_hits(transform(poses[k0], car_points(scans, k0, beams_all, ov[k0], ow[k0], max_range)))
    fld = Field(hits >= 1, x0, y0, res)
    keyframes = [k0]
    last_kf = poses[k0].copy()
    new_cells = kf_since = fallbacks = n_refresh = 0
    bad_run, end, why = 0, K, "끝까지"
    turned = 0.0                                  # heading summed over the stretch: one lap of a loop is 2 pi
    for k in range(k0 + 1, K):
        pts = car_points(scans, k, beams_m, ov[k], ow[k], max_range)
        if _blocked(pts, len(beams_m)):
            end, why = k, f"{scans.t[k] - scans.t[0]:.1f} s 에 스캔이 막힘 (충돌 등)"
            break
        rel = compose(inverse(op[k - 1]), op[k])
        pred = compose(poses[k - 1], rel)
        pose, info = match(fld, pts, pred)
        if info.inliers < 0.9:
            # the wheel odometry is the weak part (a slide, wheels spinning, the car carried): also
            # try the scan matcher's own last motion and no motion at all
            vel = compose(inverse(poses[k - 2]), poses[k - 1]) if k - 2 >= k0 else np.zeros(3)
            for alt in (compose(poses[k - 1], vel), poses[k - 1]):
                p2, i2 = match(fld, pts, alt)
                if i2.inliers > info.inliers:
                    pose, info = p2, i2
        if info.inliers < 0.6:
            fallbacks += 1
            p2, i2 = match(fld, pts, search(fld, pts, pred))
            if i2.inliers > info.inliers:
                pose, info = p2, i2
        bad_run = bad_run + 1 if info.inliers < 0.6 else 0
        if bad_run >= lost_scans:
            end, why = k - bad_run + 1, f"{scans.t[k] - scans.t[0]:.1f} s 에 위치를 잃음"
            break
        poses[k], inl[k] = pose, info.inliers
        turned += float(wrap(pose[2] - poses[k - 1, 2]))
        if abs(turned) >= 2 * np.pi * stop_laps:
            end, why = k + 1, f"{stop_laps:g}바퀴를 채움"
            break
        # the map grows only from scans it already agrees with, once it has something to agree with:
        # one mis-placed scan added at full weight is a rotated copy of the track that the next scans
        # then match perfectly
        ok = info.inliers >= (0.8 if len(keyframes) >= 40 else 0.5)
        moved = math.hypot(*(pose[:2] - last_kf[:2])) > kf_dist or abs(wrap(pose[2] - last_kf[2])) > kf_ang
        if moved and ok:
            new_cells += add_hits(transform(pose, car_points(scans, k, beams_all, ov[k], ow[k], max_range)))
            keyframes.append(k)
            last_kf = pose.copy()
            kf_since += 1
            if len(keyframes) < 10 or new_cells >= 25 or kf_since >= 60:
                fld = Field(hits >= (1 if len(keyframes) < 10 else 2), x0, y0, res)
                new_cells, kf_since = 0, 0
                n_refresh += 1
        if time.time() - t_log[0] > 15:
            t_log[0] = time.time()
            log(f"  스캔 {k}/{K} 배치, 키프레임 {len(keyframes)}, 표면에 닿은 점 중앙값 {np.median(inl[k0 + 1:k + 1]):.0%}")
    dt = max(float(np.median(np.diff(scans.t))), 1e-3)
    if end < K and not why.endswith("채움"):
        # a stretch that ended in an event: drop its last second, the scans the event was building up in
        end = max(k0 + 1, end - int(round(1.0 / dt)))
        keyframes = [k for k in keyframes if k < end]
    every = max(1, int(round(0.25 / dt)))
    path = float(np.linalg.norm(np.diff(poses[k0:end:every, :2], axis=0), axis=1).sum())
    return dict(k0=k0, end=end, why=why, poses=poses[k0:end], inliers=inl[k0:end], keyframes=np.array(keyframes) - k0,
                path=path, fallbacks=fallbacks, field_rebuilds=n_refresh,
                laps=float(abs(np.sum(wrap(np.diff(poses[k0:end, 2])))) / (2 * np.pi)) if end - k0 > 1 else 0.0)


def close_loops(scans: Scans, poses: np.ndarray, keyframes: np.ndarray, k0: int, ov, ow, *, res: float = 0.05,
                max_range: float = 12.0, match_step: int = 4, min_gap_s: float = 6.0, near_m: float = 1.0,
                sub_s: float = 0.6, log=print) -> Tuple[np.ndarray, Dict[str, float]]:
    """Make the stretch's keyframe poses agree with themselves where the car came back.

    Scan-to-map tracking drifts a few centimetres a lap. Where the drift builds up between two looks
    at the same wall (the two legs of a lap both see the infield), the map keeps both copies --
    doubled walls 10-20 cm apart (competition bag 0826-192819). A 2-D pose graph: one edge between
    consecutive keyframes (what tracking measured), and one edge for every return to a place driven
    more than `min_gap_s` before -- the later keyframe's scan matched to a submap of the scans around
    the earlier one. Least squares (soft L1) then spreads the drift over the lap, and the corrected
    poses replace the tracked ones (scans between keyframes move with the keyframe before them).
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    t0 = time.time()
    N = scans.ranges.shape[1]
    beams = np.arange(0, N, match_step)
    kf = np.asarray(keyframes)
    P = poses[kf].copy()
    T = scans.t[k0 + kf]
    n = len(kf)
    if n < 20:
        return poses, dict(loops=0, seconds=0.0)
    pts = [car_points(scans, k0 + k, beams, ov[k], ow[k], max_range) for k in kf]
    # candidate returns: each keyframe j, the closest earlier keyframe of every earlier pass (heading alike)
    cands = []
    for j in range(0, n, 2):
        d = np.hypot(P[:j, 0] - P[j, 0], P[:j, 1] - P[j, 1])
        ok = (T[j] - T[:j] > min_gap_s) & (d < near_m) & (np.abs(wrap(P[:j, 2] - P[j, 2])) < 0.6)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        runs = np.split(idx, np.nonzero(np.diff(idx) > 3)[0] + 1)          # one per earlier pass
        for r in runs:
            cands.append((int(r[np.argmin(d[r])]), j))
    loops = []
    cache = {}
    cands.sort()                                                           # by the earlier keyframe: submaps reused
    for i, j in cands:
        key = i // 4
        if key not in cache:
            # submap: the keyframes within sub_s of i, in the world frame of the tracked poses
            near = np.nonzero(np.abs(T - T[i]) <= sub_s)[0]
            wp = np.concatenate([transform(P[m], pts[m]) for m in near])
            x0, y0 = wp.min(0) - 1.5
            W, H = (np.ceil((wp.max(0) + 1.5 - np.array([x0, y0])) / res)).astype(int)
            grid = np.zeros((H, W), bool)
            jj = np.floor((wp[:, 0] - x0) / res).astype(int); ii = np.floor((wp[:, 1] - y0) / res).astype(int)
            grid[ii, jj] = True
            cache = {key: Field(grid, x0, y0, res, d_max=0.6)}             # one at a time: they are big
        fld = cache[key]
        pj, info = match(fld, pts[j], P[j], iters=30)
        moved = math.hypot(pj[0] - P[j, 0], pj[1] - P[j, 1])
        if info.inliers >= 0.85 and moved < 0.5:
            loops.append((i, j, compose(inverse(P[i]), pj)))
    if not loops:
        return poses, dict(loops=0, candidates=len(cands), seconds=round(time.time() - t0, 1))
    odo = [(a, a + 1, compose(inverse(P[a]), P[a + 1])) for a in range(n - 1)]
    edges = odo + loops
    ea = np.array([e[0] for e in edges]); eb = np.array([e[1] for e in edges]); ez = np.array([e[2] for e in edges])
    # weights: tracking between neighbours (0.02 m, 0.005 rad), a return matched to a submap (0.03 m, 0.01 rad)
    sw = np.where(np.arange(len(edges)) < len(odo), 1.0, 0.0)[:, None] * np.array([1 / 0.02, 1 / 0.02, 1 / 0.005]) + \
        np.where(np.arange(len(edges)) >= len(odo), 1.0, 0.0)[:, None] * np.array([1 / 0.03, 1 / 0.03, 1 / 0.01])

    def resid(x):
        X = x.reshape(n, 3)
        rel = compose(inverse(X[ea]), X[eb])
        r = rel - ez
        r[:, 2] = wrap(r[:, 2])
        anchor = (X[0] - P[0]) * 1e3
        return np.concatenate([(r * sw).ravel(), anchor])

    m = len(edges)
    S = lil_matrix((3 * m + 3, 3 * n), dtype=np.int8)
    for e in range(m):
        for c in range(3):
            S[3 * e + c, 3 * ea[e]:3 * ea[e] + 3] = 1
            S[3 * e + c, 3 * eb[e]:3 * eb[e] + 3] = 1
    S[3 * m:, 0:3] = 1
    r0 = resid(P.ravel())
    sol = least_squares(resid, P.ravel(), jac_sparsity=S, loss="soft_l1", f_scale=3.0, max_nfev=60, x_scale="jac")
    Pn = sol.x.reshape(n, 3)
    Pn[:, 2] = wrap(Pn[:, 2])
    # every scan moves with the keyframe at or before it
    out = poses.copy()
    owner = np.searchsorted(kf, np.arange(len(poses)), side="right") - 1
    owner = np.clip(owner, 0, n - 1)
    corr = compose(Pn, inverse(P))                                         # world-frame correction per keyframe
    out = compose(corr[owner], poses)
    lr = np.array([np.linalg.norm(((compose(inverse(Pn[i]), Pn[j]) - z) * [1, 1, 0])[:2]) for i, j, z in loops])
    return out, dict(loops=len(loops), candidates=len(cands), moved_max_m=float(np.hypot(*(Pn[:, :2] - P[:, :2]).T).max()),
                     loop_residual_cm=float(100 * np.median(lr)), cost_before=float(0.5 * np.sum(r0 ** 2)),
                     cost_after=float(sol.cost), seconds=round(time.time() - t0, 1))


def build_map(scans: Scans, odom: Odom, *, res: float = 0.05, max_range: float = 15.0,
              match_step: int = 4, kf_dist: float = 0.15, kf_ang: float = 0.08,
              cast_step: int = 2, cast_range: float = 12.0, stop_laps: float = 3.0, min_laps: float = 1.1,
              lost_scans: int = 8, loop_closure: bool = True, log=print) -> MapResult:
    """Place the scans, then ray cast the keyframes into a hit / pass-through count grid.

    The map comes from ONE continuous stretch of clean tracking. A crash is the one thing this does
    not track through -- the scan is blocked by the duct the car is pressed into while the wheels
    spin on, and the car is often carried back by hand -- and carrying on past one put a second,
    rotated copy of the track into the map (competition bag 0827-115713 at 38 s). So tracking stops
    at an event, and at `stop_laps` laps (more adds nothing to a map of a loop). Laps are counted
    by the heading turned: one lap of a loop is 2 pi.

    Which stretch: the first that covers `min_laps` (the start of a recording is its cleanest
    part: the car on the grid, nobody on the track yet; a later stretch begins wherever the car
    was put back after a crash, with people around it). When none does, the stretch with the most
    laps, and `stats['laps']` says how far short it is.
    """
    t_start = time.time()
    K, N = scans.ranges.shape
    op, ov, ow = odom_at(odom, scans.t)
    beams_m = np.arange(0, N, match_step)
    dt = max(float(np.median(np.diff(scans.t))), 1e-3)
    t_log = [time.time()]
    stretches, k0 = [], 0
    while k0 < K - 40:
        st = _track(scans, op, ov, ow, k0, res, max_range, match_step, kf_dist, kf_ang, lost_scans, stop_laps,
                    log, t_log)
        stretches.append(st)
        if st["end"] >= K or st["laps"] >= min_laps:
            break
        k0 = st["end"] + int(round(1.0 / dt))
        while k0 < K and _blocked(car_points(scans, k0, beams_m, ov[k0], ow[k0], max_range), len(beams_m)):
            k0 += 1                                   # past the event: the next scan that sees the hall again
        k0 += int(round(1.0 / dt))                     # and a second to settle
    ok = [z for z in stretches if z["laps"] >= min_laps]
    best = ok[0] if ok else max(stretches, key=lambda z: z["laps"])
    for z in stretches:
        log(f"  구간 {scans.t[z['k0']] - scans.t[0]:.1f}-{scans.t[z['end'] - 1] - scans.t[0]:.1f} s: {z['laps']:.1f}바퀴 "
            f"{z['path']:.0f} m, 끝난 이유: {z['why']}{'  <- 이 구간으로 지도를 만듦' if z is best else ''}")
    if best["laps"] < min_laps:
        log(f"  주의: 깨끗한 구간이 {best['laps']:.1f}바퀴뿐이라 지도에 빠진 곳이 있을 수 있음 (그림을 꼭 볼 것)")
    k0, end = best["k0"], best["end"]
    poses, keyframes = best["poses"], best["keyframes"]
    ov_s, ow_s = ov[k0:end], ow[k0:end]
    lc = dict(loops=0)
    if loop_closure:
        poses, lc = close_loops(scans, poses, keyframes, k0, ov_s, ow_s, res=res, log=log)
        if lc.get("loops"):
            log(f"  루프 닫기: 다시 지나간 곳 {lc['loops']}개 (후보 {lc['candidates']}), 키프레임 최대 {100 * lc['moved_max_m']:.1f} cm 옮김, "
                f"맞춘 뒤 남은 차이 중앙값 {lc['loop_residual_cm']:.1f} cm | {lc['seconds']:.0f} s")
        else:
            log(f"  루프 닫기: 다시 지나간 곳을 못 찾음 (후보 {lc.get('candidates', 0)}개) -> 추적한 위치 그대로")
    t_match = time.time() - t_start
    # ray cast the keyframes: hits at the beam ends, pass-throughs along the beams up to 0.10 m short
    slack = cast_range + 2.0
    x0 = float(poses[:, 0].min() - slack); y0 = float(poses[:, 1].min() - slack)
    W = int(math.ceil((poses[:, 0].max() + slack - x0) / res)); H = int(math.ceil((poses[:, 1].max() + slack - y0) / res))
    hit = np.zeros(H * W, np.int32)
    free = np.zeros(H * W, np.int32)
    beams_c = np.arange(0, N, cast_step)
    step = 0.7 * res
    tt = np.arange(int(cast_range / step) + 1) * step
    batch_i, batch_f = [], []

    def flush():
        if batch_i:
            hit[:] += np.bincount(np.concatenate(batch_i), minlength=H * W).astype(np.int32)
            free[:] += np.bincount(np.concatenate(batch_f), minlength=H * W).astype(np.int32)
            batch_i.clear(); batch_f.clear()

    for kk in keyframes:
        pts = car_points(scans, k0 + kk, beams_c, ov_s[kk], ow_s[kk], cast_range)
        if not len(pts):
            continue
        wpt = transform(poses[kk], pts)
        o = transform(poses[kk], np.array([scans.laser[:2]]))[0]
        j = np.floor((wpt[:, 0] - x0) / res).astype(np.int64); i = np.floor((wpt[:, 1] - y0) / res).astype(np.int64)
        ok = (j >= 0) & (j < W) & (i >= 0) & (i < H)
        batch_i.append(i[ok] * W + j[ok])
        dvec = wpt - o
        r = np.linalg.norm(dvec, axis=1)
        u = dvec / np.maximum(r, 1e-6)[:, None]
        m = tt[None, :] < (r - 0.10)[:, None]                 # (M, S)
        sx = o[0] + u[:, :1] * tt[None, :]
        sy = o[1] + u[:, 1:] * tt[None, :]
        jj = np.floor((sx[m] - x0) / res).astype(np.int64); ii = np.floor((sy[m] - y0) / res).astype(np.int64)
        ok = (jj >= 0) & (jj < W) & (ii >= 0) & (ii < H)
        batch_f.append(ii[ok] * W + jj[ok])
        if len(batch_f) >= 20:
            flush()
    flush()
    hit, free = hit.reshape(H, W), free.reshape(H, W)
    # crop to what was seen, plus a margin
    rr, cc = np.nonzero((hit > 0) | (free > 0))
    m = int(round(1.0 / res))
    r0, r1 = max(rr.min() - m, 0), min(rr.max() + m + 1, H)
    c0, c1 = max(cc.min() - m, 0), min(cc.max() + m + 1, W)
    hit, free = hit[r0:r1, c0:c1].copy(), free[r0:r1, c0:c1].copy()
    every = max(1, int(round(0.25 / dt)))
    path_odom = float(np.linalg.norm(np.diff(op[k0:end:every, :2], axis=0), axis=1).sum())
    inl = best["inliers"]
    stats = dict(scans=int(end - k0), scans_read=int(K), keyframes=int(len(keyframes)), fallbacks=int(best["fallbacks"]),
                 field_rebuilds=int(best["field_rebuilds"]), stretches=len(stretches), laps=round(float(best["laps"]), 2),
                 stretch_s=[round(float(scans.t[k0] - scans.t[0]), 1), round(float(scans.t[end - 1] - scans.t[0]), 1)],
                 inliers_median=float(np.median(inl[1:])), inliers_p05=float(np.percentile(inl[1:], 5)),
                 match_seconds=t_match, path_m=best["path"], odom_path_m=path_odom, loop_closure=lc,
                 odom_over_map=path_odom / max(best["path"], 1e-6))
    return MapResult(poses=poses, odom_poses=op[k0:end], t=scans.t[k0:end], keyframes=np.asarray(keyframes),
                     inliers=inl, hit=hit, free=free, x0=x0 + c0 * res, y0=y0 + r0 * res, res=res,
                     seconds=time.time() - t_start, stats=stats,
                     stretches=[dict(start_s=round(float(scans.t[z["k0"]] - scans.t[0]), 1),
                                     end_s=round(float(scans.t[z["end"] - 1] - scans.t[0]), 1),
                                     path_m=round(z["path"], 1), laps=round(float(z["laps"]), 2), why=z["why"]) for z in stretches])


# ---------------------------------------------------------------- reading a finished map
def write_ros_map(img: np.ndarray, x0: float, y0: float, res: float, out_dir: str, name: str,
                  comment: str = "") -> str:
    """map_server pair: <name>.pgm (row 0 = the top) + <name>.yaml."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    pgm = os.path.join(out_dir, f"{name}.pgm")
    with open(pgm, "wb") as f:
        f.write(b"P5\n")
        if comment:
            f.write(f"# {comment}\n".encode())
        f.write(f"{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(np.ascontiguousarray(img, np.uint8).tobytes())
    with open(os.path.join(out_dir, f"{name}.yaml"), "w") as f:
        f.write(f"image: {name}.pgm\nresolution: {res}\norigin: [{x0}, {y0}, 0.0]\nnegate: 0\n"
                f"occupied_thresh: 0.65\nfree_thresh: 0.196\n")
    return os.path.join(out_dir, f"{name}.yaml")


@dataclass
class GridMap:
    """A map_server map in memory: classes per cell (row 0 = y0), origin, resolution."""
    free: np.ndarray
    occ: np.ndarray
    x0: float
    y0: float
    res: float

    @staticmethod
    def load(yaml_path: str) -> "GridMap":
        import os
        import yaml
        from PIL import Image
        meta = yaml.safe_load(open(yaml_path))
        img_path = meta["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(yaml_path), img_path)
        img = np.flipud(np.array(Image.open(img_path).convert("L"), np.float32))
        p = img / 255.0 if int(meta.get("negate", 0)) else (255.0 - img) / 255.0
        occ = p > float(meta.get("occupied_thresh", 0.65))
        free = p < float(meta.get("free_thresh", 0.196))
        o = meta.get("origin", [0.0, 0.0, 0.0])
        return GridMap(free=free, occ=occ, x0=float(o[0]), y0=float(o[1]), res=float(meta["resolution"]))

    def surface(self) -> np.ndarray:
        """Occupied cells that touch free space: the faces a LiDAR sees."""
        return self.occ & ndimage.binary_dilation(self.free, structure=np.ones((3, 3), bool))

    def cells_xy(self, mask: np.ndarray) -> np.ndarray:
        r, c = np.nonzero(mask)
        return np.stack([self.x0 + (c + 0.5) * self.res, self.y0 + (r + 0.5) * self.res], 1)

    def lane(self, seed_xy=None) -> np.ndarray:
        """The free region the car drove in: the loader's `keep_region` (0.15 m opening, the
        component around `seed_xy` or the largest)."""
        r_open = max(1, int(round(0.15 / self.res)))
        opened = ndimage.binary_opening(self.free, structure=np.ones((3, 3), bool), iterations=r_open)
        lab, n = ndimage.label(opened)
        if n == 0:
            return np.zeros_like(self.free)
        if seed_xy is not None:
            c = int((seed_xy[0] - self.x0) / self.res); r = int((seed_xy[1] - self.y0) / self.res)
            if 0 <= r < lab.shape[0] and 0 <= c < lab.shape[1] and lab[r, c] > 0:
                comp = lab == lab[r, c]
            else:
                comp = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        else:
            comp = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        return ndimage.binary_dilation(comp, structure=np.ones((3, 3), bool), iterations=r_open + 1) & self.free


def sample_classes(src: GridMap, dst: GridMap, T: np.ndarray, scale: float = 1.0):
    """`src`'s free / occupied masks on `dst`'s cells, where `T` (x, y, yaw) and `scale` take src
    points into dst: dst = T . (scale * src). Nearest neighbour."""
    H, W = dst.free.shape
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    pd = np.stack([dst.x0 + (jj.ravel() + 0.5) * dst.res, dst.y0 + (ii.ravel() + 0.5) * dst.res], 1)
    ps = transform(inverse(T), pd) / scale
    j = np.floor((ps[:, 0] - src.x0) / src.res).astype(np.int64)
    i = np.floor((ps[:, 1] - src.y0) / src.res).astype(np.int64)
    ok = (j >= 0) & (j < src.free.shape[1]) & (i >= 0) & (i < src.free.shape[0])
    jc, ic = np.clip(j, 0, src.free.shape[1] - 1), np.clip(i, 0, src.free.shape[0] - 1)
    f = (ok & src.free[ic, jc]).reshape(H, W)
    o = (ok & src.occ[ic, jc]).reshape(H, W)
    return f, o


def align(src: GridMap, dst: GridMap, guess=None, fit_scale: bool = True, log=print):
    """(T, scale, inliers) taking `src`'s surfaces onto `dst`'s. Without a guess, every yaw at 3 deg
    is tried with the translation found by FFT correlation, and the best few are refined."""
    from scipy.signal import fftconvolve
    sp = src.cells_xy(src.surface())
    if len(sp) > 6000:
        sp = sp[np.random.default_rng(0).choice(len(sp), 6000, replace=False)]
    fld = Field(dst.surface(), dst.x0, dst.y0, dst.res, d_max=1.0)
    cands = []
    if guess is not None:
        cands.append(np.asarray(guess, float))
    else:
        # coarse grid (0.10 m), a blurred target, each yaw's best shift by correlation
        rc = 0.10
        tgt = Field(dst.surface(), dst.x0, dst.y0, dst.res, d_max=0.5)
        f = max(1, int(round(rc / dst.res)))
        like = np.exp(-(tgt.D / 0.15) ** 2)
        like = like[: like.shape[0] // f * f, : like.shape[1] // f * f]
        like = like.reshape(like.shape[0] // f, f, like.shape[1] // f, f).max(axis=(1, 3))
        cen = sp.mean(0)
        scored = []
        for th in np.deg2rad(np.arange(0, 360, 3)):
            q = transform((0, 0, th), sp - cen)
            q0 = q.min(0)
            idx = np.floor((q - q0) / rc).astype(int)
            img = np.zeros(idx.max(0)[::-1] + 1)
            np.add.at(img, (idx[:, 1], idx[:, 0]), 1.0)
            corr = fftconvolve(like, img[::-1, ::-1], mode="full")
            k = int(np.argmax(corr))
            r, c = divmod(k, corr.shape[1])
            # shift of img's (0,0) cell inside `like`: full-mode index minus (img size - 1)
            oy, ox = r - (img.shape[0] - 1), c - (img.shape[1] - 1)
            tx = tgt.x0 + ox * rc - q0[0]
            ty = tgt.y0 + oy * rc - q0[1]
            # pose taking (src - cen) to dst; as a pose on src points: rotate about cen then shift
            Tp = compose(np.array([tx, ty, th]), np.array([-cen[0], -cen[1], 0.0]))
            scored.append((float(corr.flat[k]), Tp))
        scored.sort(key=lambda z: -z[0])
        cands = [p for _, p in scored[:6]]
    best = None
    for c in cands:
        T, info = match(fld, sp, c, iters=60, cauchy=0.2)
        if fit_scale:
            T, info = match(fld, sp, T, iters=80, cauchy=0.2, fit_scale=True)
        if best is None or info.inliers > best[2]:
            best = (T, info.scale, info.inliers)
    return best


def compare(ours: GridMap, ref: GridMap, T: np.ndarray, scale: float, seed_xy_ref=None) -> Dict[str, float]:
    """How alike two maps are on `ref`'s grid, `ours` placed by (T, scale).

    * `free_iou`: free cells, as written in each map.
    * `lane_iou`: the drivable lane (the loader's keep_region) of each.
    * `wall_f1_10cm`: surface cells of each within 0.10 m of a surface cell of the other (both
      directions, harmonic mean) -- the walls agree to a cell or two.
    * `wall_mean_cm` / `wall_p95_cm`: distance from our surface cells to the reference's surface.
    """
    f_o, o_o = sample_classes(ours, ref, T, scale)
    on = GridMap(free=f_o, occ=o_o, x0=ref.x0, y0=ref.y0, res=ref.res)
    inter = (f_o & ref.free).sum(); union = (f_o | ref.free).sum()
    lane_o, lane_r = on.lane(seed_xy_ref), ref.lane(seed_xy_ref)
    li = (lane_o & lane_r).sum(); lu = (lane_o | lane_r).sum()
    so, sr = on.surface(), ref.surface()
    d_to_r = ndimage.distance_transform_edt(~sr) * ref.res
    d_to_o = ndimage.distance_transform_edt(~so) * ref.res
    p = float((d_to_r[so] <= 0.10).mean()) if so.any() else 0.0
    r = float((d_to_o[sr] <= 0.10).mean()) if sr.any() else 0.0
    return dict(free_iou=float(inter / max(union, 1)), lane_iou=float(li / max(lu, 1)),
                wall_f1_10cm=float(2 * p * r / max(p + r, 1e-9)), wall_precision_10cm=p, wall_recall_10cm=r,
                wall_mean_cm=float(100 * d_to_r[so].mean()) if so.any() else float("nan"),
                wall_p95_cm=float(100 * np.percentile(d_to_r[so], 95)) if so.any() else float("nan"),
                lane_area_ours_m2=float(lane_o.sum() * ref.res ** 2), lane_area_ref_m2=float(lane_r.sum() * ref.res ** 2))


def scale_from_scans(ref: GridMap, scans: Scans, poses: np.ndarray, odom: Odom, n: int = 150,
                     step: int = 3, max_range: float = 12.0) -> Dict[str, float]:
    """The map's scale against the LiDAR: each of `n` scans spread over the drive is matched to the
    map's surfaces from its pose in the map frame (`poses`, e.g. the stack's particle filter), with
    one extra factor on every range. A true-scale map gives 1.00; a map drawn 20 % small gives 0.80
    (the scan has to shrink to fit it)."""
    fld = Field(ref.surface(), ref.x0, ref.y0, ref.res, d_max=1.0)
    _, ov, ow = odom_at(odom, scans.t)
    K = len(scans.t)
    beams = np.arange(0, scans.ranges.shape[1], step)
    s_all, inl_all = [], []
    for k in np.linspace(0, K - 1, min(n, K)).astype(int):
        pts = car_points(scans, k, beams, ov[k], ow[k], max_range)
        if len(pts) < 50:
            continue
        T, info = match(fld, pts, poses[k], iters=30)
        if info.inliers < 0.5:
            continue
        T, info = match(fld, pts, T, iters=60, fit_scale=True)
        if info.inliers >= 0.6:
            s_all.append(info.scale); inl_all.append(info.inliers)
    if not s_all:
        return dict(scale=float("nan"), scale_iqr=float("nan"), scans_used=0)
    s_all = np.array(s_all)
    q1, q3 = np.percentile(s_all, [25, 75])
    return dict(scale=float(np.median(s_all)), scale_iqr=float(q3 - q1), scans_used=int(len(s_all)),
                inliers_median=float(np.median(inl_all)))


def driving_direction(path_xy: np.ndarray) -> str:
    """'ccw' / 'cw' from the signed area the driven path sweeps (laps summed)."""
    x, y = path_xy[:, 0], path_xy[:, 1]
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    return "ccw" if area > 0 else "cw"
