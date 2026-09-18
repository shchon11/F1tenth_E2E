"""`+hard<seed>`: obstacle layouts copied from what the user builds by hand in the environment editor.

The catalogue's `+obs` and `+rlobs` put three-ish single boxes on a lap, each hugging a wall or the
racing line and each leaving 1.0-1.2 m open. Watching the policy drive hand-built scenes
(`~/f1sim_scenes/scene_0912_*`, 2026-09-12) showed what it actually fails at, and none of it is a
single box with a metre of room:

  gate      two or three boxes side by side across the lane, 55-75 % of the lane left open at a wall
  diagonal  three to five boxes stepping diagonally across the lane, the gap at the far end
  chicane   a block from one wall, then 2-3.5 m later a block from the other -- an S the car has
            to thread, not a box it can ignore
  apex      a block on the inside of a corner, where the line wants to be
  cluster   two or three boxes touching, wedged into one side of the lane
  scatter   one to three SMALL objects (0.10-0.22 m: a cone, a small box) at random places across the
            lane, mid-lane included -- the user's point that avoiding only big boxes is not avoiding

Apex and cluster boxes are drawn small (0.15-0.25 m) about a third of the time for the same reason.

The open share is what the user's scenes leave (1.5-1.85 m of a 2.25-2.5 m lane; first cut of this
file left 0.6-0.85 m and the user rightly called it impassable). Every pattern is built with an
explicit gap and then *proved* passable: the free space is eroded by
0.25 m (car half-width 0.14 m plus margin) and the lane before the pattern must still connect to
the lane after it. A pattern that closes the lap is undone and redrawn. Boxes go into both
`occupancy` and `tall`, so the LiDAR sees them at every beam height and the car collides with them.

Grid rectangles rather than modelled props on purpose: every checkpoint in the catalogue was
trained against rasterised boxes, and the hand-built scenes are cardboard boxes of about the same
footprint (0.36 x 0.30 m).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from .track import Track

CAR_HALF_W = 0.14
ERODE_M = 0.25            # proof margin: a 0.6 m gap survives erosion with 0.1 m to spare
GAP_FRAC = (0.55, 0.75)   # open share of the lane beside a pattern: the hand-built scenes leave
                          # 1.50-1.85 m of a 2.25-2.50 m lane (measured, scene_0912_2344/2355)
GAP_MIN = 1.20            # floor where the lane is wide enough (>= 1.65 m)
GAP_NARROW = 0.70         # floor on narrower lanes: 2.5 car widths
BOX_W, BOX_D = 0.36, 0.30 # cardboard box: across the lane, along it
SMALL = (0.10, 0.22)      # cone / small box footprint
SMALL_SHARE = 0.35        # share of apex and cluster boxes drawn at 0.15-0.25 m
PATTERNS = ("gate", "diagonal", "chicane", "apex", "cluster", "scatter")


def _lane_halves(occ: np.ndarray, res: float, origin, p: np.ndarray, nrm: np.ndarray, limit: float = 4.0):
    """Free distance from centerline point p to the wall on the +nrm side and the -nrm side [m]."""
    H, W = occ.shape
    out = []
    for s in (1.0, -1.0):
        d = 0.0
        while d < limit:
            q = p + s * nrm * d
            j = int((q[0] - origin[0]) / res); i = int((q[1] - origin[1]) / res)
            if not (0 <= i < H and 0 <= j < W) or occ[i, j]:
                break
            d += res * 0.5
        out.append(max(0.0, d - res * 0.5))
    return out[0], out[1]


def _stamp(occ, tall, res, origin, cx, cy, tang, nrm, sx, sy):
    """Rasterise a rectangle sx along `tang`, sy along `nrm`, centred at (cx, cy)."""
    H, W = occ.shape
    r = 0.5 * float(np.hypot(sx, sy)) + res
    c0 = max(0, int((cx - r - origin[0]) / res)); c1 = min(W, int((cx + r - origin[0]) / res) + 2)
    r0 = max(0, int((cy - r - origin[1]) / res)); r1 = min(H, int((cy + r - origin[1]) / res) + 2)
    if c1 <= c0 or r1 <= r0:
        return
    xs = np.arange(c0, c1) * res + origin[0]; ys = np.arange(r0, r1) * res + origin[1]
    gx, gy = np.meshgrid(xs, ys)
    dx, dy = gx - cx, gy - cy
    u = dx * tang[0] + dy * tang[1]; v = dx * nrm[0] + dy * nrm[1]
    m = (np.abs(u) <= sx / 2) & (np.abs(v) <= sy / 2)
    occ[r0:r1, c0:c1] |= m; tall[r0:r1, c0:c1] |= m


ROW_GAP = (0.05, 0.25)    # daylight between adjacent boxes in a row: the hand-built rows are boxes
                          # set down one by one, not a wall, and the LiDAR sees through the cracks


def _block(v_lo: float, v_hi: float, rng=None) -> List[Tuple[float, float]]:
    """Split a lateral interval into a row of boxes with small gaps between them:
    [(v_centre, width), ...].

    The first cut of this file made rows as touching pieces, i.e. a wall with one opening. On the
    user's scene_0912_2355 the policy trained on that drove into a *gapped* row at 5-7 m/s eleven
    times out of sixteen (crash map, 2026-09-13 04:25): between two boxes 0.1-0.2 m apart the beams
    pass, and a policy that has only ever seen solid rows reads the cracks as a way through. So a row
    is boxes with 0.05-0.25 m of daylight, narrower than the car; the erosion proof (0.25 m) closes
    them, so the lap's real opening is still the one the pattern left."""
    span = v_hi - v_lo
    if span < 0.12:
        return []
    rng = rng or np.random.default_rng(0)
    out, v = [], v_lo
    while v_hi - v >= 0.12:
        w = min(BOX_W, v_hi - v)
        out.append((v + w / 2, w))
        v += w + float(rng.uniform(*ROW_GAP))
    return out


def with_hard_obstacles(track: Track, seed: int = 0, n: Optional[int] = None,
                        gap_frac: Tuple[float, float] = GAP_FRAC, min_spacing: float = 6.0,
                        patterns=PATTERNS) -> Track:
    """Copy of `track` with `n` hand-built-style obstacle patterns stamped into its grids."""
    if track.centerline is None:
        raise ValueError("hard obstacles need a centerline")
    rng = np.random.default_rng(seed)
    cl = np.asarray(track.centerline, float); N = len(cl)
    res = track.resolution; origin = track.origin
    tang = np.roll(cl, -1, 0) - np.roll(cl, 1, 0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)                    # +nrm = left of travel
    ds = float(np.linalg.norm(np.roll(cl, -1, 0) - cl, axis=1).mean())
    L = ds * N
    if n is None:
        n = int(np.clip(round(L / 10.0), 4, 14))
    # signed curvature: + = turning left, so the inside of the corner is the +nrm side
    heading = np.arctan2(tang[:, 1], tang[:, 0])
    dh = np.angle(np.exp(1j * (np.roll(heading, -2) - np.roll(heading, 2))))
    curv = dh / (4.0 * ds + 1e-9)
    curv = ndimage.uniform_filter1d(curv, size=max(3, int(1.0 / ds)), mode="wrap")
    apex_pool = np.argsort(-np.abs(curv))[: max(8, N // 5)]

    occ = track.occupancy.copy(); tall = track.tall.copy()
    occ0 = occ.copy()
    erode_it = int(round(ERODE_M / res))
    placed: List[int] = []
    kinds: List[str] = []
    accepted_boxes = []  # geometric recipe for asset-only consumers; does not alter legacy RNG/grids
    order = [patterns[j % len(patterns)] for j in range(n)]
    rng.shuffle(order)

    def arc_gap(i: int) -> float:
        return min((min(abs(i - j), N - abs(i - j)) for j in placed), default=1e9) * ds

    def connected(occ_now: np.ndarray, i0: int, i1: int) -> bool:
        """Do the lane just before index i0 and just after i1 still connect, with the car's width?"""
        free = ~occ_now
        if erode_it > 0:
            free = ndimage.binary_erosion(free, iterations=erode_it, border_value=0)
        lab, _ = ndimage.label(free)
        def comp_near(i: int) -> int:
            p = cl[i % N]
            for d in np.linspace(0.0, 1.2, 13):
                for s in (0.0, 1.0, -1.0):
                    q = p + s * nrm[i % N] * d
                    jj = int((q[0] - origin[0]) / res); ii = int((q[1] - origin[1]) / res)
                    if 0 <= ii < lab.shape[0] and 0 <= jj < lab.shape[1] and lab[ii, jj] > 0:
                        return int(lab[ii, jj])
            return 0
        a, b = comp_near(i0), comp_near(i1)
        return a > 0 and a == b

    tries = 0
    while len(placed) < n and tries < 60 * n:
        tries += 1
        kind = order[len(placed)]
        i = int(rng.choice(apex_pool)) if kind == "apex" else int(rng.integers(N))
        if arc_gap(i) < min_spacing:
            continue
        wl, wr = _lane_halves(occ0, res, origin, cl[i], nrm[i])
        width = wl + wr
        # The open share is the rule; the 1.2 m floor only where the lane can afford it. On a
        # 1.2-1.4 m lane (generated recipes with hoses, real:map16x07) that floor left no room for
        # any box at all, so narrow lanes fall back to a 0.70 m floor -- 2.5 car widths.
        g = width * float(rng.uniform(*gap_frac))
        g = max(GAP_MIN, g) if width >= GAP_MIN + 0.45 else max(GAP_NARROW, g)
        if width - g < 0.30:                                       # nothing fits beside the gap
            continue
        boxes = []                                                   # (index offset [m], v_centre, sx, sy)
        side = 1.0 if kind == "apex" and curv[i] > 0 else (-1.0 if kind == "apex" else float(rng.choice([-1.0, 1.0])))
        span = width - g
        if kind == "gate":
            lo, hi = (wl - span, wl) if side > 0 else (-wr, -wr + span)
            boxes += [(0.0, v, BOX_D, w) for v, w in _block(lo, hi, rng)]
        elif kind == "diagonal":
            k = int(rng.integers(3, 6))
            step_v = span / k
            for j in range(k):
                v = (wl - (j + 0.5) * step_v) if side > 0 else (-wr + (j + 0.5) * step_v)
                boxes.append((j * 0.5, v, BOX_D, min(BOX_W, step_v)))
        elif kind == "chicane":
            g2 = width * float(rng.uniform(*gap_frac))
            g2 = max(GAP_MIN, g2) if width >= GAP_MIN + 0.45 else max(GAP_NARROW, g2); span2 = width - g2
            lo, hi = (wl - span, wl) if side > 0 else (-wr, -wr + span)
            boxes += [(0.0, v, BOX_D, w) for v, w in _block(lo, hi, rng)]
            lo2, hi2 = (-wr, -wr + span2) if side > 0 else (wl - span2, wl)
            along = float(rng.uniform(2.0, 3.5))
            boxes += [(along, v, BOX_D, w) for v, w in _block(lo2, hi2, rng)]
        elif kind == "apex":
            lo, hi = (wl - span, wl) if side > 0 else (-wr, -wr + span)
            boxes += [(0.0, v, BOX_D, w) for v, w in _block(lo, hi, rng)]
            if rng.random() < SMALL_SHARE:                           # a small thing at the apex instead
                sz = float(rng.uniform(0.15, 0.25))
                v = (wl - sz / 2 - 0.05) if side > 0 else (-wr + sz / 2 + 0.05)
                boxes = [(0.0, v, sz, sz)]
        elif kind == "cluster":
            k = int(rng.integers(2, 4))
            edge = wl if side > 0 else -wr
            for j in range(k):
                across = (j % 2) * (BOX_W + 0.02) + BOX_W / 2 + 0.03
                v = edge - side * across
                if rng.random() < SMALL_SHARE:
                    sz = float(rng.uniform(0.15, 0.25)); v = edge - side * (across - BOX_W / 2 + sz / 2)
                    boxes.append(((j // 2) * (BOX_D + 0.03), v, sz, sz))
                else:
                    boxes.append(((j // 2) * (BOX_D + 0.03), v, BOX_D, BOX_W))
            # a cluster must still leave the gap on the other side
            if width - (2 * BOX_W + 0.1) < g:
                boxes = boxes[:1]
                if width - (BOX_W + 0.1) < g:
                    continue
        elif kind == "scatter":
            k = int(rng.integers(1, 4)); vs = []
            for _ in range(20):
                if len(vs) >= k:
                    break
                v = float(rng.uniform(-wr + 0.25, wl - 0.25))
                if all(abs(v - u) >= 0.6 for u, _a in vs):
                    vs.append((v, float(rng.uniform(0.0, 3.0))))
            for v, along in vs:
                sz = float(rng.uniform(*SMALL))
                boxes.append((along, v, sz, sz))
        if not boxes:
            continue
        occ_try = occ.copy(); tall_try = tall.copy()
        last = 0
        for along, v, sx, sy in boxes:
            k_ = (i + int(round(along / ds))) % N
            last = max(last, int(round(along / ds)))
            c = cl[k_] + nrm[k_] * v
            _stamp(occ_try, tall_try, res, origin, c[0], c[1], tang[k_], nrm[k_], sx, sy)
        back = int(round(2.0 / ds)); fwd = last + int(round(2.0 / ds))
        if not connected(occ_try, i - back, i + fwd):
            continue
        occ, tall = occ_try, tall_try
        placed.append(i); kinds.append(kind)
        for along, v, sx, sy in boxes:
            k_ = (i + int(round(along / ds))) % N
            c = cl[k_] + nrm[k_] * v
            accepted_boxes.append((float(c[0]), float(c[1]),
                                   float(np.arctan2(tang[k_, 1], tang[k_, 0])), float(sx), float(sy)))
    t = Track.from_occupancy(occ, res, origin, cl, f"{track.name}_hard{seed}", duct=track.duct, tall=tall,
                             duct_height=track.duct_height)
    t.props = track.props
    t.hard_boxes = tuple(accepted_boxes)
    t.hard_patterns = list(zip(placed, kinds))            # for tests and pictures
    return t
