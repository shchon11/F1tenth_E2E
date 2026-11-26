"""(11-26) How a state decoder's answer is drawn -- numpy only (no torch): the console (`viewer/sim_worker.py`) and the site's
dash server (tools/live_dec.py, which must not import torch) both draw with it, so the two look alike.

The user (11-26): "ROI는 왜 걸어놓은 거야? 존나 거슬리게" -- no frame around the decoder's grid, no background fill, the grid's
edge invisible; draw the decoded walls and boxes only where the probability means something, fading out. So a decoded wall or
box is its EDGE: where the decoder's probability crosses one half (marching squares over the cells' centres), each piece as
strong as the decoder is sure there (how far apart the probabilities of the cells around it are: a crisp 0.05 -> 0.95 edge
full, a 0.4 -> 0.6 mush nothing) and fading to nothing over the grid's last metre (the grid's outline never shows). Drawn
dashed: everything a decoder says is dashed or hollow, in the colour of the true thing.

`edges(prob, grid)` -> segments (n, 2, 2) in the car's frame and their strength (n,); `chains` joins them into polylines;
`dashes` cuts a polyline into dashes measured in metres (the console's 3D view draws solid lines only, and the site draws
the same dashes so the two look alike); `dashed_edges` all three, in the world; `outline` / `ring` a car's outline and a
ring."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

FADE_M = 1.0              # [m] the strength falls to 0 over this much of the grid's border (0 at one cell from it)
CONF_LO, CONF_HI = 0.2, 0.7   # the cells' spread (max - min) at which an edge starts to show / is at full strength


def _grid(grid) -> Tuple[float, float, float, float, float, int, int]:
    x0, x1, y0, y1, res = (float(v) for v in list(grid)[:5])
    return x0, x1, y0, y1, res, int(round((x1 - x0) / res)), int(round((y1 - y0) / res))


def edges(prob: np.ndarray, grid, level: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`prob` (NX, NY) the decoder's cell probabilities (cell (ix, iy): x ahead, y left), `grid` (x0, x1, y0, y1, res) ->
    segs (n, 2, 2) [m] in the car's frame, strength (n,) in [0, 1], keys (n, 2) int64: the two cell edges each segment joins
    (for `chains`). Marching squares between the cells' centres; the two ambiguous cases split by the square's mean."""
    x0, x1, y0, y1, res, nx, ny = _grid(grid)
    p = np.asarray(prob, np.float32).reshape(nx, ny)
    v0, v1, v2, v3 = p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]          # corners (i, j) (i+1, j) (i+1, j+1) (i, j+1)
    b0, b1, b2, b3 = v0 >= level, v1 >= level, v2 >= level, v3 >= level
    case = b0.astype(np.int8) | (b1.astype(np.int8) << 1) | (b2.astype(np.int8) << 2) | (b3.astype(np.int8) << 3)
    I, J = np.nonzero((case != 0) & (case != 15))
    if not len(I):
        z = np.zeros((0, 2, 2), np.float32)
        return z, np.zeros(0, np.float32), np.zeros((0, 2), np.int64)
    c = case[I, J]
    a0, a1, a2, a3 = v0[I, J], v1[I, J], v2[I, J], v3[I, J]
    t = lambda a, b: np.clip((level - a) / np.where(np.abs(b - a) < 1e-6, 1e-6, b - a), 0.0, 1.0)
    cx = x0 + (I + 0.5) * res; cy = y0 + (J + 0.5) * res                   # corner 0's centre
    # the square's four sides: 0 = bottom (v0 -> v1, along x at y_j), 1 = right (v1 -> v2, along y at x_i+1),
    # 2 = top (v3 -> v2, along x at y_j+1), 3 = left (v0 -> v3, along y at x_i)
    side_pt = [np.stack([cx + t(a0, a1) * res, cy], 1), np.stack([cx + res, cy + t(a1, a2) * res], 1),
               np.stack([cx + t(a3, a2) * res, cy + res], 1), np.stack([cx, cy + t(a0, a3) * res], 1)]
    # a side's key: the cell edge it lies on, numbered over the whole grid (x-edges then y-edges)
    side_key = [(I * ny + J), (nx * ny + (I + 1) * ny + J), (I * ny + J + 1), (nx * ny + I * ny + J)]
    mean_hi = (a0 + a1 + a2 + a3) * 0.25 >= level
    # case -> the pairs of sides a segment joins (bit k = corner k at or above the level)
    table = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)], 9: [(2, 0)],
             11: [(2, 1)], 12: [(1, 3)], 13: [(1, 0)], 14: [(0, 3)]}
    segs, keys, rows = [], [], []
    for cs, pairs in table.items():
        m = np.nonzero(c == cs)[0]
        for sa, sb in pairs:
            segs.append(np.stack([side_pt[sa][m], side_pt[sb][m]], 1)); keys.append(np.stack([side_key[sa][m], side_key[sb][m]], 1))
            rows.append(m)
    for cs in (5, 10):                                                     # the saddles: two segments each
        m = np.nonzero(c == cs)[0]
        if not len(m):
            continue
        hi = mean_hi[m]
        # case 5 (corners 0, 2 high): joined through the middle when the mean is high -> the high corners connect
        if cs == 5:
            pa = [(3, 2), (1, 0)]; pb = [(3, 0), (1, 2)]
        else:
            pa = [(0, 3), (2, 1)]; pb = [(0, 1), (2, 3)]
        for k in range(2):
            for mm, (sa, sb) in ((m[hi], pa[k]), (m[~hi], pb[k])):
                if len(mm):
                    segs.append(np.stack([side_pt[sa][mm], side_pt[sb][mm]], 1))
                    keys.append(np.stack([side_key[sa][mm], side_key[sb][mm]], 1)); rows.append(mm)
    segs = np.concatenate(segs, 0).astype(np.float32); keys = np.concatenate(keys, 0).astype(np.int64)
    rows = np.concatenate(rows, 0)
    hi4 = np.maximum(np.maximum(a0, a1), np.maximum(a2, a3))[rows]
    lo4 = np.minimum(np.minimum(a0, a1), np.minimum(a2, a3))[rows]
    conf = np.clip((hi4 - lo4 - CONF_LO) / (CONF_HI - CONF_LO), 0.0, 1.0)
    mid = segs.mean(1)
    d = np.minimum(np.minimum(mid[:, 0] - x0, x1 - mid[:, 0]), np.minimum(mid[:, 1] - y0, y1 - mid[:, 1]))
    u = np.clip((d - res) / (FADE_M - res), 0.0, 1.0)                      # the outermost pieces (a cell in) are 0
    fade = u * u * (3.0 - 2.0 * u)
    return segs, (conf * fade).astype(np.float32), keys


def chains(segs: np.ndarray, strength: np.ndarray, keys: np.ndarray, smooth: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
    """the segments chained into polylines: [(points (m, 2), strength at each point (m,))] -- a point's strength is the mean
    of its segments', then averaged over `smooth` points either side along the line (a strength that flickers from cell to
    cell would chop the dashes into a shimmer)"""
    n = len(segs)
    at = {}
    for i in range(n):
        for e in (0, 1):
            at.setdefault(int(keys[i, e]), []).append(i)
    used = np.zeros(n, bool)

    def walk(i0, end):
        """from segment i0 out through its end `end` (0 / 1): [(point, segment)] beyond it"""
        out, i, e = [], i0, end
        while True:
            key = int(keys[i, e])
            nxt = [m for m in at.get(key, ()) if not used[m]]
            if not nxt:
                return out
            m = nxt[0]; used[m] = True
            e = 1 if int(keys[m, 0]) == key else 0          # leave m through its other end
            out.append((segs[m, e], m))
            i = m
    lines = []
    for i0 in range(n):
        if used[i0]:
            continue
        used[i0] = True
        fwd, back = walk(i0, 1), walk(i0, 0)
        pts = [q for q, _ in back[::-1]] + [segs[i0, 0], segs[i0, 1]] + [q for q, _ in fwd]
        sg = [m for _, m in back[::-1]] + [i0] + [m for _, m in fwd]      # segment k joins point k and k + 1
        st = np.asarray(strength, np.float32)[sg]
        w = np.concatenate([st[:1], st]) * 0.5 + np.concatenate([st, st[-1:]]) * 0.5
        if smooth > 0 and len(w) > 2:
            k = np.ones(2 * smooth + 1, np.float32)
            num = np.convolve(np.pad(w, smooth, mode="edge"), k, "valid")
            w = num / k.sum()
        lines.append((np.asarray(pts, np.float32), w.astype(np.float32)))
    return lines


def dashes(line: np.ndarray, weight=None, on: float = 0.14, off: float = 0.09, phase: float = 0.0):
    """(m, 2) polyline [and its per-point `weight` (m,)] -> its dashes as segments (k, 2, 2), `on` m drawn then `off` m
    left out along the line (a dash follows the line's bends: it can be more than one segment), and each segment's weight
    (k,) -- the line's weight where the dash is"""
    p = np.asarray(line, np.float64).reshape(-1, 2)
    wt = np.ones(len(p)) if weight is None else np.asarray(weight, np.float64).reshape(-1)
    empty = (np.zeros((0, 2, 2), np.float32), np.zeros(0, np.float32))
    if len(p) < 2:
        return empty
    seg = np.diff(p, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
    L = float(s[-1])
    if L <= 1e-9:
        return empty
    starts = np.arange(-phase, L, on + off)
    a, b = np.clip(starts, 0.0, L), np.clip(starts + on, 0.0, L)
    keep = b - a > 1e-4
    out, ws = [], []
    for lo, hi in zip(a[keep], b[keep]):
        q = np.concatenate([[lo], s[(s > lo) & (s < hi)], [hi]])
        pts = np.stack([np.interp(q, s, p[:, 0]), np.interp(q, s, p[:, 1])], 1)
        out.append(np.stack([pts[:-1], pts[1:]], 1))
        ws.append(np.full(len(pts) - 1, np.interp(0.5 * (lo + hi), s, wt)))
    if not out:
        return empty
    return np.concatenate(out, 0).astype(np.float32), np.concatenate(ws).astype(np.float32)


def dashed_edges(prob: np.ndarray, grid, pose, on: float = 0.14, off: float = 0.09, floor: float = 0.06):
    """`edges` -> `chains` -> `dashes` in the world (the grid in the frame of the car at `pose` (x, y, yaw)): dash segments
    (k, 2, 2) and their strength (k,), the ones under `floor` dropped"""
    segs, st, keys = edges(prob, grid)
    if not len(segs):
        return np.zeros((0, 2, 2), np.float32), np.zeros(0, np.float32)
    D, W = [], []
    for pts, w in chains(segs, st, keys):
        d, wd = dashes(to_world(pts, pose), w, on, off)
        D.append(d); W.append(wd)
    d, w = np.concatenate(D, 0), np.concatenate(W, 0)
    k = w >= floor
    return d[k], w[k]


def to_world(xy: np.ndarray, pose: Sequence[float]) -> np.ndarray:
    """car-frame points (..., 2) at pose (x, y, yaw) -> world"""
    x, y, h = (float(v) for v in pose[:3])
    c, s = math.cos(h), math.sin(h)
    q = np.asarray(xy, np.float64)
    return np.stack([x + q[..., 0] * c - q[..., 1] * s, y + q[..., 0] * s + q[..., 1] * c], -1).astype(np.float32)


def outline(x: float, y: float, yaw: float, length: float = 0.58, width: float = 0.31) -> np.ndarray:
    """a car's closed outline (5, 2) in the world"""
    c, s = math.cos(yaw), math.sin(yaw)
    q = [(length / 2, width / 2), (length / 2, -width / 2), (-length / 2, -width / 2), (-length / 2, width / 2),
         (length / 2, width / 2)]
    return np.array([[x + a * c - b * s, y + a * s + b * c] for a, b in q], np.float32)


def ring(x: float, y: float, r: float, n: int = 24) -> np.ndarray:
    """a closed circle (n + 1, 2)"""
    a = np.linspace(0.0, 2.0 * math.pi, n + 1)
    return np.stack([x + r * np.cos(a), y + r * np.sin(a)], 1).astype(np.float32)
