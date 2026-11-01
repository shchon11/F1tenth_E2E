"""F1TENTH venue-style race maps as ROS maps (2026-11-01).

The user, on `trackgen`'s constant-width hose loops ("야 씨발 맵 장난하냐?"), on a grid maze ("넌 저게 레이싱 트랙처럼
보이냐?") and on F1 circuits as the reference ("이게 F1이냐?"): make the maps properly -- wide ones, narrow hard ones and
ones that mix the two -- the way F1TENTH venues are built.

What an F1TENTH venue is (the measured ones in `maps.REAL`: Korea 2026, ICRA 2022, ICCAS 2025, the TU Wien Blackbox
tracks, the two lab floors): a hall with an ISLAND of duct hose in it, and the track is the band round the island. The
island's two ends are the hairpins; where the island bends or bulges the track has an S or a chicane; an island with a
short side branch (ICRA 2022's hook) gives nested hairpins; the band's width changes along the lap (Korea 2026: about
0.9 to 2 m) and is not centred on the line.

How a map is made here:
  1. the islands' spine in a hall: the comb below;
  2. the centre line of the track: the closed level curve at distance r(p) round the spine, r = the duct's half
     thickness + the lane's half width on that side of the nearest spine segment, its inside corners rounded;
  3. the lane: a half width per kind along the lap (mixed: wide and narrow sections of the spine), a little off-centre
     and a few centimetres ragged as a SLAM wall is, capped where another part of the lap passes close;
  4. rasterised at 5 cm into the ROS map format of the measured venues, so training, the console and the raceline solver
     read it as they read Korea 2026 (registered in `maps.REAL` / `tracks._REAL`).

References (11-01, the user: "F1tenth Roboracer 경기들좀 레퍼런스들 찾아봐 ... 네 자아를 펼치지 말고"): the 2024 AI
Velocity Cup / F1TENTH rules -- "The track border is constructed from air ducts of 6 inch diameter", "The track will fit
into an area of around 30x10 m", "at least 3 car widths (90cm) wide everywhere"; the ForzaETH race stack paper -- tracks
"similar in length to competition tracks" are 30-60 m; photos of ICRA 2025 Atlanta, IROS 2024 Abu Dhabi, CDC 2024
Milano and ICRA 2024 Yokohama (TU Wien Scuderia Segfault): flexible ducts laid on the floor, so every corner is round;
duct "fingers" reaching in from the sides turn the band into a serpentine of hairpins; single-duct walls between two
passes; bulbous duct ends; lanes 1-3 m.

Layout (from the ICRA 2025 photo): a comb -- a root duct along the hall's near end and 1-2 duct fingers reaching along
it; the loop round the comb runs up and down every finger (hairpins at the tips, U-turns in the pockets) and back along
the near end (the back straight); fingers spaced so every wall between two passes is one duct.

Kinds (track width = both half widths):
  wide    2.0-2.8 m lanes, 1-2 fingers, an 18-24 m hall
  narrow  1.0-1.3 m lanes (>= 0.9 m, the rules' minimum), 2 fingers, an 18-22 m hall
  mixed   pockets of 1.0-1.3 m and 2.1-2.8 m lanes side by side, 2 fingers, a 20-26 m hall

usage: python -m f1sim.mapgen --kind mixed --seed 3 --name gen_mixed_3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, Optional

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "maps", "generated")
RES = 0.05                      # [m] per cell, as the measured venues
DS = 0.05                       # [m] centre-line sample spacing
WALL_M = 0.30                   # [m] solid kept between two parts of the lap (one duct)
PAD = 1.0                       # [m] floor kept round the comb inside the grid (a laid duct wanders)
MIN_HALF = 0.45                 # [m] no half width below this (the car is 0.31 m wide)
R_CLOSE = 0.5                   # [m] the inside turns of the centre line rounded to this (a pocket's U-turn; with half a
                                #     lane of >= 0.5 m outside it the car, 0.74 m at full lock, drives it wide)
R_OPEN = 0.4                    # [m] the outside turns, smoothed
R_MIN = 0.5                     # [m] no centre-line turn tighter than this (curvature over 0.3 m)

KINDS: Dict[str, dict] = {
    # fingers: how many duct fingers reach along the hall from its near end (each = two passes, the loop goes up one side
    # and down the other); lane: lane widths [m] per class (mixed draws a class per pocket); duct: one duct's thickness
    # [m] (a wall between two passes); reach: a finger's length as a share of the hall; wave: [m] how far a laid duct
    # wanders sideways; length: the hall [m]. The hall's width follows: 2 x fingers lanes + their ducts. Sized so a lap
    # is 50-80 m (competition tracks: 30-60 m, ICRA 2025 "significantly larger")
    "wide": dict(fingers=(1, 2), lane=((2.0, 2.8),), duct=(0.32, 0.40), reach=(0.55, 0.8), wave=0.5, length=(18.0, 24.0),
                 tip=(0.2, 0.35)),
    "narrow": dict(fingers=(2, 2), lane=((1.0, 1.3),), duct=(0.32, 0.38), reach=(0.6, 0.85), wave=0.35, length=(18.0, 22.0),
                   tip=(0.2, 0.3), pinches=3),
    "mixed": dict(fingers=(2, 2), lane=((1.0, 1.3), (2.1, 2.8)), duct=(0.32, 0.40), reach=(0.55, 0.85), wave=0.45,
                  length=(20.0, 26.0), tip=(0.2, 0.35)),
}


def _circ_gauss(x: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter1d(x, sigma, mode="wrap") if sigma > 0 else x


def _resample(c: np.ndarray, ds: float) -> np.ndarray:
    """a closed polyline resampled every `ds` along its length"""
    seg = np.linalg.norm(np.roll(c, -1, 0) - c, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(16, int(round(s[-1] / ds)))
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    cc = np.vstack([c, c[:1]])
    return np.stack([np.interp(t, s, cc[:, 0]), np.interp(t, s, cc[:, 1])], 1)


def frame(c: np.ndarray):
    """(unit tangent, unit left normal, signed curvature [1/m]) at every sample of a closed polyline"""
    d = (np.roll(c, -1, 0) - np.roll(c, 1, 0)) / (2 * DS)
    t = d / np.linalg.norm(d, axis=1, keepdims=True).clip(1e-9)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    psi = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    return t, n, np.gradient(psi) / DS


def _noise(n: int, corr_m: float, rng) -> np.ndarray:
    """smooth periodic noise along the lap, unit standard deviation, correlation length `corr_m`"""
    x = _circ_gauss(rng.standard_normal(n), corr_m / DS)
    return (x - x.mean()) / max(x.std(), 1e-9)


def _seg_dist(p, a, b):
    """distance from points p (n, 2) to the segment a-b"""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def _smooth_poly(P: np.ndarray, n: int = 40) -> np.ndarray:
    """an open polyline through P as a smooth curve (Catmull-Rom), n samples per leg: ducts bend, they do not kink"""
    P = np.vstack([P[:1] * 2 - P[1:2], P, P[-1:] * 2 - P[-2:-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return np.array(out)


def spine(kind: str, rng):
    """(hall (W, H), spine polylines (the root along the near end, then the fingers), and per finger its pocket lane
    width, duct thickness): a comb -- the loop round it runs up and down every finger (the long passes of an F1TENTH
    serpentine, hairpins at the finger tips and in the pockets) and back along the near end (the back straight)"""
    spec = KINDS[kind]
    nf = int(rng.integers(spec["fingers"][0], spec["fingers"][1] + 1))
    H = float(rng.uniform(*spec["length"]))
    # lanes across the width: each finger has a lane each side; pocket k (between finger k and k+1, and the two outer
    # strips) is one lane class
    cls = rng.integers(len(spec["lane"]), size=nf + 1)
    lanes = [float(rng.uniform(*spec["lane"][int(k)])) for k in cls]
    duct = float(rng.uniform(*spec["duct"]))
    # x of each finger: outer wall, lane, finger (a duct), lane | lane, duct, lane, finger ... (two lanes and a duct
    # between fingers, one lane and the wall outside the first and the last)
    x = PAD + duct + lanes[0] + duct / 2
    xs = [x]
    for k in range(1, nf):
        x += duct / 2 + lanes[k] + duct + lanes[k] + duct / 2     # lane, pocket wall, lane between fingers k-1 and k
        xs.append(x)
    W = x + duct / 2 + lanes[nf] + duct + PAD
    lane_max = max(lanes)
    y0 = PAD + duct + lane_max + 0.3                             # the root: one lane in from the near end (back straight)
    H = H + 2 * PAD
    spines = [np.array([[xs[0], y0], [xs[-1], y0]])]
    per = []
    for k, xk in enumerate(xs):
        L = float(rng.uniform(*spec["reach"])) * (H - PAD - y0 - duct - lane_max - 0.3)
        ys = np.linspace(y0, y0 + L, 6)
        wob = spec["wave"] * rng.uniform(-1.0, 1.0, 6); wob[0] = 0.0
        f_poly = _smooth_poly(np.stack([xk + wob, ys], 1), 10)
        spines.append(f_poly)
        per.append((lanes[k], lanes[k + 1], duct))
    return (W, H), spines, per


def segments(kind: str, spines, per, rng):
    """(point, query point -> (lane half width, island half thickness) on that side of the nearest spine segment):
    a finger is one duct with its left pocket's lane on its left and its right pocket's on its right, a bulb at its
    tip; the root is a duct with the pockets above it and the back straight below"""
    spec = KINDS[kind]
    A_, B_, hl, hr, ih = [], [], [], [], []
    root = spines[0]
    back = max(p[0] for p in per)                                   # the back straight: the widest lane of the hall
    for a_, b_ in zip(root[:-1], root[1:]):
        A_.append(a_); B_.append(b_); hl.append(back / 2); hr.append(back / 2); ih.append(per[0][2] / 2)
    for sp, (l_left, l_right, duct) in zip(spines[1:], per):
        n = len(sp) - 1
        tip = float(rng.uniform(*spec["tip"]))
        for j, (a_, b_) in enumerate(zip(sp[:-1], sp[1:])):
            A_.append(a_); B_.append(b_); hl.append(l_left / 2); hr.append(l_right / 2)
            w_tip = max(0.0, 1.0 - (n - 1 - j) / 3.0)
            ih.append(duct / 2 + (tip - duct / 2) * w_tip if tip > duct / 2 else duct / 2)
    A_, B_ = np.array(A_), np.array(B_)
    D_ = (B_ - A_) / np.linalg.norm(B_ - A_, axis=1, keepdims=True).clip(1e-9)
    hl, hr, ih = np.array(hl), np.array(hr), np.array(ih)

    def nearest(p):
        return np.stack([_seg_dist(p, a_, b_) for a_, b_ in zip(A_, B_)], 1).argmin(1)

    def lane_at(p_spine, p):
        """(lane half width, island half thickness) for query points p whose nearest spine point is p_spine"""
        k = nearest(p_spine)
        v = p - A_[k]
        left = (D_[k, 0] * v[:, 1] - D_[k, 1] * v[:, 0]) > 0
        return np.where(left, hl[k], hr[k]), ih[k]

    return nearest, lane_at


def loop_round(spines, radius_fn, hall, res: float = RES):
    """the closed level curve at distance r(p) round the spine (r at the nearest spine point): (n, 2) metric"""
    from skimage.measure import find_contours
    W, H = hall
    nx, ny = int(math.ceil(W / res)) + 1, int(math.ceil(H / res)) + 1
    mask = np.ones((ny, nx), bool)                              # True = not spine
    for sp in spines:
        for a_, b_ in zip(sp[:-1], sp[1:]):
            n = max(2, int(np.linalg.norm(b_ - a_) / (res / 2)))
            p = a_[None] + (b_ - a_)[None] * np.linspace(0, 1, n)[:, None]
            ij = np.clip(np.round(p / res).astype(int), 0, [nx - 1, ny - 1])
            mask[ij[:, 1], ij[:, 0]] = False
    dist, ind = ndimage.distance_transform_edt(mask, sampling=res, return_indices=True)
    sp_xy = np.stack([ind[1].ravel() * res, ind[0].ravel() * res], 1)
    gy, gx = np.mgrid[0:ny, 0:nx]
    cell_xy = np.stack([gx.ravel() * res, gy.ravel() * res], 1)
    r = radius_fn(sp_xy, cell_xy).reshape(dist.shape)
    # round the inside corners (where a finger leaves the main spine the level curve has a kink of radius ~0): a
    # closing of the region inside the curve by a disc of R_CLOSE -- every concave turn becomes an arc of at least
    # R_CLOSE, and a pocket between two fingers too narrow to drive into is filled
    A = (dist - r) <= 0
    B = ndimage.distance_transform_edt(~A, sampling=res) <= R_CLOSE
    C = ndimage.distance_transform_edt(B, sampling=res) > R_CLOSE
    # and a light opening: the outside turns lose their raster kinks
    D = ndimage.distance_transform_edt(C, sampling=res) > R_OPEN
    C = ndimage.distance_transform_edt(~D, sampling=res) <= R_OPEN
    sd = ndimage.distance_transform_edt(C, sampling=res) - ndimage.distance_transform_edt(~C, sampling=res)
    closed = [c for c in find_contours(sd, 0.0) if np.linalg.norm(c[0] - c[-1]) < 2.0 and len(c) > 50]
    if len(closed) != 1:
        raise ValueError(f"{len(closed)} closed loops round the island")
    return closed[0][:, ::-1] * res                             # (row, col) -> (x, y)


def rasterize(c: np.ndarray, wl: np.ndarray, wr: np.ndarray, margin: float = 1.0):
    """(image rows top-down, 254 free / 0 occupied, origin (x, y) of its lower-left cell) at RES"""
    reach = float(max(wl.max(), wr.max())) + margin
    x0, y0 = c.min(0) - reach
    x1, y1 = c.max(0) + reach
    W, H = int(math.ceil((x1 - x0) / RES)), int(math.ceil((y1 - y0) / RES))
    gx, gy = np.meshgrid(x0 + (np.arange(W) + 0.5) * RES, y0 + (np.arange(H) + 0.5) * RES)
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    _t, nrm, _k = frame(c)
    d, j = cKDTree(c).query(P)
    lat = ((P - c[j]) * nrm[j]).sum(1)
    free = (lat <= wl[j]) & (lat >= -wr[j]) & (d <= np.maximum(wl[j], wr[j]) + 0.02)
    free = ndimage.binary_opening(free.reshape(H, W), iterations=1)
    # a duct ends round: wall slivers thinner than a duct (where two inside corners meet) are cut away; a wall between
    # two passes is at least WALL_M thick (the clearance cap) and stays
    r_c = int(round(0.12 / RES))
    yy, xx = np.mgrid[-r_c:r_c + 1, -r_c:r_c + 1]
    walls = ndimage.binary_opening(~free, structure=(xx ** 2 + yy ** 2) <= r_c ** 2)
    free = ~walls
    return np.where(free, 254, 0).astype(np.uint8)[::-1], (float(x0), float(y0))   # ROS: row 0 is the top


def clearance(c: np.ndarray, window_m: float) -> np.ndarray:
    """[m] at every sample, the distance to the nearest sample of ANOTHER part of the lap (more than `window_m` along it)"""
    n = len(c)
    d, j = cKDTree(c).query(c, k=min(n, 400))
    arc = np.abs(j - np.arange(n)[:, None])
    arc = np.minimum(arc, n - arc) * DS
    dd = np.where(arc > window_m, d, np.inf).min(1)
    return np.where(np.isfinite(dd), dd, 50.0)


def save(name: str, img: np.ndarray, origin, out_dir: str = OUT_DIR, meta: Optional[dict] = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.pgm"), "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())
    yml = os.path.join(out_dir, f"{name}.yaml")
    with open(yml, "w") as f:
        f.write(f"image: {name}.pgm\nresolution: {RES}\norigin: [{origin[0]:.4f}, {origin[1]:.4f}, 0.0]\n"
                f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\nmode: trinary\n")
    if meta is not None:
        json.dump(meta, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1, ensure_ascii=False)
    return yml


def make(kind: str, seed: int, name: str, out_dir: str = OUT_DIR, tries: int = 80):
    """generate, rasterise and save one map; returns (yaml path, summary, (centre line, left, right half widths))"""
    spec = KINDS[kind]
    rng = np.random.default_rng(seed + 1000 * (list(KINDS).index(kind) + 1))
    for attempt in range(tries):
        hall, spines, per = spine(kind, rng)
        nearest, lane_at = segments(kind, spines, per, rng)

        def radius(p_spine, p):
            h_, i_ = lane_at(p_spine, p)
            return h_ + i_
        try:
            raw = loop_round(spines, radius, hall)
        except ValueError:
            continue
        c = _resample(raw, DS)
        c = _resample(np.stack([_circ_gauss(c[:, 0], 0.3 / DS), _circ_gauss(c[:, 1], 0.3 / DS)], 1), DS)
        n = len(c)
        # the lane at each centre-line point: that of the side of the spine it runs along (its nearest spine point)
        tree_ = cKDTree(np.concatenate([np.stack([a_ + (b_ - a_) * t for t in np.linspace(0, 1, 8)])
                                        for sp in spines for a_, b_ in zip(sp[:-1], sp[1:])]))
        sp_pts = tree_.data[tree_.query(c)[1]]
        w = _circ_gauss(lane_at(sp_pts, c)[0], 1.0 / DS)            # sections ease into each other over ~2 m
        w = w * (1.0 + 0.06 * _noise(n, 3.0, rng))
        for _ in range(int(spec.get("pinches", 0))):              # a few places the hoses squeeze the lane
            at = rng.integers(n)
            ar = np.minimum(np.abs(np.arange(n) - at), n - np.abs(np.arange(n) - at)) * DS
            w = w - 0.08 * np.exp(-0.5 * (ar / 0.6) ** 2)
        asym = np.clip(0.10 * _noise(n, 3.0, rng), -0.2, 0.2)    # the line is not in the middle of the lane
        wl, wr = w * (1 + asym), w * (1 - asym)
        wl = wl + 0.015 * _noise(n, 0.25, rng); wr = wr + 0.015 * _noise(n, 0.25, rng)   # a SLAM wall is ragged
        cap = (clearance(c, 4.0 * float(w.max()) + 1.0) - WALL_M) / 2.0 - 0.02
        cap = np.minimum(cap, _circ_gauss(ndimage.minimum_filter1d(cap, size=int(3.0 / DS), mode="wrap"), 0.4 / DS))
        wl = np.minimum(wl, cap); wr = np.minimum(wr, cap)
        if float(min(wl.min(), wr.min())) < MIN_HALF - 0.05:
            continue                                             # two parts of the lap too close: another layout
        wl = np.clip(wl, MIN_HALF, None); wr = np.clip(wr, MIN_HALF, None)
        _t, _n, kap = frame(c)
        r_min = float(1.0 / max(np.abs(_circ_gauss(kap, 0.3 / DS)).max(), 1e-6))
        if r_min < R_MIN:
            continue                                             # a turn tighter than the car can drive
        img, origin = rasterize(c, wl, wr)
        tw = wl + wr
        meta = {"kind": kind, "seed": seed, "attempt": attempt, "hall_m": [round(hall[0], 1), round(hall[1], 1)],
                "branches": len(spines) - 1, "length_m": round(n * DS, 1), "tightest_radius_m": round(r_min, 2),
                "track_width_m": {"min": round(float(tw.min()), 2), "p10": round(float(np.quantile(tw, 0.1)), 2),
                                  "median": round(float(np.median(tw)), 2), "max": round(float(tw.max()), 2)},
                "size_m": [round(img.shape[1] * RES, 1), round(img.shape[0] * RES, 1)]}
        return save(name, img, origin, out_dir, meta), meta, (c, wl, wr)
    raise RuntimeError(f"{kind} seed {seed}: no valid layout in {tries} tries")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m f1sim.mapgen", description=__doc__.split("\n")[0])
    ap.add_argument("--kind", choices=list(KINDS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    yml, meta, _ = make(a.kind, a.seed, a.name, a.out)
    print(json.dumps(dict(meta, yaml=yml), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
