"""Random closed tracks from a recipe of features: straights, corners, hairpins, chicanes, slaloms.

Pick which features a track may contain and how many of each, a size and a lane width, and a
seed; `generate` returns a closed centreline as an editable path (`(x, y, smooth)` vertices, the
same kind the 환경 page's 트랙 경로 tool draws) plus the props some features place. The scene it
becomes is then an ordinary `scene:<name>` map -- editable, drivable, trainable.

How a track is built
--------------------
A lap is a ring of *turns* joined by *runs*. The turns are drawn from the enabled turn features
(90° corner, gentle 45°, hairpin 180°, a -90° corner that folds the loop back on itself) until
their signed angles sum to one full turn; the runs between them are drawn from the enabled run
features (straight, chicane, fast / slow slalom, cone slalom). Every run is a displacement along
the current heading of a length `L` plus a fixed lateral term, so closing the loop is a linear
condition on the run lengths: random lengths are projected onto the two closure equations, clamped
to each feature's minimum, and the loop is rejected when it self-intersects or two parts come
closer than a lane plus a hose. Retries change the turn order and the lengths; the seed fixes the
whole sequence, so `(recipe, seed)` is reproducible.

Arcs are emitted as curve vertices (Catmull-Rom through points spaced ≤ 30°), everything else as
corner vertices with curve vertices where a feature bends, so the result is what a person would
have clicked in the editor -- a few dozen vertices, not a thousand samples.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Run features: what happens between two turns. `min_len` is in lane widths, so a chicane in a
#: wide lane needs a longer run; `weight` is the draw probability when the user leaves counts free.
RUN_FEATURES: Dict[str, dict] = {
    "straight":    {"label": "직선",        "min_len": 0.4, "weight": 3.0,
                    "hint": "아무것도 없는 구간. 고속 구간의 바탕"},
    "chicane":     {"label": "시케인",      "min_len": 4.0, "weight": 1.5,
                    "hint": "차선이 한 차선 폭쯤 옆으로 꺾였다가 이어지는 구간"},
    "slalom_fast": {"label": "고속 슬라럼", "min_len": 6.0, "weight": 1.0,
                    "hint": "긴 주기·작은 진폭의 물결. 속도를 유지하며 좌우로 흔드는 구간"},
    "slalom_slow": {"label": "저속 슬라럼", "min_len": 5.0, "weight": 1.0,
                    "hint": "짧은 주기·큰 진폭의 물결. 크게 감속해야 하는 구간"},
    "cone_slalom": {"label": "콘 슬라럼",   "min_len": 5.0, "weight": 0.8,
                    "hint": "직선 위에 표지 기둥을 일정 간격으로 세워 좌우로 비켜 가게 하는 구간"},
}
#: Turn features: a signed angle (degrees, positive = left) and a radius in lane widths.
TURN_FEATURES: Dict[str, dict] = {
    "corner":   {"label": "90° 코너",     "angle": 90.0,  "radius": 1.2, "weight": 3.0,
                 "hint": "보통 코너"},
    "sweeper":  {"label": "완만한 커브",   "angle": 45.0,  "radius": 3.0, "weight": 1.5,
                 "hint": "반지름이 큰 45° 커브. 고속으로 도는 구간"},
    "hairpin":  {"label": "유턴 (헤어핀)", "angle": 180.0, "radius": 1.3, "weight": 1.2,
                 "hint": "180° 되돌아오는 구간"},
    "reverse":  {"label": "역방향 코너",   "angle": -90.0, "radius": 1.2, "weight": 1.0,
                 "hint": "바깥으로 꺾는 코너. 트랙이 접히는 모양을 만듭니다"},
}
SIZES: Dict[str, float] = {"소 (약 12 m)": 12.0, "중 (약 20 m)": 20.0, "대 (약 32 m)": 32.0}


@dataclass
class TrackRecipe:
    """Which features, how many, how big. A count of -1 means "any number, by weight"."""
    runs: Dict[str, int] = field(default_factory=lambda: {"straight": -1, "chicane": -1, "slalom_fast": -1})
    turns: Dict[str, int] = field(default_factory=lambda: {"corner": -1, "sweeper": -1, "hairpin": -1})
    size_m: float = 20.0                    # target long side of the track's bounding box
    lane_width: float = 1.6
    hose: float = 0.33                      # duct hose diameter (band width of the two hoses)
    n_turns: Tuple[int, int] = (4, 8)       # how many turn features a lap may have
    margin_m: float = 1.5                   # free floor around the track inside the canvas

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "TrackRecipe":
        r = TrackRecipe()
        for k, v in d.items():
            if hasattr(r, k):
                setattr(r, k, tuple(v) if k == "n_turns" else v)
        return r


@dataclass
class GeneratedTrack:
    points: List[Tuple[float, float, bool]]     # closed centreline, (x, y, smooth)
    lane_width: float
    hose: float
    props: List[dict]                            # {"style", "x", "y", "yaw", "dims"}
    features: List[str]                          # the lap, in order: run / turn keys
    seed: int
    attempts: int
    bounds: Tuple[float, float, float, float]

    @property
    def length_m(self) -> float:
        from .scene import polyline_length, sample_path
        return polyline_length(sample_path(self.points, True), True)

    def to_scene(self, name: str):
        """A `SceneDoc` with the track as one closed track path, the props placed, and the canvas
        fitted around it (`margin_m` of floor, a tall wall on the border)."""
        from .scene import SceneDoc
        x0, y0, x1, y1 = self.bounds
        doc = SceneDoc.new_blank(name, (x1 - x0), (y1 - y0), duct_height=self.hose)
        doc.origin = (float(x0), float(y0))
        doc.paint_border("tall", 0.15)
        doc.add_path("track", self.points, self.lane_width, closed=True, band=self.hose)
        for p in self.props:
            doc.add_prop(p["style"], p["x"], p["y"], p.get("yaw", 0.0), seed=len(doc.props), dims=dict(p.get("dims", {})))
        doc.source = {"map": None, "created": doc.source.get("created"), "generator": "trackgen",
                      "seed": self.seed, "features": list(self.features)}
        return doc


# ==================================================================== feature geometry
def _arc(angle_deg: float, radius: float) -> Tuple[List[Tuple[float, float, bool]], float, float]:
    """Points of a turn of `angle_deg` (left positive) starting at the origin heading +x, as
    curve vertices every <= 30°, plus the end point and the end heading (rad)."""
    a = math.radians(angle_deg)
    n = max(2, int(math.ceil(abs(angle_deg) / 30.0)))
    cy = radius if a > 0 else -radius             # centre of the arc, to the left or right
    pts = []
    for i in range(1, n + 1):
        t = a * i / n
        # position on a circle of `radius` about (0, cy), starting at the origin heading +x
        x = radius * math.sin(abs(t))
        y = cy - cy * math.cos(t)
        pts.append((x, y, True))
    end = pts[-1]
    pts[-1] = (end[0], end[1], True)
    return pts, a, radius


def _run(kind: str, L: float, w: float, rng: np.random.Generator) -> Tuple[List[Tuple[float, float, bool]], float, List[dict]]:
    """Vertices of a run of length `L` starting at the origin heading +x (the origin itself is not
    included), its lateral end offset, and any props (in the same local frame)."""
    props: List[dict] = []
    if kind == "straight":
        return [(L, 0.0, False)], 0.0, props
    if kind == "chicane":
        s = w * float(rng.uniform(0.8, 1.3)) * (1 if rng.random() < 0.5 else -1)
        a, b = 0.35 * L, 0.65 * L
        return [(a, 0.0, True), (b, s, True), (L, s, False)], s, props
    if kind in ("slalom_fast", "slalom_slow"):
        if kind == "slalom_fast":
            period = w * float(rng.uniform(3.8, 5.0))
            amp = w * float(rng.uniform(0.35, 0.6))
        else:
            period = w * float(rng.uniform(2.6, 3.2))
            amp = w * float(rng.uniform(0.4, 0.7))
        n = max(1, int(L / period))
        period = L / n
        # the weave's curvature radius at a peak is P^2 / (4 pi^2 A); below (lane/2 + hose) the
        # inner hoses of two bends fold into each other, so the amplitude is capped there
        r_min = 0.5 * w + 0.33 * 1.15
        amp = min(amp, period ** 2 / (4 * math.pi ** 2 * r_min))
        pts = []
        sign = 1 if rng.random() < 0.5 else -1
        for i in range(n):
            x0 = i * period
            pts.append((x0 + 0.25 * period, sign * amp, True))
            pts.append((x0 + 0.75 * period, -sign * amp, True))
        pts.append((L, 0.0, False))
        return pts, 0.0, props
    if kind == "cone_slalom":
        gap = w * float(rng.uniform(1.6, 2.2))
        n = max(2, int((L - 2 * w) / gap))
        gap = (L - 2 * w) / n
        for i in range(n + 1):
            props.append({"style": "marker_post", "x": w + i * gap, "y": 0.0, "yaw": 0.0, "dims": {}})
        return [(L, 0.0, False)], 0.0, props
    raise ValueError(f"unknown run feature {kind!r}")


# ==================================================================== the lap
def _pick_turns(recipe: TrackRecipe, rng: np.random.Generator) -> Optional[List[str]]:
    """A multiset of turn features whose signed angles sum to 360°, honouring fixed counts.

    Free features are drawn by weight until the count budget is used; a draw is kept when the
    angles close a lap. When the fixed features already close the lap on their own (two
    hairpins, four corners) that lap is allowed even below the usual minimum turn count."""
    enabled = [k for k, c in recipe.turns.items() if c != 0 and k in TURN_FEATURES]
    if not enabled:
        return None
    fixed = [k for k, c in recipe.turns.items() if c > 0 and k in TURN_FEATURES for _ in range(c)]
    fixed_total = sum(TURN_FEATURES[k]["angle"] for k in fixed)
    free = [k for k in enabled if recipe.turns[k] < 0]
    weights = np.asarray([TURN_FEATURES[k]["weight"] for k in free], float)
    lo, hi = recipe.n_turns
    if abs(fixed_total - 360.0) < 1e-6 and len(fixed) >= 2:
        lo = min(lo, len(fixed))
    if len(fixed) > hi:
        hi = len(fixed)
    if not free:
        if abs(fixed_total - 360.0) < 1e-6 and len(fixed) >= 2:
            seq = list(fixed)
            rng.shuffle(seq)
            return seq
        return None
    lo_s = max(2, len(fixed), min(lo, 3))
    for _ in range(2000):
        n_total = int(rng.integers(lo_s, hi + 1))
        seq = list(fixed) + [str(rng.choice(free, p=weights / weights.sum())) for _ in range(n_total - len(fixed))]
        if abs(sum(TURN_FEATURES[k]["angle"] for k in seq) - 360.0) < 1e-6:
            rng.shuffle(seq)
            return seq
    return None


def _pick_runs(recipe: TrackRecipe, n: int, rng: np.random.Generator) -> Optional[List[str]]:
    enabled = [k for k, c in recipe.runs.items() if c != 0 and k in RUN_FEATURES]
    if not enabled:
        enabled = ["straight"]
    fixed = [k for k, c in recipe.runs.items() if c > 0 and k in RUN_FEATURES for _ in range(c)]
    if len(fixed) > n:
        return None
    free = [k for k in enabled if recipe.runs.get(k, -1) < 0] or ["straight"]
    weights = np.asarray([RUN_FEATURES[k]["weight"] for k in free], float)
    seq = list(fixed) + [str(rng.choice(free, p=weights / weights.sum())) for _ in range(n - len(fixed))]
    rng.shuffle(seq)
    return seq


def _segments_intersect(p, q, r, s) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    o1, o2, o3, o4 = orient(p, q, r), orient(p, q, s), orient(r, s, p), orient(r, s, q)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def _self_clear(poly: np.ndarray, clearance: float) -> bool:
    """No crossing, and no two non-neighbouring samples closer than `clearance`."""
    n = len(poly)
    if n < 4:
        return False
    P = np.vstack([poly, poly[:1]])
    # crossings (coarse: every 3rd segment against every 3rd, then exact near hits)
    step = max(1, n // 400)
    idx = list(range(0, n, step))
    for a in idx:
        for b in idx:
            if b <= a + 1 or (a == 0 and b == n - 1):
                continue
            if _segments_intersect(P[a], P[a + step] if a + step <= n else P[n], P[b], P[b + step] if b + step <= n else P[n]):
                return False
    # proximity of non-neighbouring parts
    d = np.linalg.norm(poly[:, None, :] - poly[None, :, :], axis=2)
    L = np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(L)[:-1]])
    total = float(L.sum())
    da = np.abs(arc[:, None] - arc[None, :])
    da = np.minimum(da, total - da)
    # samples closer along the path than 1.5 clearances are neighbours (a bend's own inside),
    # not a self-approach; everything else must keep a lane plus a hose apart
    near_in_arc = da < 1.5 * clearance
    return not bool((d[~near_in_arc] < clearance).any())


def generate(recipe: TrackRecipe, seed: int = 0, max_attempts: int = 300) -> GeneratedTrack:
    """A random closed track for `recipe`; raises `RuntimeError` when no lap could be closed."""
    from .scene import sample_path
    rng = np.random.default_rng(int(seed))
    w = float(recipe.lane_width)
    clearance = w + float(recipe.hose) + 0.15
    from collections import Counter
    reasons: Counter = Counter()
    last_reason = "no attempt"
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            reasons[last_reason] += 1
        turns = _pick_turns(recipe, rng)
        if turns is None:
            raise RuntimeError("켜 둔 코너 항목으로는 한 바퀴(360°)를 만들 수 없습니다. 90° 코너나 유턴을 켜 주세요.")
        runs = _pick_runs(recipe, len(turns), rng)
        if runs is None:
            last_reason = "run count"
            continue
        # headings after each turn; run i goes from turn i-1 to turn i. A turn's chord scales
        # with its radius, so the radii join the run lengths as unknowns of the closure.
        heading = 0.0
        dirs, norms, unit_chords = [], [], []
        for k in turns:
            dirs.append((math.cos(heading), math.sin(heading)))
            norms.append((-math.sin(heading), math.cos(heading)))
            pts, a, _r = _arc(TURN_FEATURES[k]["angle"], 1.0)
            c, s_ = math.cos(heading), math.sin(heading)
            chord = pts[-1]
            unit_chords.append((chord[0] * c - chord[1] * s_, chord[0] * s_ + chord[1] * c))
            heading += a
        r_nom = [TURN_FEATURES[k]["radius"] * w for k in turns]
        r0 = np.asarray([r * float(rng.uniform(0.7, 1.8) if k == "hairpin" else rng.uniform(0.85, 1.25))
                         for k, r in zip(turns, r_nom)])
        r_lo = np.asarray([max(0.55 * w, 0.6 * r) for r in r_nom])
        r_hi = np.asarray([2.6 * r for r in r_nom])
        # run geometry in local frames: lengths L_i free, lateral offsets fixed by the feature draw
        n = len(runs)
        local = []
        lat = []
        min_len = []
        run_seeds = [int(v) for v in rng.integers(0, 2**31, size=len(runs))]
        for i, k in enumerate(runs):
            L_guess = w * RUN_FEATURES[k]["min_len"]
            # the feature's random parameters come from its own seed, so the draw made here for
            # the closure equations is the draw made again when the lap is assembled below
            pts, off, props = _run(k, L_guess, w, np.random.default_rng(run_seeds[i]))
            local.append((k, pts, props, L_guess))
            lat.append(off)
            min_len.append(w * RUN_FEATURES[k]["min_len"])
        # closure: sum_i L_i d_i + sum_j r_j chord_j + sum_i lat_i n_i = 0
        D = np.asarray(dirs, float).T                             # (2, n)
        Cr = np.asarray(unit_chords, float).T                     # (2, n)
        const = np.asarray([[lat[i] * norms[i][0], lat[i] * norms[i][1]] for i in range(n)]).sum(0)
        scale = float(recipe.size_m) / 3.2
        L0 = np.asarray([max(m, scale * float(rng.uniform(0.5, 1.4))) for m in min_len])
        mins = np.asarray(min_len)
        from scipy.optimize import lsq_linear
        lam_L, lam_r = 0.03, 0.3                    # radii are cheap to move, lengths less so
        A = np.vstack([np.hstack([D, Cr]), np.hstack([lam_L * np.eye(n), np.zeros((n, n))]),
                       np.hstack([np.zeros((n, n)), lam_r * np.eye(n)])])
        b_vec = np.concatenate([-const, lam_L * L0, lam_r * r0])
        lo_b = np.concatenate([mins, r_lo])
        hi_b = np.concatenate([np.full(n, np.inf), r_hi])
        ok = False
        try:
            sol = lsq_linear(A, b_vec, bounds=(lo_b, hi_b), method="bvls")
            xv = np.asarray(sol.x, float)
            M = np.hstack([D, Cr])
            free = (xv > lo_b + 1e-9) & (xv < hi_b - 1e-9)
            if free.sum() >= 2:
                Mf = M[:, free]
                xv[free] += Mf.T @ np.linalg.solve(Mf @ Mf.T + 1e-12 * np.eye(2), -(M @ xv + const))
                ok = bool(np.linalg.norm(M @ xv + const) < 1e-6 and (xv >= lo_b - 1e-9).all()
                          and (xv <= hi_b + 1e-9).all())
            L, radius = xv[:n], list(xv[n:])
        except Exception:
            ok = False
        if not ok:
            last_reason = "closure"
            continue
        # assemble the loop
        pts_world: List[Tuple[float, float, bool]] = [(0.0, 0.0, False)]     # the start, a corner
        props_world: List[dict] = []
        x = y = 0.0
        heading = 0.0
        features: List[str] = []
        for i in range(n):
            k, _pts, _props, _Lg = local[i]
            pts, off, props = _run(k, float(L[i]), w, np.random.default_rng(run_seeds[i]))
            c, s_ = math.cos(heading), math.sin(heading)
            for px, py, sm in pts:
                pts_world.append((x + px * c - py * s_, y + px * s_ + py * c, sm))
            for p in props:
                props_world.append({**p, "x": x + p["x"] * c - p["y"] * s_, "y": y + p["x"] * s_ + p["y"] * c,
                                    "yaw": heading})
            ex, ey = pts[-1][0], pts[-1][1]
            x, y = x + ex * c - ey * s_, y + ex * s_ + ey * c
            features.append(k)
            # the turn
            k_t = turns[i]
            apts, a, _r = _arc(TURN_FEATURES[k_t]["angle"], radius[i])
            for px, py, sm in apts:
                pts_world.append((x + px * c - py * s_, y + px * s_ + py * c, sm))
            ex, ey = apts[-1][0], apts[-1][1]
            x, y = x + ex * c - ey * s_, y + ex * s_ + ey * c
            heading += a
            features.append(k_t)
        # the last vertex coincides with the start: drop it (closed path)
        if math.hypot(pts_world[-1][0], pts_world[-1][1]) < 1e-6:
            pts_world = pts_world[:-1]
        else:
            last_reason = "closure residual"
            continue
        # drop near-duplicate consecutive vertices (a zero-length straight leaves one)
        cleaned: List[Tuple[float, float, bool]] = []
        for p in pts_world:
            if not cleaned or math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) > 1e-3:
                cleaned.append(p)
        if math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) < 1e-3:
            cleaned.pop()
        if len(cleaned) < 4:
            last_reason = "too few vertices"
            continue
        poly = sample_path(cleaned, True, step=0.15)
        if not _self_clear(poly, clearance):
            last_reason = "self-intersection"
            continue
        # fit the requested size: scale about the centroid so the long side matches size_m
        ext = poly.max(0) - poly.min(0)
        k_scale = float(recipe.size_m) / max(float(ext.max()), 1e-6)
        if k_scale < 0.55 or k_scale > 1.8:
            last_reason = "size"
            continue
        # scaling changes the lane width relation only through radii: keep it if the smallest
        # radius stays drivable (>= 0.55 lane widths)
        if min(radius) * k_scale < 0.55 * w:
            last_reason = "radius"
            continue
        cx, cy = poly.mean(0)
        pts_final = [((px - cx) * k_scale, (py - cy) * k_scale, sm) for px, py, sm in cleaned]
        props_final = [{**p, "x": (p["x"] - cx) * k_scale, "y": (p["y"] - cy) * k_scale} for p in props_world]
        poly = sample_path(pts_final, True, step=0.1)
        m = float(recipe.margin_m) + 0.5 * w + float(recipe.hose)
        bounds = (float(poly[:, 0].min() - m), float(poly[:, 1].min() - m),
                  float(poly[:, 0].max() + m), float(poly[:, 1].max() + m))
        # centerline direction: counter-clockwise, as the simulator expects
        area = 0.5 * float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]))
        if area < 0:
            pts_final = list(reversed(pts_final))
        return GeneratedTrack(pts_final, w, float(recipe.hose), props_final, features, int(seed), attempt, bounds)
    reasons[last_reason] += 1
    why = ", ".join(f"{k} ×{v}" for k, v in reasons.most_common())
    raise RuntimeError(f"{max_attempts}번 시도했지만 겹치지 않는 트랙을 만들지 못했습니다 ({why}). "
                       f"규모를 키우거나 항목 수를 줄여 보세요.")


# ==================================================================== CLI
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m f1sim.trackgen",
                                 description="Random closed tracks as editable scenes (scene:<name>).")
    ap.add_argument("--name", default="rand", help="scene name; with --count, a _<seed> suffix is added")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=1, help="how many scenes (seeds seed..seed+count-1)")
    ap.add_argument("--size", type=float, default=20.0, help="long side of the track in metres")
    ap.add_argument("--lane", type=float, default=1.6, help="lane width in metres")
    ap.add_argument("--hose", type=float, default=0.33)
    ap.add_argument("--runs", default="straight,chicane,slalom_fast",
                    help=f"run features, comma separated, optional =count: {','.join(RUN_FEATURES)}")
    ap.add_argument("--turns", default="corner,sweeper,hairpin",
                    help=f"turn features, comma separated, optional =count: {','.join(TURN_FEATURES)}")
    ap.add_argument("--dry-run", action="store_true", help="print the lap, save nothing")
    args = ap.parse_args(argv)

    def parse(spec: str, table: dict) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for item in [s for s in spec.split(",") if s.strip()]:
            k, _, c = item.partition("=")
            if k.strip() not in table:
                ap.error(f"unknown feature {k!r}; known: {', '.join(table)}")
            out[k.strip()] = int(c) if c else -1
        return out

    recipe = TrackRecipe(runs=parse(args.runs, RUN_FEATURES), turns=parse(args.turns, TURN_FEATURES),
                         size_m=args.size, lane_width=args.lane, hose=args.hose)
    results = []
    for seed in range(args.seed, args.seed + max(1, args.count)):
        g = generate(recipe, seed)
        name = args.name if args.count <= 1 else f"{args.name}_{seed}"
        rec = {"name": name, "seed": seed, "attempts": g.attempts, "length_m": round(g.length_m, 2),
               "features": g.features, "props": len(g.props), "bounds": [round(v, 2) for v in g.bounds]}
        if not args.dry_run:
            doc = g.to_scene(name)
            rec["dir"] = doc.save()
        results.append(rec)
    print(json.dumps(results if args.count > 1 else results[0], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
