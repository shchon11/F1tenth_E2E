"""A fresh obstacle layout for every environment at every reset, placed as analytic props.

Why this exists
---------------
Every finetune so far improves on the training tracks and not on held-out ones. The training set is
204 tracks whose obstacle layouts are *fixed*: `+rlobs`, `+obs` and `+hard<seed>` rasterise their
boxes into the occupancy grid once, at load. A layout that never changes can be learned, and the
progress reward (+4/s, `docs/research/reward-audit-2026-09-13.md`) pays for learning it -- speed
through a known layout is worth the same as speed through a seen one, and on an unseen layout the
same speed is a collision. So the layout has to change at every reset, which means it cannot be
baked into a grid: re-rasterising and re-running the erosion connectivity proof per env per reset
is seconds of CPU work for something that has to happen thousands of times a second on the GPU.

What replaces the bake
----------------------
The same six patterns `hard_obstacles.py` draws -- gate, diagonal, chicane, apex, cluster, scatter,
with their sizes and their gap rules -- placed as `f1sim.props` **props**: convex prisms that the
LiDAR and the contact test already handle analytically (`prop_math.py`), with no grid involved.
Plus a seventh, `center`, which `hard_obstacles` has no equivalent of: one wide piece in the middle
of the lane with a way past on each side (`KINDS`). A layout is then a few dozen numbers per env (a pose and a shape index per piece), which is a batched
gather on the GPU.

The erosion proof is replaced by *construction*, not dropped. `hard_obstacles` draws a pattern, then
erodes the free space by 0.25 m and checks that the lane still connects across it; a draw that fails
is undone and redrawn. That loop cannot be batched. Here the free gap is instead guaranteed before
anything is placed:

* every piece of a row pattern lies inside a band of width `span` measured from **one** wall, so the
  free lateral space beside it is `lane width - span` by construction;
* `span` is chosen as `min(width - g, span_max)` where `g >= max(1.2 m, 0.55 * width)` is the drawn
  gap, so the free space is never less than `g`;
* `width` is the **narrowest** the lane gets anywhere *that* pattern reaches (`REACH`: 0.6 m for a
  gate, 4.0 m for a chicane's second row), while the band is measured from the wall at each piece's
  own arc index. Both errors go the same way: the realised gap is greater than or equal to the drawn
  one, never less.
* a scatter object is placed so that at least `GAP_MIN` of lane is left on one side of it.

`tests/test_procedural_obstacles.py` measures the realised gap on 1000 draws over the catalogue maps
rather than trusting that argument.

What the teacher sees
---------------------
The raceline teacher that drives the opponent cars is blind to props -- it is pure pursuit on a
raceline computed from the grid, and the props are not in the grid. Driving it into a crate is not a
lesson about anything. So when a teacher is installed, the gap is additionally required to contain
the **raceline corridor**: the lateral band the raceline occupies over the arc one piece covers,
widened by the car's half-width and `raceline_margin`. It bounds the band at the pattern's index and
then pushes every piece clear of the corridor at its *own* index (`_clear_raceline`), so the line the
teacher drives is clear by the same construction that guarantees the gap.
This is the first of the two options the contract offers ("place patterns only where the raceline is
not"); the second (widening the teacher's *event* offset clamp to know about props) does not help,
because that clamp only bounds a lateral offset away from the line and the problem is a prop sitting
on the line itself.

The cost is stated rather than hidden: with a teacher installed, the gap is always where the racing
line is. The layout still moves every reset -- the arc positions, the patterns, the sizes and which
wall is blocked all change -- but a policy that could already find the racing line would find the
gap. With `--race-size 1` there is no teacher and no corridor, and the gap is free to be anywhere.

What does not see these props
-----------------------------
Everything that reads the occupancy grid or its distance field:

* `reward_proximity` (wall gap from the EDT) and `reward_plan_clearance` (plan points against the
  EDT) do not price a prop;
* `StepResult.wall_dist` is the distance to the nearest grid wall, so a car alongside a crate reads
  the same clearance as a car alone on the lane;
* the raceline builder and the teacher's speed profile were computed before any of this existed.

The collision test, the LiDAR and the spawn rejection *do* see them, which is what makes the layout
something the policy has to look at. See `docs/training.md`.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import props as _props
from .prop_math import section_halfplanes

def _is_compiling() -> bool:
    fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    return bool(fn()) if fn is not None else False


# ---------------------------------------------------------------- the pattern rules
# Every number here is `hard_obstacles.py`'s, and through it the user's hand-built scenes
# (`~/f1sim_scenes/scene_0912_*`). They are repeated rather than imported because that module is
# numpy/scipy over a grid and this one is torch over a batch; sharing the constants is the whole of
# what the two have in common. `tests/test_procedural_obstacles.py` pins them equal.
PATTERNS: Tuple[str, ...] = ("gate", "diagonal", "chicane", "apex", "cluster", "scatter")
#: What this module draws: `hard_obstacles`' six patterns, plus `center` -- one wide piece standing in
#: the middle of the lane with a way past on each side. None of the six makes that: every row grows
#: from a wall, and scatter's objects are small. It is the console's 랜덤 · 중간 (`asset_obstacles`,
#: family "line"), and s912 and s913 hit that block on 40 of 41 contacts, at walking pace, unable to
#: pick a side. `PATTERNS` stays `hard_obstacles`' list; `last_kind` indexes this one.
KINDS: Tuple[str, ...] = PATTERNS + ("center",)
GAP_FRAC = (0.55, 0.75)   # open share of the lane beside a pattern
GAP_MIN = 1.20            # [m] never less than this, whatever the lane width
CENTER_SIDE_MIN = 0.70    # [m] a centre piece's narrow way past: a car and its margin, as the console's
                          # line family requires. The other side keeps GAP_MIN, like every pattern.
CENTER_MAX = 0.62         # [m] widest centre piece: the low crate stack
ROW_GAP = (0.05, 0.25)    # [m] daylight between adjacent boxes of a row -- narrower than the car
SMALL_SHARE = 0.35        # share of apex and cluster pieces drawn small
MIN_SPACING = 6.0         # [m] of arc between two patterns
BOX_W, BOX_D = 0.36, 0.30 # [m] the cardboard box: across the lane, along it

# ---------------------------------------------------------------- this module's own numbers
MIN_SPAN = 0.30           # [m] a blocked band thinner than this is not a pattern, it is litter
MIN_PIECE = 0.12          # [m] `_block`'s smallest box: below this nothing is placed
#: [m] of arc each pattern reaches forward from its own index, longest piece included. The lane
#: width a pattern is allowed to block is the narrowest the lane gets over *its* reach, so a gate --
#: one row, at one index -- is not held to what the lane does three metres later. Taking the widest
#: reach for all six costs patterns on a narrow map: `control_1400`'s lane is 2.00 m at the median
#: but 1.85 m as a 4 m running minimum, and 1.20 m of that is the gap.
REACH = {"gate": 0.6, "diagonal": 2.6, "chicane": 4.0, "apex": 0.6, "cluster": 0.9, "scatter": 3.2,
         "center": 0.6}
WINDOW = 4.0              # [m] the longest of them: chicane row 2 at 3.5 m plus a crate
PIECE_WINDOW = 0.8        # [m] of arc one *piece* covers, centred: its own depth plus the yaw
                          # jitter. The raceline corridor is taken over this and not over WINDOW --
                          # see `set_raceline`, where taking it over the pattern's whole reach was
                          # measured to block the lane outright.
LANE_LIMIT = 4.0          # [m] how far the lane half-width march looks before giving up
YAW_JITTER = 0.12         # [rad] a piece is set down by hand, not surveyed. Bounded, and every
                          # piece's across-lane extent is inflated by it before the gap arithmetic,
                          # so the guarantee holds for the rotated box and not just the nominal one.
PIECES_PER_PATTERN = 6    # slots a pattern may fill. A chicane wants two rows of three; a row that
                          # needs more pieces than this is truncated at its *inner* end, which can
                          # only make the gap wider.
CANDIDATES = 8            # stratified arc candidates per pattern slot (apex takes the sharpest)
PIECE_BUDGET = 3.5        # mean pieces a pattern actually places, used to size the per-env slot
                          # count when the caller does not set one. Measured over the six patterns.
CONTACT_SLOTS = 8         # slots the contact tests look at (see `ProceduralObstacles.near`)
#: `line_share` (11-03): a pattern slot turned into ONE piece standing on the racing line, its centre the line's lateral
#: offset +- this (moved toward a wall only as far as one side needs to keep GAP_MIN). The six wall-measured patterns cannot reach a line that runs mid-lane in a wide stretch (the band is
#: `width - gap` from one wall): on Korea 2025 the line-blocking pieces per 2 % of lap were 200-390 on most of it and
#: 21-42 over 84-96 % of the lap (the wide stretch before the start line), and every teacher hit exactly those crates
LINE_JITTER = 0.30        # [m]
#: `place="uniform"` (11-03, user: "고르게 뽑히라고 uniform하게. 다만 지나갈 차 폭 고려해서 아예 못지나가게만 배치하지 말고. 사실상
#: 거의 전 영역에 고르게 뽑혀야해"): the patterns above put 42-87 % of pieces within 0.3 m of a wall and only 8-26 % on the line.
#: Uniform draws every piece anywhere on the lane, area-weighted, and keeps only the one guarantee: in every cross-section
#: from PASS_SMEAR before a piece to PASS_SMEAR after it, some straight way of PASS_MIN is left (checked on a lane grid
#: of PASS_BIN x PASS_CELL as the pieces are put down; a piece that would close the way is not placed).
PASS_MIN = 0.70           # [m] a car (0.30 m) and 0.20 m each side -- CENTER_SIDE_MIN's narrow way
PASS_SMEAR = 0.50         # [m] of arc before and after a piece over which that way must run straight, about a car length
PASS_BIN = 0.10           # [m] of arc per cross-section of the passage grid
PASS_CELL = 0.05          # [m] across the lane per cell of the passage grid
U_CANDIDATES = 8          # area-weighted resampling candidates per piece
CURVE_INSET = 0.08        # [m] the blocked band is inset by this before anything is placed.
                          # A piece is set down square to the tangent at its own centerline index,
                          # and over the 0.30-0.44 m it is long the frame turns: on a 0.8 m-radius
                          # hairpin -- and `apex` deliberately puts a pattern at the sharpest corner
                          # it can find -- the tangent rotates 0.27 rad over half a crate, which
                          # moves its corner up to 0.09 m across the lane relative to the frame the
                          # gap was measured in. Without the inset that came out of the gap:
                          # measured over 1000 layouts on the catalogue maps, 99.5 % kept 1.20 m and
                          # the worst was 1.076 m. With it, and with the scatter side drawn per
                          # pattern, 100 % and a worst of 1.31 m.

#: Row pieces: what a row across the lane is built from, any size `hard_obstacles` would draw.
#: Sorted by across-lane extent at build time; a row takes the largest that still fits.
ROW_SHAPES: Tuple[Tuple[str, dict], ...] = tuple(
    [("cardboard_box", dict(width=round(0.12 + 0.03 * i, 3), depth=BOX_D, height=0.30)) for i in range(9)]
    + [("wooden_crate", dict(width=w, depth=0.34, height=0.36)) for w in (0.24, 0.30, 0.36, 0.42)]
    + [("steel_drum", dict(radius=0.145, height=0.44, facets=8)),
       ("barrier_block", dict(width=0.54, depth=0.26, height=0.22)),
       ("crate_stack_low", dict(width=0.62, depth=0.44, height=0.26))]
)

#: Small pieces: 0.10-0.25 m, the user's point that avoiding only big boxes is not avoiding.
#: Both lists build their drums with 8 facets rather than the default 16: a prop costs the beam
#: tracer one `(B, N, k_pad)` pass per slot and `k_pad` is the widest footprint in the catalogue, so
#: a 16-sided drum would make *every* piece on every track cost twice what a box costs. Eight facets
#: is still a drum and keeps `k_pad` at 8.
SMALL_SHAPES: Tuple[Tuple[str, dict], ...] = tuple(
    [("cardboard_box", dict(width=s, depth=s, height=round(s * 0.9, 3))) for s in (0.15, 0.18, 0.21, 0.25)]
    + [("marker_post", dict(radius=r, height=0.34)) for r in (0.05, 0.07, 0.09, 0.11)]
    + [("steel_drum", dict(radius=0.11, height=0.30, facets=8))]
)

#: Box pieces (11-06, the user: "장애물 크기 좀 박스 크기로 해서 좀 해라" -- after a policy stood 38 s nose-on against a
#: 0.05-0.11 m marker post): the venue's obstacles were cardboard boxes about 0.3 m (10-25 bags; real-size 0.3 m boxes
#: are read as boxes, narrow 0.12-0.18 m ones as a car). `shapes="boxes"` draws only these. Appended after the other
#: two groups so the shape ids an existing bank stores keep their meaning.
BOX_SHAPES: Tuple[Tuple[str, dict], ...] = tuple(
    [("cardboard_box", dict(width=s, depth=s, height=0.30)) for s in (0.27, 0.30, 0.33, 0.36)]
    + [("wooden_crate", dict(width=w, depth=0.34, height=0.36)) for w in (0.30, 0.36, 0.42)]
)


@dataclass(frozen=True)
class Shape:
    """One catalogue entry: a prop at a fixed size, reduced to what the GPU needs.

    The half-planes are the prop's **declared envelope** (`props.Envelope`) -- "the convex prism the
    prop occupies", which the props module documents as what collision and the LiDAR are meant to
    use, and whose `bevel_tolerance` (12-31 mm here) is the measured bound on how far it stands
    outside the drawn surface. `+props` catalogue props are instead cut into four height bands and
    merged, which bounds a taper more tightly at the cost of up to three slots per prop; here every
    piece is one slot, so a layout of twenty pieces costs the tracer twenty passes and not sixty.
    """
    style: str
    dims: Tuple[Tuple[str, float], ...]
    n: np.ndarray             # (K, 2) outward unit normals, prop-local
    d: np.ndarray             # (K,) offsets, padding n = 0, d = +inf
    z0: float
    z1: float
    across: float             # across-lane extent, inflated by YAW_JITTER
    along: float              # along-lane extent of the nominal footprint
    mass: float = 0.0         # [kg]; 0 = immovable, which is what every prop was before this


#: [kg] what each catalogue style weighs. OURS, not measured: the user's brief was "장애물은 질량이
#: 10kg정도 되는 강체는 아니지만 준하는 정도", so the working set is around that with the light and
#: the heavy ends spread either side of it. The car is 3.74 kg, so a wooden crate at 10 kg takes
#: about a quarter of the impulse and a cardboard box takes most of it -- which is the point: the
#: policy should learn that some things can be brushed aside and some things stop you.
#:
#: A style with no entry is immovable, which is also what every prop was before this existed.
STYLE_MASS = {
    "cardboard_box": 1.2,
    "wooden_crate": 10.0,
    "steel_drum": 18.0,
    "barrier_block": 12.0,
    "crate_stack_low": 9.0,
    "marker_post": 1.5,
}

#: Deceleration of a shoved prop on the floor [m/s^2]: mu * g with mu ~ 0.5, so a crate given
#: 1 m/s travels about 0.10 m. A shove is a shove, not a bowling ball.
PROP_GROUND_DECEL = 4.9


def build_catalogue(k_pad: int, box_ids: Optional[List[int]] = None) -> Tuple[List[Shape], List[int], List[int]]:
    """Every shape, plus the indices of the row ladder (sorted by extent) and the small set; `box_ids`, when given, is
    filled with the indices of `BOX_SHAPES`."""
    shapes: List[Shape] = []
    row_ids: List[int] = []
    small_ids: List[int] = []
    box_sink: List[int] = box_ids if box_ids is not None else []
    for group, sink in ((ROW_SHAPES, row_ids), (SMALL_SHAPES, small_ids), (BOX_SHAPES, box_sink)):
        for style, dims in group:
            prop = _props.build(style, seed=0, **dims)
            env = prop.envelope
            n, d = section_halfplanes(env.footprint, k_pad)
            fp = np.asarray(env.footprint, float)
            x_span = float(fp[:, 0].max() - fp[:, 0].min())
            y_span = float(fp[:, 1].max() - fp[:, 1].min())
            sink.append(len(shapes))
            shapes.append(Shape(style, tuple(sorted(dims.items())), n, d, 0.0, float(env.height),
                                x_span + y_span * math.sin(YAW_JITTER), y_span,
                                float(STYLE_MASS.get(style, 0.0))))
    row_ids.sort(key=lambda i: shapes[i].across)
    return shapes, row_ids, small_ids


def catalogue_k_pad() -> int:
    """Widest footprint in the catalogue -- the padded half-plane count every slot pays for."""
    k = 3
    for group in (ROW_SHAPES, SMALL_SHAPES, BOX_SHAPES):
        for style, dims in group:
            k = max(k, len(_props.build(style, seed=0, **dims).envelope.footprint))
    return k


# ==================================================================== per-track geometry (numpy)
def _lane_halves(track, cl: np.ndarray, nrm: np.ndarray, limit: float = LANE_LIMIT):
    """(N,), (N,): free distance from each centerline point to the wall on the +nrm and -nrm side.

    The same march `hard_obstacles._lane_halves` does, vectorised over the whole centerline: step
    out in half cells until a cell is occupied or the grid ends, and report the last free distance.
    """
    res = float(track.resolution)
    ox, oy = float(track.origin[0]), float(track.origin[1])
    occ = track.occupancy
    H, W = occ.shape
    step = res * 0.5
    S = int(limit / step) + 1
    ds = np.arange(S) * step                                          # (S,)
    out = []
    for s in (1.0, -1.0):
        q = cl[:, None, :] + s * nrm[:, None, :] * ds[None, :, None]  # (N, S, 2)
        col = np.floor((q[..., 0] - ox) / res).astype(np.int64)
        row = np.floor((q[..., 1] - oy) / res).astype(np.int64)
        inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        blocked = np.ones(q.shape[:2], bool)
        blocked[inside] = occ[row[inside], col[inside]]
        any_b = blocked.any(1)
        first = np.where(any_b, blocked.argmax(1), S)
        out.append(np.clip((first - 1) * step, 0.0, limit))
    return out[0], out[1]


def _forward_window(x: np.ndarray, w: int, op: str) -> np.ndarray:
    """min / max of `x` over the `w` points starting at each index, wrapping at the lap."""
    w = max(1, int(w))
    n = len(x)
    idx = (np.arange(n)[:, None] + np.arange(w)[None, :]) % n
    v = x[idx]
    return v.min(1) if op == "min" else v.max(1)


def _centred_window(x: np.ndarray, w: int, op: str) -> np.ndarray:
    """min / max of `x` over the `w` points centred on each index, wrapping at the lap."""
    w = max(1, int(w))
    n = len(x)
    idx = (np.arange(n)[:, None] + np.arange(w)[None, :] - w // 2) % n
    v = x[idx]
    return v.min(1) if op == "min" else v.max(1)


def _curvature(cl: np.ndarray, ds: float) -> np.ndarray:
    """Signed curvature along the centerline: + turns left, so +normal is the inside of the corner."""
    from scipy import ndimage
    tang = np.roll(cl, -1, 0) - np.roll(cl, 1, 0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    heading = np.arctan2(tang[:, 1], tang[:, 0])
    dh = np.angle(np.exp(1j * (np.roll(heading, -2) - np.roll(heading, 2))))
    k = dh / (4.0 * ds + 1e-9)
    return ndimage.uniform_filter1d(k, size=max(3, int(1.0 / max(ds, 1e-6))), mode="wrap")


# ==================================================================== the drawer
class ProceduralObstacles:
    """Per-env prop layouts, redrawn on the GPU at every reset of that env.

    Owns five tensors shaped like `TrackTensors`'s own prop slots but keyed by **env** rather than
    by track -- `(B, C, ...)` -- which is what makes a layout per env rather than per map. They are
    allocated once and written in place, so a CUDA graph that captured them replays against the new
    layout instead of having to be re-recorded.
    """

    #: Carried across a captured teacher call (`viewer.graph_fastpath._state_owners`): the cull's
    #: running diagnostic totals, which the teachers' prop queries add to.
    GRAPH_STATE = ("_stat",)

    def __init__(self, tracks, B: int, gen: torch.Generator, *, density: float = 1.0,
                 fraction: float = 1.0, max_props: int = 0, raceline_margin: float = 0.25,
                 car_half_width: float = 0.15, count: str = "fixed", line_share: float = 0.0,
                 place: str = "pattern", bank: str = "", shapes: str = "all"):
        self.tr = tracks
        self.B = int(B)
        self.gen = gen
        shape_set = str(shapes)                          # `shapes` is reused below for the catalogue list
        self.device = tracks.device
        self.density = float(density)
        self.fraction = float(fraction)
        #: How many of the lap's patterns a layout gets. "fixed": all of them, every layout (every run before
        #: 2026-11-03). "uniform": a number drawn uniformly from 1 .. all, the kept ones a random subset of the
        #: lap's sectors. 11-03: at "fixed" every layout on Korea 2025 (43.4 m, density 1.5) had 7 patterns
        #: (13-18 pieces) or none, the car was clear of props within 10 m 34 % of the time, and the props policy
        #: drove a narrow centre line on the empty track too (7.18 s against the solo policy's 6.50 s).
        #: "max<N>" (11-06, the user: "너무 장애물 많아 평가환경에서", "2개임 대회에서는"): a bank layout's pieces, at most N of
        #: them, a random subset (a subset of a layout keeps its drivable way); the venue had two boxes
        self.keep = 0
        if count.startswith("max") and count[3:].isdigit():
            self.keep = int(count[3:])
            count = "fixed"
        if count not in ("fixed", "uniform"):
            raise ValueError(f"procedural count must be 'fixed', 'uniform' or 'max<N>', got {count!r}")
        self.count = count
        #: share of the drawn pattern slots that become one piece on the racing line (`LINE_JITTER`); needs `set_line`.
        #: Off while a raceline corridor is installed: the planner opponents are blind to props and drive that line.
        self.line_share = float(line_share)
        self.line_off = None
        #: "pattern": the seven wall-measured kinds (every run before 11-03 night); "uniform": see PASS_MIN
        if place not in ("pattern", "uniform"):
            raise ValueError(f"procedural place must be 'pattern' or 'uniform', got {place!r}")
        self.place = place
        self.raceline_margin = float(raceline_margin)
        self.car_half_width = float(car_half_width)
        if tracks.cl is None:
            raise ValueError("procedural obstacles need centerlines: every track in the set must have one")
        dev = self.device
        k_pad = int(tracks.k_pad)
        need = catalogue_k_pad()
        if k_pad < need:
            raise ValueError(f"track prop slots are padded to {k_pad} half-planes, the procedural "
                             f"catalogue needs {need}")
        box_ids: List[int] = []
        shapes, row_ids, small_ids = build_catalogue(k_pad, box_ids)
        self.shapes = shapes
        self.sh_n = torch.tensor(np.stack([s.n for s in shapes]), dtype=torch.float32, device=dev)
        self.sh_d = torch.tensor(np.stack([s.d for s in shapes]), dtype=torch.float32, device=dev)
        self.sh_z0 = torch.tensor([s.z0 for s in shapes], dtype=torch.float32, device=dev)
        self.sh_z1 = torch.tensor([s.z1 for s in shapes], dtype=torch.float32, device=dev)
        self.sh_mass = torch.tensor([s.mass for s in shapes], dtype=torch.float32, device=dev)
        self.row_id = torch.tensor(row_ids, dtype=torch.long, device=dev)
        self.row_across = torch.tensor([shapes[i].across for i in row_ids], dtype=torch.float32, device=dev)
        self.small_id = torch.tensor(small_ids, dtype=torch.long, device=dev)
        self.small_across = torch.tensor([shapes[i].across for i in small_ids], dtype=torch.float32, device=dev)
        self.sh_across = torch.tensor([s_.across for s_ in shapes], dtype=torch.float32, device=dev)
        self.sh_along = torch.tensor([s_.along for s_ in shapes], dtype=torch.float32, device=dev)
        self.line_pool = torch.cat([self.row_id, self.small_id]).unique()
        #: "all": the row and small pieces (every run before 11-06); "boxes": `BOX_SHAPES` only (`_redraw_uniform`, the line
        #: pieces and every layout a bank made with it)
        if shape_set not in ("all", "boxes"):
            raise ValueError(f"procedural shapes must be 'all' or 'boxes', got {shape_set!r}")
        self.shape_set = shape_set
        if shape_set == "boxes":
            self.line_pool = torch.tensor(box_ids, dtype=torch.long, device=dev)

        self.N = int(tracks.n_cl)
        self._build_track_geometry()

        if not self.density > 0:
            raise ValueError(f"procedural_density must be positive, got {density}: at zero no "
                             f"pattern is ever placed and the flag is silently the off run")
        L = tracks.length                                             # (T,)
        n_pat = torch.round(L / 10.0 * self.density)
        n_pat = torch.minimum(n_pat, torch.floor(L / MIN_SPACING))
        self.n_pat = n_pat.clamp(min=1.0).long()
        self.P = int(self.n_pat.max().item())
        self.Q = PIECES_PER_PATTERN
        auto = int(min(self.P * self.Q, max(6, round(self.P * PIECE_BUDGET))))
        self.C = int(max_props) if max_props else auto
        if bank:
            self._load_bank(bank)
        if self.place == "uniform":
            self._build_passage_grid()

        z = lambda *s: torch.zeros(*s, device=dev, dtype=torch.float32)
        self.p_poses = z(self.B, self.C, 3)
        self.p_n = z(self.B, self.C, k_pad, 2)
        self.p_d = torch.full((self.B, self.C, k_pad), float("inf"), device=dev, dtype=torch.float32)
        self.p_zlo = z(self.B, self.C)
        self.p_zhi = z(self.B, self.C)                                # z_hi <= z_lo: a dead slot
        #: [kg] per slot, and the velocity a shove left it with [m/s]. Zero mass is immovable, and
        #: a layout whose env never enables movable obstacles simply never has these touched.
        self.p_mass = z(self.B, self.C)
        #: which catalogue shape (`shapes`) stands in each slot, -1 for a dead one. Nothing in the
        #: simulation reads it -- the half-planes are the shape -- but a viewer has to know what to draw.
        self.p_sid = torch.full((self.B, self.C), -1, dtype=torch.long, device=dev)
        self.p_vel = torch.zeros(self.B, self.C, 2, device=self.device)
        #: What the last draw decided, per env: where each pattern sits on the lap [m of arc], which
        #: of `KINDS` it is, and whether that slot is a pattern at all. Two (B, P) tensors and a
        #: mask, written by the same `index_copy_` as the slots. Kept because the properties worth
        #: checking -- the 6 m spacing, that all six kinds appear -- are properties of the *draw*,
        #: and recovering them from the placed pieces means projecting a prop back onto the
        #: centerline, which on a track that folds back on itself answers with the wrong lap
        #: position. Three floats per pattern per env; the histograms in the research note come
        #: from here too.
        self.last_s = z(self.B, self.P)
        self.last_kind = torch.zeros(self.B, self.P, dtype=torch.long, device=dev)
        self.last_live = torch.zeros(self.B, self.P, dtype=torch.bool, device=dev)
        #: which pattern slot each *piece* came from, so a measurement can name the pattern that
        #: placed a given prop without guessing from its position
        self.slot_pattern = torch.zeros(self.B, self.C, dtype=torch.long, device=dev)
        self.k_pad = k_pad
        # device-side counters; read with `stats()`, which is the only thing that syncs
        self._stat = torch.zeros(4, dtype=torch.float64, device=dev)   # draws, pieces, dropped, missed
        #: circumradius of the widest catalogue piece, half of the contact cull's reach
        self.max_radius = max(0.5 * math.hypot(sh.across, sh.along) for sh in shapes)

    # ------------------------------------------------------------------ build time
    def _build_track_geometry(self):
        """Per-track lane halves, their forward-window minima, and the curvature, as (T, N) tensors.

        The running minimum is what makes the gap arithmetic constructive: a pattern reaches forward
        up to `REACH[kind]` metres of arc, so the width it is allowed to block is set by the
        narrowest the lane gets over that whole reach, not by the width where its first box happens
        to stand. One row per distinct reach, indexed by `kind_window`.
        """
        tr = self.tr
        dev = self.device
        cl_all = tr.cl.detach().cpu().numpy()                         # (T, N, 2)
        tan_all = tr.cl_tangent.detach().cpu().numpy()
        reaches = sorted(set(REACH.values()))
        self.kind_window = torch.tensor([reaches.index(REACH[k]) for k in KINDS],
                                        dtype=torch.long, device=dev)
        lanes_l, lanes_r, curvs = [], [], []
        lanes_lw = [[] for _ in reaches]
        lanes_rw = [[] for _ in reaches]
        for ti, track in enumerate(tr.tracks):
            cl = cl_all[ti].astype(np.float64)
            tang = tan_all[ti].astype(np.float64)
            nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
            if track.centerline is None:
                zeros = np.zeros(self.N)
                lanes_l.append(zeros); lanes_r.append(zeros); curvs.append(zeros)
                for w_ in range(len(reaches)):
                    lanes_lw[w_].append(zeros); lanes_rw[w_].append(zeros)
                continue
            wl, wr = _lane_halves(track, cl, nrm)
            ds = float(tr.length[ti].item()) / self.N
            lanes_l.append(wl); lanes_r.append(wr)
            for w_, reach in enumerate(reaches):
                n = int(math.ceil(reach / max(ds, 1e-6)))
                lanes_lw[w_].append(_forward_window(wl, n, "min"))
                lanes_rw[w_].append(_forward_window(wr, n, "min"))
            curvs.append(_curvature(cl, ds))
        f = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=dev)
        self.lane_l, self.lane_r = f(lanes_l), f(lanes_r)
        # (n_reaches, T, N): the lane's running minimum over each pattern's own forward reach
        self.lane_lw = torch.stack([f(a) for a in lanes_lw])
        self.lane_rw = torch.stack([f(a) for a in lanes_rw])
        self.curv = f(curvs)
        self.cl_nrm = torch.stack([-tr.cl_tangent[..., 1], tr.cl_tangent[..., 0]], -1)
        self.ds = (tr.length / self.N).clamp_min(1e-6)                # (T,)
        # no teacher yet: the corridor is empty, so nothing is kept clear of a raceline
        big = torch.full_like(self.lane_l, 1e3)
        self.rl_hi, self.rl_lo = -big, big
        self.has_raceline = False

    def _load_bank(self, path: str):
        """`bank` (11-03 night, user: "그 주행가능한지는 path generator 로직 사용해서 판단하는거지?"): layouts drawn by
        `place="uniform"` and kept only where the per-layout line generator (`layout_line.LayoutLines`) found a line a car
        body drives within its steering limit, 0.10 m clear of every prop and wall (docs/.../tools/layout_bank.py). A file
        or a directory of them, keyed by track name. Every track of this set must be in it; when one is not, the draw falls
        back to `place="uniform"` and says so."""
        import glob
        files = sorted(glob.glob(os.path.join(path, "*.pt"))) if os.path.isdir(path) else [path]
        per = {}
        for f in files:
            per.update(torch.load(f, map_location="cpu", weights_only=False)["per_track"])
        names = [t.name for t in self.tr.tracks]
        missing = [n for n in names if n not in per]
        if missing:
            print(f"[procedural] bank {path}: no layouts for {missing}; drawing uniform instead", flush=True)
            self.place = "uniform"
            return
        dev = self.device
        for ti, n in enumerate(names):
            if abs(per[n]["length"] - float(self.tr.length[ti])) > 0.01:
                raise ValueError(f"bank {path}: track {n} is {per[n]['length']:.3f} m there, {float(self.tr.length[ti]):.3f} m here")
        K = max(int(per[n]["sid"].shape[0]) for n in names)
        Cb = max(int(per[n]["sid"].shape[1]) for n in names)
        if Cb > self.C:
            raise ValueError(f"bank {path}: layouts of {Cb} pieces, this env has {self.C} slots")
        T = len(names)
        self.b_sid = torch.full((T, K, self.C), -1, dtype=torch.long)
        self.b_pose = torch.zeros(T, K, self.C, 3)
        self.b_n = torch.zeros(T, dtype=torch.long)
        for ti, n in enumerate(names):
            s_, p_ = per[n]["sid"], per[n]["pose"]
            self.b_sid[ti, :s_.shape[0], :s_.shape[1]] = s_
            self.b_pose[ti, :p_.shape[0], :p_.shape[1]] = p_
            self.b_n[ti] = s_.shape[0]
        self.b_sid, self.b_pose, self.b_n = self.b_sid.to(dev), self.b_pose.to(dev), self.b_n.to(dev)
        # each piece's centerline index and offset, for keeping a raceline corridor clear when one is installed
        tq = torch.arange(T, device=dev)[:, None, None].expand(T, K, self.C).reshape(-1)
        xy = self.b_pose[..., :2].reshape(-1, 2)
        parts = [self.tr.project(xy[i:i + 4096], tq[i:i + 4096]) for i in range(0, xy.shape[0], 4096)]
        self.b_j = (torch.cat([p_[2] for p_ in parts]).long() % self.N).reshape(T, K, self.C)
        self.b_v = torch.cat([p_[1] for p_ in parts]).reshape(T, K, self.C).float()
        self.place = "bank"
        print(f"[procedural] bank {path}: {int(self.b_n.min())}-{int(self.b_n.max())} layouts per track, {T} tracks", flush=True)

    def _redraw_bank(self, t: torch.Tensor, dst: torch.Tensor, src: torch.Tensor):
        """`bank`: one stored layout of the env's own track, 1 .. all of its pieces (count "uniform") or all ("fixed")."""
        D, C, dev = int(t.numel()), self.C, self.device
        k_idx = (self._rand((D,)) * self.b_n[t]).long().clamp_max(self.b_n[t] - 1)
        sid = self.b_sid[t, k_idx]                                               # (D, C)
        pose = self.b_pose[t, k_idx]
        live = sid >= 0
        n_live = live.sum(1, keepdim=True)
        k = n_live if self.count == "fixed" else 1 + torch.floor(self._rand((D, 1)) * n_live).long()
        if self.keep > 0:
            k = k.clamp_max(self.keep)
        r = torch.where(live, self._rand((D, C)), torch.full((D, C), 2.0, device=dev))
        live = live & (r.argsort(1).argsort(1) < k) & (self._rand((D, 1)) < self.fraction)
        sid = sid.clamp_min(0)
        if self.has_raceline:
            j, v = self.b_j[t, k_idx], self.b_v[t, k_idx]
            tq = t[:, None].expand(D, C)
            a = 0.5 * self.sh_across[sid] + CURVE_INSET
            live = live & ((v - a >= self.rl_hi[tq, j]) | (v + a <= self.rl_lo[tq, j]))
        self.last_live.index_copy_(0, dst, torch.zeros(D, self.P, dtype=torch.bool, device=dev)[src])
        self._write(dst, src, pose[..., :2], pose[..., 2], sid, live, P=C, Q=1)

    def _build_passage_grid(self):
        """`place="uniform"`'s lane grid: (T, n_bins, n_cells) True where a cell lies inside the lane over its whole bin."""
        dev = self.device
        T = self.lane_l.shape[0]
        self.u_g = max(1, int(round(PASS_BIN / float(self.ds.min()))))          # centerline points per bin
        self.u_nb = -(-self.N // self.u_g)
        idx = torch.arange(self.u_nb * self.u_g, device=dev) % self.N
        ll = self.lane_l[:, idx].reshape(T, self.u_nb, self.u_g).amin(-1)
        lr = self.lane_r[:, idx].reshape(T, self.u_nb, self.u_g).amin(-1)
        nu = int(round(2.0 * LANE_LIMIT / PASS_CELL))
        self.u_cell_lo = -LANE_LIMIT + torch.arange(nu, device=dev, dtype=torch.float32) * PASS_CELL
        self.u_cell_hi = self.u_cell_lo + PASS_CELL
        self.u_lane = (self.u_cell_lo >= -lr[..., None]) & (self.u_cell_hi <= ll[..., None])
        self.u_run = int(math.ceil(PASS_MIN / PASS_CELL - 1e-6))
        bin_len = float(self.ds.min()) * self.u_g
        self.u_hmax = int(math.ceil((0.5 * float(self.sh_along.max()) + PASS_SMEAR) / bin_len))
        # Where a piece may stand: every free map cell connected to the centerline and within LANE_LIMIT of it, drawn
        # uniformly as cells (= uniformly by area). Drawing (arc, offset) and stepping out along the normal instead
        # over-weighted the cells where the normals of a bent centerline cross: Korea 2025 showed 2x stripes along them.
        from scipy import ndimage
        from scipy.spatial import cKDTree
        cl_all = self.tr.cl.detach().cpu().numpy().astype(np.float64)
        nrm_all = self.cl_nrm.detach().cpu().numpy().astype(np.float64)
        per = []
        for ti, track in enumerate(self.tr.tracks):
            occ = np.asarray(track.occupancy, bool)
            res, ox, oy = float(track.resolution), float(track.origin[0]), float(track.origin[1])
            free = ~occ
            lab, _ = ndimage.label(free)
            cl = cl_all[ti]
            rc = np.floor((cl[:, 1] - oy) / res).astype(int), np.floor((cl[:, 0] - ox) / res).astype(int)
            inb = (rc[0] >= 0) & (rc[0] < occ.shape[0]) & (rc[1] >= 0) & (rc[1] < occ.shape[1])
            ids = np.setdiff1d(np.unique(lab[rc[0][inb], rc[1][inb]]), [0])
            rr, cc = np.nonzero(np.isin(lab, ids))
            xy = np.stack([ox + (cc + 0.5) * res, oy + (rr + 0.5) * res], 1)
            dist, j = cKDTree(cl).query(xy)
            keep = dist <= LANE_LIMIT
            xy, j = xy[keep], j[keep]
            v = ((xy - cl[j]) * nrm_all[ti, j]).sum(1)
            room = (ndimage.distance_transform_edt(free) * res)[rr[keep], cc[keep]]   # [m] to the nearest wall cell
            per.append((xy, j, v, room, res))
        kmax = max(len(p[0]) for p in per)
        T = len(per)
        f32 = lambda: torch.zeros(T, kmax, dtype=torch.float32)
        self.u_xy, self.u_v, self.u_room = torch.zeros(T, kmax, 2), f32(), f32()
        self.u_j = torch.zeros(T, kmax, dtype=torch.long)
        self.u_ncell = torch.zeros(T, dtype=torch.long)
        self.u_res = torch.zeros(T)
        for ti, (xy, j, v, room, res) in enumerate(per):
            k = len(xy)
            self.u_xy[ti, :k] = torch.as_tensor(xy, dtype=torch.float32)
            self.u_j[ti, :k] = torch.as_tensor(j)
            self.u_v[ti, :k] = torch.as_tensor(v, dtype=torch.float32)
            self.u_room[ti, :k] = torch.as_tensor(room, dtype=torch.float32)
            self.u_ncell[ti], self.u_res[ti] = k, res
        for name in ("u_xy", "u_v", "u_room", "u_j", "u_ncell", "u_res"):
            setattr(self, name, getattr(self, name).to(dev))
        #: [m] half the larger side of each shape: a piece stands only where the nearest wall is at least this far
        self.sh_rad = 0.5 * torch.maximum(self.sh_across, self.sh_along)

    def set_line(self, offset: torch.Tensor):
        """(T, N) the racing line's lateral offset [m, left +] at every centerline point, for `line_share`"""
        self.line_off = offset.to(self.device, torch.float32)

    def set_raceline(self, teacher):
        """Install the raceline corridor the pieces must stay out of, from the opponents' teacher.

        For every centerline point, the nearest raceline point gives the line's lateral offset
        there; the corridor is that offset widened by the car's half-width and `raceline_margin`,
        taken over `PIECE_WINDOW` of arc centred on the point -- the arc one piece covers, so that a
        crate whose corners reach a little either side of its own index still clears the line.

        **Not** over the pattern's forward reach, the way the lane width is. The first cut of this
        did exactly that, and it is wrong in a way that is invisible until you draw a picture: over
        4 m a racing line crosses from one side of the lane to the other, so the union of where it
        has been is most of the lane and `span` collapsed below `MIN_SPAN` for nearly every pattern.
        On `control_1400` at density 1 that left one or two pieces on a whole lap. The pattern-level
        bound now uses the corridor at the pattern's own index, and every piece is separately pushed
        clear of the corridor at *its* index (`_clear_raceline`), which is both exact and far less
        conservative.
        """
        from scipy.spatial import cKDTree
        tr = self.tr
        xy = teacher.xy.detach().cpu().numpy()                        # (T, N_rl, 2)
        cl_all = tr.cl.detach().cpu().numpy()
        nrm_all = self.cl_nrm.detach().cpu().numpy()
        clr = self.car_half_width + self.raceline_margin
        his, los = [], []
        for ti in range(tr.T):
            cl, nrm = cl_all[ti].astype(np.float64), nrm_all[ti].astype(np.float64)
            tree = cKDTree(xy[min(ti, xy.shape[0] - 1)].astype(np.float64))
            _, j = tree.query(cl, k=1)
            lat = np.einsum("ij,ij->i", xy[min(ti, xy.shape[0] - 1)][j] - cl, nrm)
            ds = float(tr.length[ti].item()) / self.N
            w = max(1, int(round(PIECE_WINDOW / max(ds, 1e-6))))
            his.append(_centred_window(lat + clr, w, "max"))
            los.append(_centred_window(lat - clr, w, "min"))
        f = lambda a: torch.tensor(np.stack(a), dtype=torch.float32, device=self.device)
        self.rl_hi, self.rl_lo = f(his), f(los)
        self.has_raceline = True

    # ------------------------------------------------------------------ runtime helpers
    def _rand(self, shape) -> torch.Tensor:
        return torch.rand(shape, device=self.device, generator=self.gen)

    def _randint(self, lo: int, hi: int, shape) -> torch.Tensor:
        return torch.randint(lo, hi, shape, device=self.device, generator=self.gen)

    def _pick_row(self, rem: torch.Tensor):
        """Largest row shape that fits in `rem`, or one of the three below it. (id, across, ok)."""
        hi = torch.searchsorted(self.row_across, rem.contiguous(), right=True) - 1
        ok = hi >= 0
        idx = (hi - self._randint(0, 3, rem.shape)).clamp(min=0)
        idx = torch.where(ok, idx, torch.zeros_like(idx))
        across = self.row_across[idx]
        return self.row_id[idx], across, ok & (across <= rem)

    def _pick_row_leq(self, target: torch.Tensor):
        """Largest row shape no wider than `target`, with no random step down (the diagonal's rung)."""
        hi = torch.searchsorted(self.row_across, target.contiguous(), right=True) - 1
        ok = hi >= 0
        idx = torch.where(ok, hi, torch.zeros_like(hi))
        across = self.row_across[idx]
        return self.row_id[idx], across, ok & (across <= target)

    def _pick_small(self, shape):
        i = self._randint(0, int(self.small_id.numel()), shape)
        return self.small_id[i], self.small_across[i]

    def _row(self, span: torch.Tensor, live0: torch.Tensor, n_slots: int):
        """A row of boxes with cracks, built from the wall inward.

        Returns `(off, live, shape)` with `off` the distance from the wall edge to each piece's
        centre. Built outward-in rather than inward-out on purpose: `n_slots` truncates the row at
        its inner end, and removing a piece there can only widen the gap. Truncating at the wall end
        would leave an unproved second opening exactly where the connectivity proof used to look.
        """
        offs, lives, ids = [], [], []
        u = torch.zeros_like(span)
        for _ in range(n_slots):
            rem = span - u
            sid, across, ok = self._pick_row(rem)
            live = live0 & ok & (across >= MIN_PIECE)
            offs.append(u + 0.5 * across)
            lives.append(live)
            ids.append(sid)
            crack = ROW_GAP[0] + (ROW_GAP[1] - ROW_GAP[0]) * self._rand(span.shape)
            u = torch.where(live, u + across + crack, u)
        pad = [torch.zeros_like(span)] * (self.Q - n_slots)
        padb = [torch.zeros_like(live0)] * (self.Q - n_slots)
        padi = [torch.zeros_like(ids[0])] * (self.Q - n_slots)
        return (torch.stack(offs + pad, -1), torch.stack(lives + padb, -1), torch.stack(ids + padi, -1))

    # ------------------------------------------------------------------ the draw
    @torch.no_grad()
    def redraw(self, draw_tid: torch.Tensor, dst: torch.Tensor, src: torch.Tensor):
        """Draw `len(draw_tid)` layouts and write them into env rows `dst` (layout `src[i]` each).

        `dst`/`src` are what make a race share one layout: the cars of a race drive the same track
        and see each other, so they must see the same crates. A car that respawns alone behind its
        race mates is not in `dst` at all and keeps the layout its race is running.
        """
        D = int(draw_tid.numel())
        if D == 0:
            return
        if self.place == "uniform":
            return self._redraw_uniform(draw_tid, dst, src)
        if self.place == "bank":
            return self._redraw_bank(draw_tid, dst, src)
        t = draw_tid
        P, Q = self.P, self.Q
        dev = self.device
        pat = torch.arange(P, device=dev)[None, :].expand(D, P)       # (D,P) pattern slot index
        n_pat = self.n_pat[t][:, None]
        use = (self._rand((D, 1)) < self.fraction)
        act = (pat < n_pat) & use
        if self.count == "uniform":
            # keep k of the lap's patterns, k uniform in 1 .. n_pat, a random subset (not the first k sectors,
            # which would put every prop of a sparse layout in one stretch of the lap)
            k = 1 + torch.floor(self._rand((D, 1)) * n_pat.float()).long()
            pr = torch.where(pat < n_pat, self._rand((D, P)), torch.full((D, P), 2.0, device=dev))
            rank = pr.argsort(1).argsort(1)
            act = act & (rank < k)

        # --- where each pattern sits on the lap ------------------------------------------------
        L = self.tr.length[t][:, None]
        sector = L / n_pat.clamp_min(1).float()
        free = (sector - MIN_SPACING).clamp_min(0.0)
        # The whole grid of sectors is turned by a random phase every draw. Without it pattern k
        # always sat in sector k, and a dense lap leaves a sector almost no `free` arc: on ICCAS
        # (43.4 m, density 1.5) the seven patterns stood at the same seven points +-0.1 m at every
        # reset of s911-s913, so what was learned was seven places, not obstacles. A rigid turn
        # keeps every gap, the wrap included, so the 6 m spacing is untouched.
        phase = self._rand((D, 1)) * L
        base = pat.float() * sector + 0.5 * (sector - free) + phase
        # CANDIDATES stratified places inside the pattern's own slice of the lap. Every candidate
        # keeps the 6 m spacing, so the choice between them is free. The whole placement is worked
        # out for all of them and the pattern then takes one that *fits*: `hard_obstacles` gets the
        # same effect by drawing an index at random and retrying up to sixty times per pattern, and
        # without something equivalent a pattern whose slice of the lap happens to be narrow --
        # `control_1400`'s lane goes down to 1.42 m, which leaves 0.22 m beside a 1.20 m gap --
        # simply is not placed. Measured on that map at density 1, choosing among the candidates
        # takes the patterns actually drawn on a lap from 2.8 of 5 to 3.5 of 5.
        cand_u = (torch.arange(CANDIDATES, device=dev).float()[None, None] + self._rand((D, P, CANDIDATES))) / CANDIDATES
        s_cand = base[..., None] + cand_u * free[..., None]
        j_cand = (s_cand / self.ds[t][:, None, None]).long() % self.N
        kind = self._draw_kinds(D, P)
        is_apex = (kind == KINDS.index("apex"))[..., None]
        tid_c = t[:, None, None].expand_as(j_cand)
        win_c = self.kind_window[kind][..., None].expand_as(j_cand)   # each kind's own forward reach
        wl_c = self.lane_lw[win_c, tid_c, j_cand]
        wr_c = self.lane_rw[win_c, tid_c, j_cand]
        width_c = wl_c + wr_c
        hi_c = self.rl_hi[tid_c, j_cand]
        lo_c = self.rl_lo[tid_c, j_cand]
        curv_c = self.curv[tid_c, j_cand]

        # --- how much of the lane this pattern may block, at every candidate --------------------
        frac = (GAP_FRAC[0] + (GAP_FRAC[1] - GAP_FRAC[0]) * self._rand((D, P)))[..., None]
        g_c = torch.maximum(torch.full_like(width_c, GAP_MIN), width_c * frac)
        want_c = width_c - g_c
        # The corridor here is the one at the candidate's own index -- enough to choose which wall
        # to block and how much of the lane to ask for. Every piece is separately pushed clear of
        # the corridor at *its* index afterwards (`_clear_raceline`).
        span_l_c = wl_c - hi_c                                       # blocking from the +normal wall
        span_r_c = wr_c + lo_c                                       # ... and from the -normal wall
        prefer_l_c = torch.where(is_apex, curv_c > 0, (self._rand((D, P)) < 0.5)[..., None])
        pref_c = torch.where(prefer_l_c, span_l_c, span_r_c)
        alt_c = torch.where(prefer_l_c, span_r_c, span_l_c)
        left_c = prefer_l_c ^ ((pref_c < MIN_SPAN) & (alt_c >= MIN_SPAN))
        span_c = (torch.minimum(want_c, torch.where(left_c, span_l_c, span_r_c)) - CURVE_INSET).clamp_min(0.0)

        # one candidate: any that fits, at random -- or the sharpest corner among those, for apex
        kappa = torch.abs(curv_c)
        kappa = kappa / (kappa.amax(-1, keepdim=True) + 1e-9) * 0.999
        key = (span_c >= MIN_SPAN).to(width_c.dtype) + torch.where(is_apex, kappa, self._rand((D, P, CANDIDATES)))
        pick = key.argmax(-1)
        take = lambda a: torch.gather(a, -1, pick[..., None])[..., 0]
        j = take(j_cand)                                             # (D,P) centerline index
        wl, wr, hi, lo = take(wl_c), take(wr_c), take(hi_c), take(lo_c)
        width, curv, span = wl + wr, take(curv_c), take(span_c)
        left = take(left_c)
        side = torch.where(left, torch.ones_like(width), -torch.ones_like(width))
        row_ok = act & (span >= MIN_SPAN)

        # the chicane's second row, on the other wall, with its own drawn gap
        g2 = torch.maximum(torch.full_like(width, GAP_MIN),
                           width * (GAP_FRAC[0] + (GAP_FRAC[1] - GAP_FRAC[0]) * self._rand((D, P))))
        span2 = (torch.minimum(width - g2, torch.where(left, wr + lo, wl - hi)) - CURVE_INSET).clamp_min(0.0)
        chic_along = 2.0 + 1.5 * self._rand((D, P))

        off, live, sid, along, pside = self._pieces(
            kind, act, row_ok, span, span2, side, chic_along, wl, wr, hi, lo, curv)
        is_line = None
        if self.line_share > 0 and self.line_off is not None and not self.has_raceline:
            # one catalogue box on the line; where that leaves less than GAP_MIN on both sides it moves the least it
            # must toward a wall so one side keeps GAP_MIN (11-03: dropping it emptied the narrow Korea 2026 stretches)
            is_line = act & (self._rand((D, P)) < self.line_share)
            n_pool = self.line_pool.numel()
            sid_l = self.line_pool[(self._rand((D, P)) * n_pool).long().clamp_max(n_pool - 1)]
            a_l = 0.5 * self.sh_across[sid_l]
            tj = t[:, None].expand(D, P)
            v_l = self.line_off[tj, j] + (self._rand((D, P)) * 2.0 - 1.0) * LINE_JITTER
            wl_j, wr_j = self.lane_l[tj, j], self.lane_r[tj, j]
            lo_a, hi_a = -wr_j + a_l, wl_j - a_l - GAP_MIN                # the way past on the + side
            lo_b, hi_b = -wr_j + a_l + GAP_MIN, wl_j - a_l                # ... on the - side
            v_a = torch.minimum(torch.maximum(v_l, lo_a), hi_a)
            v_b = torch.minimum(torch.maximum(v_l, lo_b), hi_b)
            ok_a, ok_b = lo_a <= hi_a, lo_b <= hi_b
            inf = torch.full_like(v_l, float("inf"))
            d_a = torch.where(ok_a, (v_a - v_l).abs(), inf)
            d_b = torch.where(ok_b, (v_b - v_l).abs(), inf)
            v_l = torch.where(d_a <= d_b, v_a, v_b)
            ok_l = is_line & (ok_a | ok_b)
            q0 = (torch.arange(self.Q, device=dev) == 0)[None, None, :]
            live = torch.where(is_line[..., None], ok_l[..., None] & q0, live)
            sid = torch.where(is_line[..., None], sid_l[..., None].expand_as(sid), sid)
            along = torch.where(is_line[..., None], torch.zeros_like(along), along)

        # --- from lane coordinates to world ----------------------------------------------------
        jq = (j[..., None] + torch.round(along / self.ds[t][:, None, None]).long()) % self.N
        tid_q = t[:, None, None].expand_as(jq)
        c = self.tr.cl[tid_q, jq]                                    # (D,P,Q,2)
        tan = self.tr.cl_tangent[tid_q, jq]
        nrm = torch.stack([-tan[..., 1], tan[..., 0]], -1)
        edge = torch.where(pside > 0, self.lane_l[tid_q, jq], -self.lane_r[tid_q, jq])
        v = edge - pside * off
        v, live = self._clear_raceline(v, live, sid, pside, edge, tid_q, jq)
        if is_line is not None:
            v = torch.where(is_line[..., None], v_l[..., None].expand_as(v), v)
        xy = c + nrm * v[..., None]
        yaw = (torch.atan2(tan[..., 1], tan[..., 0]) + 0.5 * math.pi
               + (self._rand(along.shape) * 2.0 - 1.0) * YAW_JITTER)

        self.last_s.index_copy_(0, dst, torch.remainder(take(s_cand), L)[src])
        self.last_kind.index_copy_(0, dst, kind[src])
        self.last_live.index_copy_(0, dst, (act & live.any(-1))[src])
        self._write(dst, src, xy.reshape(D, P * Q, 2), yaw.reshape(D, P * Q),
                    sid.reshape(D, P * Q), live.reshape(D, P * Q))

    def _redraw_uniform(self, t: torch.Tensor, dst: torch.Tensor, src: torch.Tensor):
        """`place="uniform"`: C pieces per layout (1 .. C at count "uniform"), each a random catalogue shape at a point
        drawn uniformly over the lane's area, kept only while a PASS_MIN way stays open around it (`PASS_MIN`)."""
        D, C, M, dev = int(t.numel()), self.C, U_CANDIDATES, self.device
        use = self._rand((D, 1)) < self.fraction
        k = (torch.full((D, 1), C, device=dev) if self.count == "fixed"
             else 1 + torch.floor(self._rand((D, 1)) * C).long())
        live = use & (torch.arange(C, device=dev)[None, :] < k)                  # (D,C)
        n_pool = self.line_pool.numel()
        sid = self.line_pool[(self._rand((D, C)) * n_pool).long().clamp_max(n_pool - 1)]
        a = 0.5 * self.sh_across[sid]                                            # (D,C) half across, yaw jitter included
        # M candidates per piece, each a uniformly drawn free cell (+ a uniform point inside it); a candidate counts only
        # where the piece fits clear of the walls, and the first that does is taken (Gumbel-max over equal weights)
        k_t = self.u_ncell[t][:, None, None]
        ci = (self._rand((D, C, M)) * k_t).long().clamp_max(k_t - 1)
        tq = t[:, None, None].expand_as(ci)
        res = self.u_res[t][:, None, None, None]
        xy_c = self.u_xy[tq, ci] + (self._rand((D, C, M, 2)) - 0.5) * res
        j_c = self.u_j[tq, ci]
        v_c = self.u_v[tq, ci]
        w = (self.u_room[tq, ci] >= self.sh_rad[sid][..., None]).float()
        if self.has_raceline:
            # the planner opponents are blind to props and drive that line: a piece stays out of its corridor
            ac = a[..., None] + CURVE_INSET
            w = w * ((v_c - ac >= self.rl_hi[tq, j_c]) | (v_c + ac <= self.rl_lo[tq, j_c])).float()
        u = self._rand((D, C, M)).clamp(1e-12, 1.0 - 1e-7)
        key = torch.log(w.clamp_min(1e-30)) - torch.log(-torch.log(u))         # Gumbel-max = draw by weight
        pick = key.argmax(-1, keepdim=True)
        j = torch.gather(j_c, -1, pick)[..., 0]
        v = torch.gather(v_c, -1, pick)[..., 0]
        xy = torch.gather(xy_c, 2, pick[..., None].expand(D, C, 1, 2))[:, :, 0]
        live = live & (torch.gather(w, -1, pick)[..., 0] > 0)

        # the passage: put the pieces down one slot at a time on the lane grid, each only if every bin it reaches
        # (its own length plus PASS_SMEAR either side) keeps a free run of u_run cells
        grid = self.u_lane[t].clone()                                            # (D, nb, nu)
        H = self.u_hmax
        offs = torch.arange(-H, H + 1, device=dev)
        half = torch.ceil((0.5 * self.sh_along[sid] + PASS_SMEAR) / (self.ds[t][:, None] * self.u_g)).long()
        ar = torch.arange(D, device=dev)[:, None]
        m = self.u_run
        for c in range(C):
            bins = (j[:, c:c + 1] // self.u_g + offs[None]) % self.u_nb           # (D, W)
            inwin = offs[None].abs() <= half[:, c:c + 1]
            rows = grid[ar, bins]                                                # (D, W, nu)
            lat = (self.u_cell_hi > (v[:, c] - a[:, c])[:, None]) & (self.u_cell_lo < (v[:, c] + a[:, c])[:, None])
            new = rows & ~(inwin[..., None] & lat[:, None, :])
            cs = torch.nn.functional.pad(new.int().cumsum(-1), (1, 0))
            run_ok = ((cs[..., m:] - cs[..., :-m]) == m).any(-1)
            ok = live[:, c] & (run_ok | ~inwin).all(-1)
            live[:, c] = ok
            grid[ar, bins] = torch.where(ok[:, None, None], new, rows)

        tan = self.tr.cl_tangent[t[:, None].expand(D, C), j]
        yaw =torch.atan2(tan[..., 1], tan[..., 0]) + 0.5 * math.pi + (self._rand((D, C)) * 2.0 - 1.0) * YAW_JITTER
        self.last_live.index_copy_(0, dst, torch.zeros(D, self.P, dtype=torch.bool, device=dev)[src])
        self._write(dst, src, xy, yaw, sid, live, P=C, Q=1)

    def _clear_raceline(self, v, live, sid, pside, edge, tid_q, jq):
        """Push each piece out of the raceline corridor **at its own arc index**, or drop it.

        The pattern-level `span` already bounds the band against the corridor at the *pattern's*
        index; a piece placed further along the lane meets a different piece of line. Pushing it
        outward -- toward the wall it already hugs -- is the move that cannot break anything else:
        the piece stays inside `[edge - span, edge]`, so the gap on the other side only grows, and
        it stays inside the lane or it is dropped. A piece that has nowhere to stand between the
        line and the wall is not placed, which is the honest outcome for a section where the racing
        line runs against the boundary.
        """
        if not self.has_raceline:
            return v, live
        half = 0.5 * self.sh_across[sid]
        a = half + CURVE_INSET
        out = torch.where(pside > 0,
                          torch.maximum(v, self.rl_hi[tid_q, jq] + a),
                          torch.minimum(v, self.rl_lo[tid_q, jq] - a))
        # `edge` is the wall on the piece's own side, signed: +lane_l or -lane_r. The piece keeps
        # the same relationship to it that the row gave it -- its outer face on the wall, no further
        # -- so it fits exactly while its outer half is still inside the lane.
        fits = torch.where(pside > 0, out + half <= edge, out - half >= edge)
        return out, live & fits

    def _draw_kinds(self, D: int, P: int) -> torch.Tensor:
        """One pattern kind per slot, as a random permutation repeated: every seven patterns a lap
        carries all seven kinds, which is what `hard_obstacles` gets from shuffling a cycled list."""
        perm = torch.argsort(self._rand((D, len(KINDS))), dim=1)      # (D,7)
        idx = torch.arange(P, device=self.device)[None, :] % len(KINDS)
        return torch.gather(perm, 1, idx.expand(D, P))

    def _pieces(self, kind, act, row_ok, span, span2, side, chic_along, wl, wr, hi, lo, curv):
        """Every pattern's pieces as `(off, live, shape, along, side)`, each `(D, P, Q)`.

        `off` is the distance from the wall on `side` inward to the piece's centre, which is the one
        parameterisation in which the gap guarantee is a single inequality: every live piece has
        `off + across/2 <= span`, so nothing reaches past the band.
        """
        D, P = span.shape
        Q = self.Q
        dev = self.device
        zq = torch.zeros(D, P, Q, device=dev)
        bq = torch.zeros(D, P, Q, dtype=torch.bool, device=dev)
        iq = torch.zeros(D, P, Q, dtype=torch.long, device=dev)
        off, live, sid = zq.clone(), bq.clone(), iq.clone()
        along = zq.clone()
        side0 = side[..., None].expand(D, P, Q)          # the wall the pattern blocks, per piece
        pside = side0.clone()
        q = torch.arange(Q, device=dev)[None, None, :].expand(D, P, Q)

        def put(mask, off_, live_, sid_, along_=None, side_=None):
            m = mask[..., None] if mask.dim() == 2 else mask
            off.copy_(torch.where(m, off_, off))
            live.copy_(torch.where(m, live_, live))
            sid.copy_(torch.where(m, sid_, sid))
            if along_ is not None:
                along.copy_(torch.where(m, along_, along))
            if side_ is not None:
                pside.copy_(torch.where(m, side_, pside))

        # ---- gate: one row straight across ----------------------------------------------------
        g_off, g_live, g_sid = self._row(span, row_ok, Q)
        put(kind == KINDS.index("gate"), g_off, g_live, g_sid)

        # ---- diagonal: rungs stepping across, the gap at the far end --------------------------
        k = self._randint(3, 6, (D, P)).float()[..., None]
        step = (span[..., None] / k)
        d_sid, d_across, d_ok = self._pick_row_leq(step.expand(D, P, Q))
        d_live = row_ok[..., None] & (q < k.long()) & d_ok & (d_across >= MIN_PIECE)
        d_off = (q.float() + 0.5) * step
        put(kind == KINDS.index("diagonal"), d_off, d_live, d_sid, q.float() * 0.5)

        # ---- chicane: a row from one wall, then 2-3.5 m later a row from the other -------------
        h = Q // 2
        c1_off, c1_live, c1_sid = self._row(span, row_ok, h)
        c2_off, c2_live, c2_sid = self._row(span2, act & (span2 >= MIN_SPAN), h)
        # `_row` pads to Q; take the first `h` of each and interleave the halves
        c_off = torch.cat([c1_off[..., :h], c2_off[..., :h]], -1)
        c_live = torch.cat([c1_live[..., :h], c2_live[..., :h]], -1)
        c_sid = torch.cat([c1_sid[..., :h], c2_sid[..., :h]], -1)
        c_along = torch.where(q < h, torch.zeros_like(zq), chic_along[..., None].expand(D, P, Q))
        c_side = torch.where(q < h, side0, -side0)
        put(kind == KINDS.index("chicane"), c_off, c_live, c_sid, c_along, c_side)

        # ---- apex: a block on the inside of the corner, small a third of the time ---------------
        a_off, a_live, a_sid = self._row(span, row_ok, 4)
        small_id, small_ac = self._pick_small((D, P, Q))
        is_small = (self._rand((D, P)) < SMALL_SHARE)[..., None].expand(D, P, Q)
        a_off = torch.where(is_small, 0.5 * small_ac + 0.05, a_off)
        a_sid = torch.where(is_small, small_id, a_sid)
        a_live = torch.where(is_small, row_ok[..., None] & (q == 0) & (small_ac + 0.05 <= span[..., None]),
                             a_live)
        put(kind == KINDS.index("apex"), a_off, a_live, a_sid)

        # ---- cluster: two or three boxes touching, wedged into one side -------------------------
        kc = self._randint(2, 4, (D, P))[..., None]
        cl_small = self._rand((D, P, Q)) < SMALL_SHARE
        big_id, big_ac, _ = self._pick_row_leq(torch.full_like(zq, BOX_W))
        cl_across = torch.where(cl_small, small_ac, big_ac)
        cl_sid = torch.where(cl_small, small_id, big_id)
        cl_off = (q % 2).float() * (big_ac + 0.02) + 0.5 * cl_across + 0.03
        cl_along = (q // 2).float() * (BOX_D + 0.03)
        cl_live = row_ok[..., None] & (q < kc) & (cl_off + 0.5 * cl_across <= span[..., None])
        put(kind == KINDS.index("cluster"), cl_off, cl_live, cl_sid, cl_along)

        # ---- scatter: one to three small things anywhere across the lane ------------------------
        ks = self._randint(1, 4, (D, P))[..., None]
        # One side per pattern, not per object. Each object on its own leaves GAP_MIN of lane on the
        # far side of it, so objects that share a side can only ever be bounded by the innermost of
        # them and the guarantee survives; two objects on *opposite* sides do not compose, and the
        # lane frame is exactly where that bites -- on `control_1400`'s tightest hairpin the
        # centerline turns far enough inside 1.4 m of arc that two objects a metre apart along the
        # lap stand side by side, leaving 1.12 m between them. That was the last 0.2 % of layouts
        # under 1.20 m; with the side drawn per pattern there is none. Objects still land anywhere
        # from the raceline corridor's edge out to the wall, mid-lane included.
        s_side = torch.where(self._rand((D, P)) < 0.5, torch.ones_like(span), -torch.ones_like(span))
        s_side = s_side[..., None].expand(D, P, Q)
        # the object must leave GAP_MIN on the far side of it and stay clear of the raceline band,
        # with the same curvature inset every other piece gets
        a = 0.5 * small_ac + CURVE_INSET
        up_l = wl[..., None] - a
        lo_l = torch.maximum(-wr[..., None] + GAP_MIN + a, hi[..., None] + a)
        up_r = torch.minimum(wl[..., None] - GAP_MIN - a, lo[..., None] - a)
        lo_r = -wr[..., None] + a
        upper = torch.where(s_side > 0, up_l, up_r)
        lower = torch.where(s_side > 0, lo_l, lo_r)
        v = lower + self._rand((D, P, Q)) * (upper - lower).clamp_min(0.0)
        # express it as a distance in from the wall on its own side, the way every other piece is
        s_off = torch.where(s_side > 0, wl[..., None] - v, v + wr[..., None])
        s_live = act[..., None] & (q < ks) & (upper >= lower)
        # Stratified along the lane with a *gap* between the strata, not simply `q + U(0,1)`: two
        # scatter objects that land at nearly the same arc index can sit either side of the lane and
        # leave only the strip between them. Measured, that was the whole of the 0.5 % of layouts
        # whose realised gap came in under 1.20 m (worst 0.80 m, two 0.2 m objects at +-0.5 m of a
        # 2.5 m lane). `hard_obstacles` draws the offsets the same way and is saved by the erosion
        # proof that rejects the draw; there is no proof here, so the strata are separated by more
        # than a piece is deep and no two objects ever share an arc index.
        put(kind == KINDS.index("scatter"), s_off, s_live, small_id,
            q.float() * 1.0 + self._rand((D, P, Q)) * 0.5, s_side)

        # ---- center: one wide piece in the middle of the lane, a way past on each side -----------
        # Measured from the wall on `side`: a narrow way of CENTER_SIDE_MIN or more, the piece, and
        # beyond it GAP_MIN or more -- the same guarantee every other pattern gives, on the far side.
        # `wl + wr` is the narrowest the lane gets over the pattern's reach, and the piece stands at
        # its own index, so the realised ways are at least these. The narrow one is drawn anywhere
        # its bounds allow, so the piece stands anywhere from off-centre to dead centre.
        W = (wl + wr)[..., None].expand(D, P, Q)
        room = W - GAP_MIN - CENTER_SIDE_MIN - 2.0 * CURVE_INSET
        ce_sid, ce_across, ce_ok = self._pick_row(room.clamp(max=CENTER_MAX))
        n_way = CENTER_SIDE_MIN + CURVE_INSET + self._rand((D, P, Q)) * (room - ce_across).clamp_min(0.0)
        ce_live = act[..., None] & (q == 0) & ce_ok & (ce_across >= MIN_PIECE) & (ce_across <= room)
        put(kind == KINDS.index("center"), n_way + 0.5 * ce_across, ce_live, ce_sid)
        return off, live, sid, along, pside

    # ------------------------------------------------------------------ writing the slots
    def _write(self, dst, src, xy, yaw, sid, live, P=None, Q=None):
        """Compact each layout's live pieces into the first `C` slots and copy them into `dst`.

        When there are more live pieces than slots, the patterns are kept in a random order, whole,
        and the one the budget runs out in is kept in part. They used to be kept in lap order: with
        10 slots for seven patterns on ICCAS, the last three were dropped at nearly every reset
        (51 of 640 kept pieces), so the last 40 % of the lap almost never had an obstacle."""
        D = xy.shape[0]
        P = self.P if P is None else P
        Q = self.Q if Q is None else Q
        rank = torch.argsort(self._rand((D, P)), dim=1)                # a random order of the patterns
        prio = torch.empty_like(rank).scatter_(1, rank, torch.arange(P, device=rank.device).expand(D, P))
        key = (prio[:, :, None] * Q + torch.arange(Q, device=rank.device)[None, None]).reshape(D, P * Q)
        key = torch.where(live, key, key + P * Q)                      # every dead piece after every live one
        order = torch.argsort(key, dim=1)[:, :self.C]
        g = lambda a: torch.gather(a, 1, order if a.dim() == 2 else order[..., None].expand(-1, -1, a.shape[-1]))
        keep = g(live)
        sid_k = g(sid)
        pose = torch.cat([g(xy), g(yaw)[..., None]], -1)              # (D,C,3)
        n = self.sh_n[sid_k]                                          # (D,C,K,2)
        d = self.sh_d[sid_k]
        z0 = self.sh_z0[sid_k]
        z1 = torch.where(keep, self.sh_z1[sid_k], self.sh_z0[sid_k] - 1.0)   # dead: z_hi < z_lo
        self.slot_pattern.index_copy_(0, dst, (order // Q)[src])
        self.p_sid.index_copy_(0, dst, torch.where(keep, sid_k, torch.full_like(sid_k, -1))[src])
        self._stat[0] += float(D)
        self._stat[1] += keep.sum().to(torch.float64)
        self._stat[2] += (live.sum(1) - keep.sum(1)).sum().to(torch.float64)
        self.p_poses.index_copy_(0, dst, pose[src])
        self.p_n.index_copy_(0, dst, n[src])
        self.p_d.index_copy_(0, dst, d[src])
        self.p_zlo.index_copy_(0, dst, z0[src])
        self.p_zhi.index_copy_(0, dst, z1[src])
        # A fresh layout is a fresh set of objects standing still, whatever the last one was
        # shoved to -- otherwise a crate would inherit the momentum of the race before it.
        self.p_mass.index_copy_(0, dst, torch.where(keep, self.sh_mass[sid_k],
                                                    torch.zeros_like(self.sh_mass[sid_k]))[src])
        self.p_vel.index_copy_(0, dst, torch.zeros_like(self.p_vel[:D])[src])

    # ------------------------------------------------------------------ a layout kept and put back
    #: the tensors that are a row's layout (every reader takes them live, row by row)
    _LAYOUT = ("p_poses", "p_n", "p_d", "p_zlo", "p_zhi", "p_mass", "p_sid", "slot_pattern")

    def snapshot(self, rows: torch.Tensor) -> dict:
        """A copy of the layouts standing in `rows`, slot for slot (`restore` puts them back, in any rows)."""
        return {k: getattr(self, k)[rows].clone() for k in self._LAYOUT}

    def restore(self, rows: torch.Tensor, snap: dict) -> None:
        """Put layouts taken by `snapshot` into `rows` (one per row, same order), every piece at rest. In place: the
        compiled step reads these tensors, not the attributes."""
        for k in self._LAYOUT:
            getattr(self, k)[rows] = snap[k].to(getattr(self, k).dtype)
        self.p_vel[rows] = 0.0

    # ------------------------------------------------------------------ being pushed
    #: How many consecutive env rows share one prop set: the cars of a race (`SimParams.props_per_race`,
    #: set by `F1VecEnv`). Every row keeps its own copy -- every reader indexes by env row and is left as it
    #: was -- and every change is the race's total, applied to each of its copies (`_race_add`), so the
    #: copies stay identical: the crate one car shoved is where the others meet it, and two cars pushing
    #: one crate add up. 1 is a set per car, every run before 2026-10-15: an opponent drove into a crate
    #: where it stood before the learner moved it, and the console drew env 0's copy alone.
    share_m = 1

    def _race_add(self, buf: torch.Tensor, eid: torch.Tensor, slot: torch.Tensor, val: torch.Tensor) -> None:
        """`buf[eid, slot] += val`, in place, summed over each race's rows and given to all of them.
        eid must be the whole batch in row order (the physics roll's `Simulator.eid`). Deterministic --
        an `index_add_` onto distinct cells, then a sum over the race -- and capturable: no host data."""
        B, C, M = buf.shape[0], buf.shape[1], int(self.share_m)
        if eid.shape[0] != B:
            raise ValueError(f"a race-shared prop update needs the whole batch in row order ({B} rows), "
                             f"got {eid.shape[0]}")
        D = torch.zeros(B * C, val.shape[-1], device=val.device, dtype=buf.dtype)
        D.index_add_(0, eid * C + slot, val.to(buf.dtype))
        D = D.view(B // M, M, C, -1).sum(1, keepdim=True).expand(B // M, M, C, val.shape[-1])
        buf.add_(D.reshape(B, C, -1))

    def shove(self, eid: torch.Tensor, slot: torch.Tensor, impulse: torch.Tensor) -> None:
        """Add a linear impulse [kg m/s] to one prop per row, along the world vector given.

        Rotation is deliberately not modelled. A crate struck off centre really does spin, and
        adding that means an inertia per shape, a contact point rather than a normal, and a yaw
        state per slot -- for an effect the policy reads through a LiDAR at 0.25 degrees. What it
        has to learn is "that one moves and that one does not", and translation carries it.
        """
        m = self.p_mass[eid, slot].clamp_min(1e-6)
        live = (self.p_mass[eid, slot] > 0) & (self.p_zhi[eid, slot] > self.p_zlo[eid, slot])
        dv = torch.where(live[:, None], impulse / m[:, None], torch.zeros_like(impulse))
        if self.share_m > 1:
            self._race_add(self.p_vel, eid, slot, dv)
            return
        self.p_vel[eid, slot] = self.p_vel[eid, slot] + dv

    def push(self, eid: torch.Tensor, slot: torch.Tensor, dv: torch.Tensor, dxy: torch.Tensor) -> None:
        """Change one prop per row by `dv` [m/s] and move it by `dxy` [m] (world), in place: the contact's
        share of the impulse and of the overlap (`Simulator._resolve_wall_contact`, `SimParams.prop_relative`).
        Rows whose `slot` is -1, massless or dead are left alone; a race's rows add up (`share_m`)."""
        sl = slot.clamp_min(0)
        live = (slot >= 0) & (self.p_mass[eid, sl] > 0) & (self.p_zhi[eid, sl] > self.p_zlo[eid, sl])
        dv = torch.where(live[:, None], dv, torch.zeros_like(dv))
        dxy = torch.where(live[:, None], dxy, torch.zeros_like(dxy))
        if self.share_m > 1:
            self._race_add(self.p_vel, eid, sl, dv)
            self._race_add(self.p_poses[:, :, :2], eid, sl, dxy)
            return
        self.p_vel[eid, sl] = self.p_vel[eid, sl] + dv
        self.p_poses[eid, sl, :2] = self.p_poses[eid, sl, :2] + dxy

    def drift(self, dt: float, tid: Optional[torch.Tensor] = None) -> None:
        """`advance` without its early-out, for inside the physics roll: a host read there would stall a
        graph capture. Same law -- the floor takes `PROP_GROUND_DECEL` off the speed, then the prop slides
        -- run every contact window, so a shoved prop is where it is when the next contact is tested.

        With `tid` (each env row's track) a prop also stops at the walls (`stop_at_walls`) before it slides."""
        v = self.p_vel
        sp = v.norm(dim=2, keepdim=True)
        keep = ((sp - PROP_GROUND_DECEL * dt) / sp.clamp_min(1e-9)).clamp_min(0.0)
        self.p_vel.mul_(keep)
        if tid is not None:
            self.stop_at_walls(tid)
        self.p_poses[:, :, :2] += self.p_vel * dt

    def stop_at_walls(self, tid: torch.Tensor) -> None:
        """A prop sliding into a wall stops at it: once the wall's surface is within the prop's inscribed
        radius of its centre, the part of its velocity into the wall is taken out -- the hose takes the blow and
        nothing bounces off it. Velocity only, so a piece the layout stood against the wall is not moved. The
        inscribed radius is the nearest of the prop's own face planes (`p_d`), which a box rotated in the
        corner of a cell can poke past by a few cm; a crate shoved along the lane used to slide into the hose
        and on through it. In place and branch-free: it runs inside the captured physics roll."""
        B, C = self.p_vel.shape[:2]
        xy = self.p_poses[:, :, :2]
        t = tid[:, None].expand(B, C)
        r_in = self.p_d.min(-1).values.clamp(max=1.0)                 # padding faces are +inf
        # The distance field is the nearest cell's, so it reads the wall's surface to within half a cell either
        # way: stop half a cell early rather than half a cell into the hose.
        res = self.tr.t_res[t]
        reach = self.tr.sample_edt(xy, t) - 0.5 * res                    # centre to the wall's surface
        n = self.tr.edt_gradient(xy, t)                                   # away from the wall
        vn = (self.p_vel * n).sum(-1)
        stop = (self.p_zhi > self.p_zlo) & (self.p_mass > 0) & (reach <= r_in + 0.5 * res) & (vn < 0)
        self.p_vel.sub_(torch.where(stop, vn, torch.zeros_like(vn))[..., None] * n)

    def advance(self, dt: float) -> None:
        """Slide whatever was shoved, and let the floor stop it.

        `p_poses` is where a prop *is*; its half-planes are prop-local and transformed at query
        time, so moving one is this and nothing else -- no re-rasterising, no distance field. That
        is the whole reason a movable obstacle is affordable here and a deformable duct is not.
        """
        v = self.p_vel
        sp = v.norm(dim=2, keepdim=True)
        if float(sp.max()) <= 0.0:
            return
        drop = (PROP_GROUND_DECEL * dt)
        keep = ((sp - drop) / sp.clamp_min(1e-9)).clamp_min(0.0)
        # In place: `shove` runs inside the captured physics roll and writes into this buffer, so
        # replacing the tensor here would leave the graph writing into one nobody reads any more.
        self.p_vel.mul_(keep)
        self.p_poses[:, :, :2] += self.p_vel * dt

    # ------------------------------------------------------------------ readers
    def slots(self, eid: Optional[torch.Tensor] = None):
        if eid is None:
            return self.p_poses, self.p_n, self.p_d, self.p_zlo, self.p_zhi
        return (self.p_poses[eid], self.p_n[eid], self.p_d[eid], self.p_zlo[eid], self.p_zhi[eid])

    def near(self, xy: torch.Tensor, eid: Optional[torch.Tensor] = None, k: int = CONTACT_SLOTS,
             reach: float = 0.0, return_idx: bool = False):
        """The `k` layout slots nearest each point of `xy` (n, 2), as the same five tensors.

        For the *contact* tests -- the car's own footprint, and the spawn rejection -- and not for
        the beams, which see every slot. A prop can only touch a car whose centre is within the sum
        of the two circumradii, about 0.71 m here, and a layout has at most a handful of pieces that
        close together; culling to the eight nearest therefore costs nothing and takes the SAT from
        fifty-odd slots to eight. `reach` is that sum: every live slot inside it that the cull
        dropped is counted, and `stats()["missed_in_reach"]` is expected to be exactly zero.
        """
        poses, n, d, zlo, zhi = self.slots(eid)
        if poses.shape[1] <= k:
            if not return_idx:
                return poses, n, d, zlo, zhi
            # No cull: column j *is* slot j, and a caller that has to name the prop it touched
            # needs that said rather than assumed.
            straight = torch.arange(poses.shape[1], device=poses.device)
            return poses, n, d, zlo, zhi, straight[None].expand(xy.shape[0], -1)
        dead = zhi <= zlo
        d2 = (poses[..., :2] - xy[:, None, :]).pow(2).sum(-1)
        d2 = torch.where(dead, torch.full_like(d2, float("inf")), d2)
        near_d2, idx = torch.topk(d2, k, dim=1, largest=False)
        if reach > 0.0 and not _is_compiling():
            # Not under `torch.compile`: this is a running total on a device tensor, and mutating one
            # inside a traced region is a graph break at best. `_resolve_wall_contact` calls in from
            # inside the compiled physics roll whenever the walls are soft.
            r2 = reach * reach
            self._stat[3] += ((d2 <= r2).sum() - (near_d2 <= r2).sum()).to(torch.float64)
        g = lambda a: torch.gather(a, 1, idx.reshape(idx.shape + (1,) * (a.dim() - 2)).expand(-1, -1, *a.shape[2:]))
        out = (g(poses), g(n), g(d), g(zlo), g(zhi))
        return out + (idx,) if return_idx else out

    def clearance(self, xy: torch.Tensor, eid: Optional[torch.Tensor] = None,
                  k: int = CONTACT_SLOTS) -> torch.Tensor:
        """(n,) metres from each point of `xy` (n, 2) to the nearest live prop, `inf` if none.

        Why this exists: everything that reads the occupancy grid is blind to these props -- they
        were never rasterised into it -- and the drivers of the other cars are among them. That is
        the whole reason a layout has to be laid outside the racing line (`set_raceline`): a car
        following a line through a crate it cannot see is not a lesson about anything. A driver
        that can ask this question does not need the line kept clear, which is what lets a layout
        stand *on* the racing line and the policy meet one there.

        The measure is `max_j (n_j . x - d_j)` over a prism's own faces: negative inside, and
        outside it is the distance to the nearest face *plane*. For a convex prism that is the true
        distance whenever the nearest feature is an edge and an under-estimate near a corner, which
        is the safe direction for a clearance -- it never reports more room than there is.

        Culled to the `k` nearest slots like `near`, and for the same reason. A caller asking about
        room for a manoeuvre cares about what is close; a prop further away than the eighth nearest
        is not what stops the car.
        """
        from .prop_math import _slot_is_live, _to_world
        n_pts = xy.shape[0]
        if eid is None:
            eid = torch.arange(n_pts, device=xy.device) % self.p_poses.shape[0]
        poses, pn, pd, zlo, zhi = self.near(xy, eid, k)
        C, K = pn.shape[1], pn.shape[2]
        flat = lambda a: a.reshape(n_pts * C, *a.shape[2:])
        nw, dw = _to_world(flat(pn), flat(pd), flat(poses))
        valid = flat(pn).pow(2).sum(-1) > 0.5
        live = _slot_is_live(flat(pn), flat(zlo), flat(zhi))
        # max over the prism's own faces of the signed plane distance; padded faces cannot win
        sd = (nw * xy.repeat_interleave(C, 0)[:, None, :]).sum(-1) - dw
        sd = torch.where(valid, sd, torch.full_like(sd, -float("inf"))).max(1).values
        sd = torch.where(live, sd, torch.full_like(sd, float("inf")))
        return sd.view(n_pts, C).min(1).values

    def stats(self) -> Dict[str, float]:
        draws, pieces, dropped, missed = (float(x) for x in self._stat.tolist())
        return {"draws": draws, "pieces_per_layout": pieces / max(draws, 1.0),
                "dropped_per_layout": dropped / max(draws, 1.0), "missed_in_reach": missed,
                "slots": float(self.C), "patterns_max": float(self.P)}

    def describe(self) -> str:
        return (f"procedural obstacles: fraction {self.fraction:g}, density {self.density:g}/10 m, "
                f"up to {self.P} patterns and {self.C} prop slots per env, k_pad {self.k_pad}, "
                f"raceline corridor {'on' if self.has_raceline else 'off'}")
