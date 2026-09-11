"""Procedural obstacle props: stylised meshes plus the convex volume they physically occupy.

Every prop returns two things that must stay consistent:

* **parts** -- flat-shaded triangle meshes in the format `gl_scene.Scene.add_static_mesh` already
  takes: `pos (V,3) f32`, `nrm (V,3) f32`, `col (V,4) f32`, `idx (M,) i32`, plus a material name
  from `gl_scene.MATERIALS`.
* **envelope** -- the convex prism the prop occupies: a CCW footprint polygon in local XY and a
  height. Collision and LiDAR are meant to use this, so the visual mesh is built to stay *inside*
  it. Bevels, grooves and ribs are cut inward only; nothing decorative pokes out. The prism is
  therefore an over-approximation of the visible surface by at most `bevel_tolerance` metres, and
  never an under-approximation.

Conventions the caller owns
---------------------------
Metres, local **z-up**, origin at the footprint centroid, base at **z = 0**. Yaw and position are
applied outside, by the instance matrix -- no prop bakes in a rotation.

Two details about the vertex colour that are easy to get wrong:

* `col[:, 3]` is **not opacity**. In the scene's fragment shader alpha is only read when the
  stripe mode is on, where it is an arc-length parameter for duct hose banding
  (`base *= 0.74 + 0.26*sin(v_col.a * 2pi/0.024)`). Every prop here writes alpha 1.0 and must be
  uploaded with `stripes=False`.
* The final colour is `col.rgb * instance_tint.rgb`, so a tint of white leaves these colours as
  authored.

Solidity
--------
No prop has a hole you can see through. A pallet's slots or a crate's gaps would read as openings
while the collision prism stays solid, which is a lie the renderer tells on every frame; plank
courses and slats are cut as shallow relief instead.

Determinism
-----------
Shape, size and colour follow from a spec plus an integer seed. The generator uses
`np.random.default_rng(seed)` only for style jitter within declared bounds -- face shading, plank
widths -- never for anything that changes the envelope. Same spec and seed, byte-identical arrays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The contract version in `work/claude-map-redesign/interface.md`. The integration side asserts
#: this equals 1 when it vendors this file, so any shape change to the dataclasses below must bump
#: it and be written up there.
PROPS_SCHEMA_VERSION = 1

#: Material names `gl_scene.MATERIALS` knows. Anything else silently becomes "plastic" there, so
#: the tests check membership rather than letting a typo turn a steel drum matte. Kept as a plain
#: tuple of literals on purpose: this module must not import `gl_scene`, which would drag GL into
#: the worker process that builds geometry on the CPU.
MATERIALS = ("rubber", "plastic", "metal", "blue", "glass", "paint", "pcb")

#: Per-prop budgets the integration side sets: footprint vertex count drives rasterisation and SAT
#: contact cost linearly, triangles drive the static batch.
MAX_TRIS_PER_PROP = 1500
MAX_FOOTPRINT_VERTS = 24

F32 = np.float32
I32 = np.int32


@dataclass(frozen=True)
class MeshPart:
    """One flat-shaded, single-material chunk of a prop."""
    name: str
    material: str
    pos: np.ndarray          # (V,3) float32
    nrm: np.ndarray          # (V,3) float32
    col: np.ndarray          # (V,4) float32, alpha always 1.0
    idx: np.ndarray          # (M,) int32

    @property
    def n_tris(self) -> int:
        return int(len(self.idx) // 3)

    @property
    def n_verts(self) -> int:
        return int(len(self.pos))


@dataclass(frozen=True)
class Envelope:
    """The convex prism the prop physically occupies: footprint (CCW, metres) x [0, height]."""
    footprint: np.ndarray    # (K,2) float64, counter-clockwise, centroid at the origin
    height: float
    #: How far the prism can sit outside the visible surface, in metres -- the **measured** maximum
    #: over a dense set of horizontal rays (`envelope_error.py`), not the nominal chamfer depth.
    #: Because the prism encloses the mesh, this is the distance at which a contact or a LiDAR
    #: return can occur in space where nothing is drawn. A test enforces that the declared value
    #: bounds the measured one.
    bevel_tolerance: float

    @property
    def radius(self) -> float:
        """Circumscribed radius about the local origin -- handy for broad-phase culling."""
        return float(np.max(np.linalg.norm(self.footprint, axis=1)))

    @property
    def inradius(self) -> float:
        """Largest circle about the local origin that fits inside the footprint.

        This is the number that decides whether a prop survives rasterisation onto a grid: a
        footprint thinner than a cell can fall between sample points. It is reported rather than
        used to force a minimum size -- see the note on `marker_post`.
        """
        a = self.footprint
        e = np.roll(a, -1, 0) - a
        return float(np.min(np.abs(e[:, 0] * a[:, 1] - e[:, 1] * a[:, 0])
                            / (np.linalg.norm(e, axis=1) + 1e-15)))

    def bounds(self) -> Tuple[float, float, float, float, float, float]:
        lo = self.footprint.min(0)
        hi = self.footprint.max(0)
        return float(lo[0]), float(lo[1]), 0.0, float(hi[0]), float(hi[1]), float(self.height)


@dataclass(frozen=True)
class Prop:
    name: str
    parts: Tuple[MeshPart, ...]
    envelope: Envelope
    spec: Dict[str, float] = field(default_factory=dict)

    @property
    def n_tris(self) -> int:
        return sum(p.n_tris for p in self.parts)

    @property
    def n_verts(self) -> int:
        return sum(p.n_verts for p in self.parts)


# ==================================================================== geometry helpers
def _regular_polygon(n: int, radius: float, phase: float = 0.0) -> np.ndarray:
    a = phase + np.arange(n) * (2 * math.pi / n)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], 1)


def _rect(hx: float, hy: float) -> np.ndarray:
    return np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]][::-1], float)   # CCW


def _chamfered_rect(hx: float, hy: float, c: float) -> np.ndarray:
    """Rectangle with its four corners cut by `c`. Strictly inside the rectangle."""
    c = min(c, 0.45 * min(hx, hy))
    return np.array([
        [hx, hy - c], [hx - c, hy], [-hx + c, hy], [-hx, hy - c],
        [-hx, -hy + c], [-hx + c, -hy], [hx - c, -hy], [hx, -hy + c],
    ], float)


def _trapezoid(hx: float, hy_front: float, hy_back: float) -> np.ndarray:
    """Convex trapezoid, wider at -x. Centroid is *not* the origin; recentred by the caller."""
    return np.array([[hx, hy_front], [-hx, hy_back], [-hx, -hy_back], [hx, -hy_front]], float)


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _centroid(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = 0.5 * float(cross.sum())
    cx = float(((x + np.roll(x, -1)) * cross).sum() / (6 * a))
    cy = float(((y + np.roll(y, -1)) * cross).sum() / (6 * a))
    return np.array([cx, cy])


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if _polygon_area(poly) > 0 else poly[::-1].copy()


def _shrink(poly: np.ndarray, d: float) -> np.ndarray:
    """Scale a convex polygon about its centroid so it sits `~d` metres inside.

    A true offset would need edge intersection; scaling is enough here because every inset is a
    shallow decorative one and the result is always *inside* the original, which is the property
    the envelope guarantee depends on.
    """
    c = _centroid(poly)
    r = float(np.max(np.linalg.norm(poly - c, axis=1)))
    return c + (poly - c) * max(0.0, (r - d) / r)


class _Builder:
    """Accumulates flat-shaded triangles. Every face gets its own vertices, so normals are exact
    and the shading stays faceted, which is the look these props want."""

    def __init__(self):
        self.pos: List[np.ndarray] = []
        self.nrm: List[np.ndarray] = []
        self.col: List[np.ndarray] = []
        self.idx: List[np.ndarray] = []
        self._n = 0

    def face(self, verts: Sequence[Sequence[float]], colour) -> None:
        """One planar convex polygon, triangulated as a fan, normal from its winding."""
        v = np.asarray(verts, float)
        if len(v) < 3:
            return
        n = np.cross(v[1] - v[0], v[2] - v[0])
        ln = float(np.linalg.norm(n))
        if ln < 1e-12:
            return                                     # degenerate: drop rather than emit NaN
        n = n / ln
        k = len(v)
        self.pos.append(v)
        self.nrm.append(np.tile(n, (k, 1)))
        c = np.asarray(colour, float)
        if c.shape == (3,):
            c = np.concatenate([c, [1.0]])
        self.col.append(np.tile(c, (k, 1)))
        fan = [(0, i, i + 1) for i in range(1, k - 1)]
        self.idx.append(np.asarray(fan, int).reshape(-1) + self._n)
        self._n += k

    def prism(self, poly: np.ndarray, z0: float, z1: float, side_col, top_col=None, bot_col=None,
              cap_top: bool = True, cap_bot: bool = True, top_poly: Optional[np.ndarray] = None,
              side_shade: Optional[Callable[[int], float]] = None) -> None:
        """Closed prism (or frustum, with `top_poly`) over a convex CCW footprint."""
        top = poly if top_poly is None else top_poly
        k = len(poly)
        for i in range(k):
            j = (i + 1) % k
            shade = 1.0 if side_shade is None else side_shade(i)
            c = np.asarray(side_col, float).copy()
            c[:3] = np.clip(c[:3] * shade, 0.0, 1.0)
            self.face([[poly[i, 0], poly[i, 1], z0], [poly[j, 0], poly[j, 1], z0],
                       [top[j, 0], top[j, 1], z1], [top[i, 0], top[i, 1], z1]], c)
        if cap_top:
            self.face([[p[0], p[1], z1] for p in top], top_col if top_col is not None else side_col)
        if cap_bot:
            self.face([[p[0], p[1], z0] for p in poly[::-1]],
                      bot_col if bot_col is not None else side_col)

    def build(self, name: str, material: str) -> MeshPart:
        return MeshPart(
            name=name, material=material,
            pos=np.concatenate(self.pos).astype(F32),
            nrm=np.concatenate(self.nrm).astype(F32),
            col=np.concatenate(self.col).astype(F32),
            idx=np.concatenate(self.idx).astype(I32),
        )


def _panel_shade(poly: np.ndarray) -> Callable[[int], float]:
    """A small, *coherent* brightness step per side, from the side's outward direction.

    An earlier version jittered each face randomly. On a chamfered box that gives every narrow
    corner strip its own tone, and a flat cardboard panel ends up reading as creased foil. Real
    variation between faces comes from which way they point, so that is what this uses: coplanar
    faces get identical tone, and a chamfer strip lands between its two neighbours instead of
    somewhere random.
    """
    e = np.roll(poly, -1, 0) - poly
    n = np.stack([e[:, 1], -e[:, 0]], 1)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    shade = 1.0 + 0.045 * n[:, 0] - 0.030 * n[:, 1]          # gentle directional key
    return lambda i: float(shade[i % len(shade)])


def _tone(rng: np.random.Generator, spread: float = 0.03) -> float:
    """One overall tone multiplier per prop, so two instances differ without any face crumpling."""
    return float(1.0 + rng.uniform(-spread, spread))


def _spec(env: Envelope, seed: int, **extra) -> Dict[str, float]:
    """The dimensions a prop **actually came out at**, read back off its envelope.

    Deliberately not the requested values. A builder may clamp a chamfer or round a facet count,
    and the integration side checks lane clearance against this dict before placing anything -- so
    it has to describe the geometry that exists, not the geometry that was asked for.
    """
    x0, y0, _, x1, y1, _ = env.bounds()
    s: Dict[str, float] = {
        "width": round(x1 - x0, 6),                 # x extent of the footprint
        "depth": round(y1 - y0, 6),                 # y extent of the footprint
        "height": round(float(env.height), 6),
        "circumradius": round(env.radius, 6),
        "inradius": round(env.inradius, 6),
        "footprint_vertices": int(len(env.footprint)),
        "bevel_tolerance": round(float(env.bevel_tolerance), 6),
        "seed": int(seed),
    }
    s.update({k: round(float(v), 6) for k, v in extra.items()})
    return s


# ==================================================================== the props
def cardboard_box(width=0.36, depth=0.30, height=0.30, seed=0) -> Prop:
    """A taped cardboard box: chamfered edges, a tape band over the lid and down two sides, and
    the seam where the flaps meet."""
    rng = np.random.default_rng(seed)
    hx, hy, c = width / 2, depth / 2, 0.010
    # The declared footprint is the full rectangle; the card wall is set in by one decal
    # thickness so tape and seams can sit proud of it and still be inside the prism.
    decal = 0.0015
    foot = _ensure_ccw(_rect(hx, hy))
    body = _ensure_ccw(_chamfered_rect(hx - decal, hy - decal, c))

    t = _tone(rng)
    base = np.array([0.62, 0.47, 0.31]) * t
    lid = base * 1.05
    tape = np.array([0.78, 0.72, 0.60]) * t

    b = _Builder()
    # Body up to just under the lid, then a shallow chamfered cap: the top edge reads as folded
    # card rather than a cut cube, and both stay inside the rectangle.
    z_top = height - 2 * decal                       # the lid surface; decals stack up to `height`
    z_lid = z_top - c
    panel = _panel_shade(body)
    b.prism(body, 0.0, z_lid, base, cap_top=False, side_shade=panel)
    b.prism(body, z_lid, z_top, lid, top_poly=_shrink(body, c), cap_bot=False,
            top_col=lid * 0.98, side_shade=panel)

    # Flap seam: a thin darker strip along the lid's long axis, drawn just above the lid face.
    eps = 0.0006
    seam_w = 0.006
    z_seam = z_top + eps
    b.face([[-hx + c, -seam_w, z_seam], [hx - c, -seam_w, z_seam],
            [hx - c, seam_w, z_seam], [-hx + c, seam_w, z_seam]], base * 0.82)
    # Tape: over the lid across the seam, and down the two short sides. Offset outward by `eps`
    # only, which is far inside the chamfer, so the envelope still holds.
    tw = 0.026
    z_tape = height - eps                            # on top of the lid, still under the prism lid
    b.face([[-tw, -hy + c + 0.004, z_tape], [tw, -hy + c + 0.004, z_tape],
            [tw, hy - c - 0.004, z_tape], [-tw, hy - c - 0.004, z_tape]], tape)
    for sgn in (+1, -1):
        y = sgn * (hy - eps)                         # proud of the inset wall, inside the footprint
        b.face([[-tw, y, z_top - c], [tw, y, z_top - c],
                [tw, y, height * 0.45], [-tw, y, height * 0.45]][::sgn], tape * 0.95)
    part = b.build("box", "rubber")
    # 18 mm, not the 10 mm chamfer: at a cut corner the prism corner stands sqrt(2)*c out, plus the
    # decal inset. Measured by envelope_error.py, enforced by test_declared_tolerance_bounds_...
    env = Envelope(foot, height, 0.018)
    return Prop("cardboard_box", (part,), env, _spec(env, seed))


def wooden_crate(width=0.42, depth=0.34, height=0.36, seed=1) -> Prop:
    """A closed wooden crate: corner posts and horizontal plank courses, cut as relief rather than
    as gaps -- an open slat would read as a hole the collision prism does not have."""
    rng = np.random.default_rng(seed)
    hx, hy = width / 2, depth / 2
    groove = 0.008                                  # how deep the plank joints are recessed
    foot = _ensure_ccw(_rect(hx, hy))
    post = 0.045                                    # corner post width

    t = _tone(rng)
    wood = np.array([0.55, 0.40, 0.24]) * t
    plank = np.array([0.62, 0.46, 0.28]) * t
    b = _Builder()
    shade = _panel_shade(_ensure_ccw(_rect(hx, hy)))

    # Recessed panel body, then the four corner posts and the plank courses on top of it.
    inner = _ensure_ccw(_rect(hx - groove, hy - groove))
    b.prism(inner, 0.0, height, wood * 0.92, top_col=wood * 0.86, side_shade=shade)

    n_courses = 3
    edges = np.linspace(0.0, height, n_courses + 1)
    gap = 0.010
    for ci in range(n_courses):
        z0, z1 = edges[ci] + (gap if ci else 0.004), edges[ci + 1] - gap * 0.5
        tone = plank * (0.95 + 0.05 * rng.random())      # per plank course, deliberately varied
        for sgn, axis in ((+1, 0), (-1, 0), (+1, 1), (-1, 1)):
            if axis == 0:
                x = sgn * hx
                b.face([[x, -hy + post, z0], [x, hy - post, z0], [x, hy - post, z1], [x, -hy + post, z1]][::sgn],
                       tone)
            else:
                y = sgn * hy
                b.face([[-hx + post, y, z0], [hx - post, y, z0], [hx - post, y, z1], [-hx + post, y, z1]][::-sgn],
                       tone * 0.97)
    # Corner posts: full height, flush with the footprint corners.
    for sx in (+1, -1):
        for sy in (+1, -1):
            p = _ensure_ccw(np.array([
                [sx * hx, sy * hy], [sx * (hx - post), sy * hy],
                [sx * (hx - post), sy * (hy - post)], [sx * hx, sy * (hy - post)]], float))
            b.prism(p, 0.0, height, wood, top_col=wood * 1.04, cap_bot=False)
    part = b.build("crate", "rubber")
    env = Envelope(foot, height, 0.012)
    return Prop("wooden_crate", (part,), env, _spec(env, seed))


def steel_drum(radius=0.145, height=0.44, facets=16, seed=2) -> Prop:
    """A faceted oil drum: two rolling ribs and a lip at each end, all cut inward from the
    circumscribed prism so the declared footprint stays valid."""
    rng = np.random.default_rng(seed)
    foot = _ensure_ccw(_regular_polygon(facets, radius))
    inset = 0.012
    body = _shrink(foot, inset)                     # barrel wall sits inside the ribs

    t = _tone(rng)
    steel = np.array([0.40, 0.45, 0.50]) * t
    rib = np.array([0.34, 0.38, 0.43]) * t
    top = np.array([0.46, 0.51, 0.55]) * t

    b = _Builder()
    shade = _panel_shade(foot)
    rim_h = 0.022
    rib_h = 0.030
    rib_z = (height * 0.34, height * 0.64)

    # bottom rim -> wall -> rib -> wall -> rib -> wall -> top rim, each a closed prism band
    bands = [(0.0, rim_h, foot, rib), (rim_h, rib_z[0], body, steel),
             (rib_z[0], rib_z[0] + rib_h, foot, rib), (rib_z[0] + rib_h, rib_z[1], body, steel),
             (rib_z[1], rib_z[1] + rib_h, foot, rib),
             (rib_z[1] + rib_h, height - rim_h, body, steel),
             (height - rim_h, height, foot, rib)]
    for z0, z1, poly, colour in bands:
        b.prism(poly, z0, z1, colour, cap_top=False, cap_bot=False, side_shade=shade)
        # close the step between a wide band and the narrower wall above/below it
    for z, outer, inner, up in ((rim_h, foot, body, True), (rib_z[0], body, foot, True),
                                (rib_z[0] + rib_h, foot, body, True), (rib_z[1], body, foot, True),
                                (rib_z[1] + rib_h, foot, body, True),
                                (height - rim_h, body, foot, True)):
        ring = [[outer[i, 0], outer[i, 1], z] for i in range(len(outer))]
        ring2 = [[inner[i, 0], inner[i, 1], z] for i in range(len(inner))]
        for i in range(len(outer)):
            j = (i + 1) % len(outer)
            b.face([ring[i], ring[j], ring2[j], ring2[i]], rib * 0.9)
    b.face([[p[0], p[1], 0.0] for p in foot[::-1]], steel * 0.7)
    # The lid is *recessed*, not stacked on top: an annular rim at the full height, a step down,
    # and the lid face below it. Nothing sits above the declared height.
    lid = _shrink(foot, 0.018)
    z_lid = height - 0.005
    k = len(foot)
    for i in range(k):
        j = (i + 1) % k
        b.face([[foot[i, 0], foot[i, 1], height], [foot[j, 0], foot[j, 1], height],
                [lid[j, 0], lid[j, 1], height], [lid[i, 0], lid[i, 1], height]], top)
        b.face([[lid[i, 0], lid[i, 1], height], [lid[j, 0], lid[j, 1], height],
                [lid[j, 0], lid[j, 1], z_lid], [lid[i, 0], lid[i, 1], z_lid]], top * 0.88)
    b.face([[p[0], p[1], z_lid] for p in lid], top * 0.94)
    bung = _regular_polygon(8, radius * 0.22, phase=0.2)
    b.face([[p[0] + radius * 0.40, p[1], z_lid + 0.0018] for p in bung], top * 1.05)
    part = b.build("drum", "metal")
    env = Envelope(foot, height, 0.013)
    return Prop("steel_drum", (part,), env, _spec(env, seed, facets=facets))


def crate_stack_low(width=0.62, depth=0.44, height=0.26, seed=3) -> Prop:
    """Low and wide: two courses of boxes banded together. The upper course is inset, never
    overhanging, so the ground rectangle stays the true footprint."""
    rng = np.random.default_rng(seed)
    hx, hy = width / 2, depth / 2
    foot = _ensure_ccw(_rect(hx, hy))
    c = 0.008
    split = height * 0.52

    t = _tone(rng)
    lower = np.array([0.58, 0.44, 0.30]) * t
    upper = np.array([0.64, 0.50, 0.34]) * t
    strap = np.array([0.25, 0.26, 0.30])

    b = _Builder()
    low_poly = _chamfered_rect(hx, hy, c)
    up = _chamfered_rect(hx - 0.02, hy - 0.015, c)
    b.prism(low_poly, 0.0, split, lower, top_col=lower * 0.9, side_shade=_panel_shade(low_poly))
    # The upper course starts *below* the lower one's top face. An earlier version left a 4 mm
    # slot here, which the ray measurement caught as 144 rays passing straight through a prop the
    # collision prism says is solid.
    b.prism(up, split - 0.006, height, upper, top_col=upper * 1.03, cap_bot=False,
            side_shade=_panel_shade(up))
    # a strap around the lower course, drawn as a thin band just proud of the wall
    eps = 0.0012
    for sgn in (+1, -1):
        y = sgn * (hy - eps)
        b.face([[-hx + c, y, split * 0.35], [hx - c, y, split * 0.35],
                [hx - c, y, split * 0.55], [-hx + c, y, split * 0.55]][::sgn], strap)
    for sgn in (+1, -1):
        x = sgn * (hx - eps)
        b.face([[x, -hy + c, split * 0.35], [x, hy - c, split * 0.35],
                [x, hy - c, split * 0.55], [x, -hy + c, split * 0.55]][::-sgn], strap * 0.95)
    part = b.build("stack", "rubber")
    env = Envelope(foot, height, 0.030)
    return Prop("crate_stack_low", (part,), env, _spec(env, seed))


def barrier_block(width=0.54, depth=0.26, height=0.22, taper=0.6923, seed=4) -> Prop:
    """A non-rectangular, low, wide block: a trapezoidal plan with a battered face, the shape a
    track-side barrier has. The footprint is a genuine trapezoid, not a rectangle.

    `width`/`depth` are the common dimension keys; `depth` is the wide (-x) end and the narrow end
    is `depth * taper`, which keeps the plan a trapezoid at any size.
    """
    rng = np.random.default_rng(seed)
    front, back = depth * taper, depth
    raw = _trapezoid(width / 2, front / 2, back / 2)
    poly = _ensure_ccw(raw - _centroid(raw))         # recentre so the origin is the centroid
    foot = poly.copy()

    t = _tone(rng)
    body = np.array([0.72, 0.71, 0.68]) * t
    top = np.array([0.78, 0.77, 0.74]) * t
    stripe = np.array([0.80, 0.42, 0.16])

    b = _Builder()
    shade = _panel_shade(poly)
    batter = 0.030                                   # inward lean of the upper part
    kick = height * 0.30
    b.prism(poly, 0.0, kick, body * 0.94, cap_top=False, side_shade=shade)
    b.prism(poly, kick, height, body, top_poly=_shrink(poly, batter), cap_bot=False,
            top_col=top, side_shade=shade)
    # Hazard chevrons on the two long faces. Drawn on the battered wall by interpolating between
    # the base polygon and the shrunk top one, so each band follows the lean instead of floating,
    # and pushed out by `eps` only -- still inside the footprint.
    eps = 0.0015
    top_poly = _shrink(poly, batter)
    lengths = np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1)
    long_edges = [i for i in range(len(poly)) if lengths[i] > 0.6 * lengths.max()]
    z0, z1 = kick + 0.010, height - 0.006
    f0 = (z0 - kick) / (height - kick)
    f1 = (z1 - kick) / (height - kick)
    n_bands = 5
    for i in long_edges:
        j = (i + 1) % len(poly)
        a0, a1 = poly[i], poly[j]
        b0, b1 = top_poly[i], top_poly[j]
        out = np.array([-(a1 - a0)[1], (a1 - a0)[0]])
        out = out / (np.linalg.norm(out) + 1e-12) * eps
        for k in range(n_bands):
            t0, t1 = k / n_bands, (k + 0.5) / n_bands
            lo0 = a0 + (a1 - a0) * t0
            lo1 = a0 + (a1 - a0) * t1
            hi0 = b0 + (b1 - b0) * t0
            hi1 = b0 + (b1 - b0) * t1
            p00 = lo0 + (hi0 - lo0) * f0
            p10 = lo1 + (hi1 - lo1) * f0
            p01 = lo0 + (hi0 - lo0) * f1
            p11 = lo1 + (hi1 - lo1) * f1
            b.face([[p00[0] + out[0], p00[1] + out[1], z0],
                    [p10[0] + out[0], p10[1] + out[1], z0],
                    [p11[0] + out[0], p11[1] + out[1], z1],
                    [p01[0] + out[0], p01[1] + out[1], z1]],
                   stripe if k % 2 == 0 else body * 0.82)
    part = b.build("barrier", "paint")
    env = Envelope(foot, height, 0.031)
    return Prop("barrier_block", (part,), env, _spec(env, seed, taper=taper, narrow_end=front))


def marker_post(radius=0.055, height=0.62, seed=5) -> Prop:
    """Tall and thin: a tapered hexagonal post with reflective bands and a weighted foot.

    The default hexagon has a 55 mm circumradius, so an inradius of 47.6 mm -- **about half a cell
    on Monza's 0.09585 m grid**. That is real: a marker post is a slender thing, and inflating it
    to suit the coarser grid would make every map show a fat post to hide a rasteriser limit. The
    geometry stays honest and `spec["inradius"]` states it; representing it is the sensor and
    contact side's call.
    """
    rng = np.random.default_rng(seed)
    foot_poly = _ensure_ccw(_regular_polygon(6, radius, phase=math.pi / 6))
    base_h = 0.045
    taper = 0.016

    t = _tone(rng)
    dark = np.array([0.20, 0.21, 0.24]) * t
    post = np.array([0.86, 0.84, 0.80]) * t
    band = np.array([0.88, 0.32, 0.22])

    b = _Builder()
    shade = _panel_shade(foot_poly)
    b.prism(foot_poly, 0.0, base_h, dark, top_col=dark * 1.1, side_shade=shade)
    shaft = _shrink(foot_poly, 0.012)
    tip = _shrink(foot_poly, 0.012 + taper)
    b.prism(shaft, base_h, height, post, top_poly=tip, cap_bot=False, top_col=post * 1.05,
            side_shade=shade)
    # two reflective bands, sitting just outside the shaft but inside the base footprint
    eps = 0.0012
    for z0, z1 in ((height * 0.52, height * 0.62), (height * 0.74, height * 0.84)):
        f0 = (z0 - base_h) / (height - base_h)
        f1 = (z1 - base_h) / (height - base_h)
        r0 = shaft + (tip - shaft) * f0
        r1 = shaft + (tip - shaft) * f1
        k = len(r0)
        for i in range(k):
            j = (i + 1) % k
            out = np.array([r0[i], r0[j]]).mean(0)
            out = out / (np.linalg.norm(out) + 1e-12) * eps
            b.face([[r0[i, 0] + out[0], r0[i, 1] + out[1], z0],
                    [r0[j, 0] + out[0], r0[j, 1] + out[1], z0],
                    [r1[j, 0] + out[0], r1[j, 1] + out[1], z1],
                    [r1[i, 0] + out[0], r1[i, 1] + out[1], z1]], band)
    part = b.build("post", "plastic")
    env = Envelope(foot_poly, height, 0.029)
    return Prop("marker_post", (part,), env, _spec(env, seed))


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2D scalar cross product. `np.cross` on 2-vectors is deprecated in NumPy 2."""
    return float(a[0] * b[1] - a[1] * b[0])


def _hull2d(pts: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points, CCW (Andrew's monotone chain). numpy only, no scipy."""
    p = np.unique(np.round(pts, 9), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(points):
        out = []
        for q in points:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], q - out[-2]) <= 1e-15:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(p), half(p[::-1])
    return np.asarray(lower[:-1] + upper[:-1])


def cross_section(prop: "Prop", z: float) -> np.ndarray:
    """Points where the prop's surface meets the plane at height `z`.

    Every triangle edge that straddles `z` is interpolated, and vertices sitting on the plane are
    kept. Used to measure how far the declared prism sits outside the real silhouette, and to build
    the piecewise sections below.
    """
    pts = []
    for part in prop.parts:
        v = part.pos.astype(float)[part.idx.reshape(-1, 3)]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            p0, p1 = v[:, a], v[:, b]
            dz = p1[:, 2] - p0[:, 2]
            straddle = ((p0[:, 2] - z) * (p1[:, 2] - z) < 0) & (np.abs(dz) > 1e-12)
            if straddle.any():
                t = ((z - p0[straddle, 2]) / dz[straddle])[:, None]
                pts.append((p0[straddle] + t * (p1[straddle] - p0[straddle]))[:, :2])
        on = np.abs(v[:, :, 2] - z) < 1e-9
        if on.any():
            pts.append(v[on][:, :2])
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def sections(prop: "Prop", n_bands: int = 8, samples_per_band: int = 12) -> List[dict]:
    """Piecewise convex sections: a tighter physical bound than the single prism.

    Each band is the convex hull of the prop's cross-sections sampled across that band, so it
    contains the surface everywhere in the band while hugging a taper far more closely than the
    full-height prism does. Offered for the case where core wants accurate tracing; the single
    prism stays available for the cheap case.
    """
    h = prop.envelope.height
    edges = np.linspace(0.0, h, n_bands + 1)
    out = []
    for i in range(n_bands):
        z0, z1 = float(edges[i]), float(edges[i + 1])
        zs = np.linspace(z0 + 1e-6, z1 - 1e-6, samples_per_band)
        pts = [cross_section(prop, float(z)) for z in zs]
        pts = [q for q in pts if len(q)]
        if not pts:
            continue
        hull = _hull2d(np.concatenate(pts))
        if len(hull) >= 3:
            out.append({"z0": z0, "z1": z1, "polygon": _ensure_ccw(hull)})
    return out


# ==================================================================== registry
#: Style name -> builder. The integration side names props as strings in map metadata, so it
#: resolves through here rather than importing each function; adding a style then costs it nothing.
REGISTRY: Dict[str, Callable[..., Prop]] = {
    "cardboard_box": cardboard_box,
    "wooden_crate": wooden_crate,
    "steel_drum": steel_drum,
    "crate_stack_low": crate_stack_low,
    "barrier_block": barrier_block,
    "marker_post": marker_post,
}

#: The style names, in a stable order.
STYLES: Tuple[str, ...] = tuple(REGISTRY)

#: Which dimension keywords each style accepts. `height` is universal; rectangular plans take
#: `width`/`depth`, radial ones take `radius`. Anything else is a typo and `build` refuses it.
DIMS: Dict[str, Tuple[str, ...]] = {
    "cardboard_box": ("width", "depth", "height"),
    "wooden_crate": ("width", "depth", "height"),
    "steel_drum": ("radius", "height", "facets"),
    "crate_stack_low": ("width", "depth", "height"),
    "barrier_block": ("width", "depth", "height", "taper"),
    "marker_post": ("radius", "height"),
}

#: Older name for `REGISTRY`, kept so existing scripts here keep working. Same object.
PRESETS = REGISTRY


def build(style: str, seed: int = 0, **dims) -> Prop:
    """Build a prop by style name.

    Every failure here is loud on purpose. A silent fallback to some default prop would let a typo
    in map metadata put a different object on the track and never say so, which is the one failure
    mode that survives all the way to a rendered frame looking plausible.
    """
    if style not in REGISTRY:
        raise ValueError(f"unknown prop style {style!r}; known: {', '.join(STYLES)}")
    allowed = DIMS[style]
    unknown = sorted(set(dims) - set(allowed))
    if unknown:
        raise TypeError(f"{style} does not take {', '.join(unknown)}; it takes {', '.join(allowed)}")
    for key, value in dims.items():
        if key == "facets":
            if int(value) != value or int(value) < 6:
                raise ValueError(f"{style}: facets must be an integer >= 6, got {value!r}")
            continue
        v = float(value)
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(f"{style}: {key} must be finite and positive, got {value!r}")
    return REGISTRY[style](seed=int(seed), **dims)


def build_all(seed_offset: int = 0) -> List[Prop]:
    """Every style at its default size. `seed_offset` shifts the style jitter, not the sizes."""
    return [build(s, seed=i + seed_offset) for i, s in enumerate(STYLES)]


def prop_stats(prop: Prop) -> dict:
    x0, y0, z0, x1, y1, z1 = prop.envelope.bounds()
    return {
        "name": prop.name,
        "parts": [{"name": p.name, "material": p.material, "verts": p.n_verts, "tris": p.n_tris}
                  for p in prop.parts],
        "verts": prop.n_verts, "tris": prop.n_tris,
        "footprint_vertices": int(len(prop.envelope.footprint)),
        "height_m": round(float(prop.envelope.height), 4),
        "bevel_tolerance_m": round(float(prop.envelope.bevel_tolerance), 4),
        "circumradius_m": round(prop.envelope.radius, 4),
        "bounds_m": [round(v, 4) for v in (x0, y0, z0, x1, y1, z1)],
        "footprint_area_m2": round(abs(_polygon_area(prop.envelope.footprint)), 5),
        "bytes_gpu": int(sum(p.pos.nbytes + p.nrm.nbytes + p.col.nbytes + p.idx.nbytes
                             for p in prop.parts)),
        "spec": prop.spec,
    }
