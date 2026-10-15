"""What the console shows of a contact: a marker where it touched, the car lit while it is held, and
the episode's count -- all read off the frame's contact block (`sim_worker.contact_frame_dict`).

A contact used to be one boolean per car per 25 ms frame (`frame["coll"]`): under soft contact it is
true on the steps that end touching something, so a hit that bounces off turned a car red for one
frame, a wall, a crate and a car looked the same, and nothing said where, how hard or how often. The
simulator now records all of that (`Simulator.contact_frame`), per drawn car:

* ``kind`` / ``point`` / ``severity`` -- this step's hardest contact (0 none, 1 wall, 2 prop, 3 car),
  where it touched (world, m) and the speed into each other there (m/s);
* ``held_s`` -- seconds left of a ``hold_s`` (0.3 s) hold that every contact step restarts, and
  ``held_kind`` / ``held_point`` / ``held_severity`` -- the hardest contact of that hold;
* ``onsets`` / ``hard`` (n, 3) -- the episode's contacts per kind (wall, prop, car), all of them and
  those at or above ``hard_mps``.

No Qt and no GL in here, like `frames.py`: the viewport and the window hand a frame to these functions
and draw what comes back, so the whole data path is testable without a window.

**Interpolation.** `FrameBuffer.frame_to_draw` interpolates poses between two frames and takes every
other key -- this block included -- from the newer one, `f1`, whose simulated time it records as
``t_frame``. Everything here reads one frame's block and nothing else: a marker sits at the point the
block names (never a blend of two frames' points, which would slide a marker between two different
contacts), one per car, so a frame drawn several times draws the same markers each time. Only the
fade follows the drawn time: ``held_s + (t_frame - t)``, so a hold runs out smoothly between frames
instead of in 25 ms steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .theme import C

KIND_NONE, KIND_WALL, KIND_PROP, KIND_CAR = 0, 1, 2, 3
#: What each kind is called on the panels. A "wall" contact is a duct hose or a wall: the simulator
#: resolves both as the track's boundary.
KIND_LABEL = {KIND_WALL: "벽", KIND_PROP: "장애물", KIND_CAR: "차"}
#: Theme tokens, so the panels and the scene spend one colour per kind.
KIND_TOKEN = {KIND_WALL: "contact.wall", KIND_PROP: "contact.prop", KIND_CAR: "contact.car"}


def _rgb(token: str) -> Tuple[float, float, float]:
    h = C[token].lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


#: (n_kinds + 1, 3) float RGB by kind id; row 0 (none) is never drawn.
KIND_RGB = np.array([(0.0, 0.0, 0.0)] + [_rgb(KIND_TOKEN[k]) for k in (KIND_WALL, KIND_PROP, KIND_CAR)],
                    np.float32)

# ---------------------------------------------------------------- how a marker looks
#: [m/s] A marker reaches its full size and brightness at this impact speed. A display scale only:
#: what counts as a *hard* contact is the simulator's `hard_mps` (`contact_severity.HARD_MPS`), which
#: the frame carries and the panel prints, so the console and the evaluation reports agree on it.
MARK_FULL_MPS = 1.5
#: [m] drawn at bumper / hose-centre height, where cars, crates and hoses meet
MARK_Z = 0.08
#: [m] the marker's diameter in the world: a standing touch .. a hit at `MARK_FULL_MPS` or harder
MARK_D = (0.10, 0.44)
#: [px] never smaller than this on screen: a graze .. a full-scale hit. 전체보기 is the default camera,
#: and on a 200 m circuit a car is two pixels long there.
MARK_PX = (8.0, 32.0)
#: a graze's brightness relative to a full-scale hit's
MARK_ALPHA_MIN = 0.5
#: [m] the two cars of a car-car contact both record it, at the same point; one marker, not two
MERGE_M = 0.05
#: columns of `contact_marks`, the vertex layout `gl_scene.Scene.set_contact_marks` uploads
MARK_COLUMNS = ("x", "y", "z", "r", "g", "b", "a", "diameter_m", "min_px", "phase", "hard")

# ---------------------------------------------------------------- how a car is lit
#: How far a contact pulls a car's tint toward its kind's colour: a graze this much, a full-scale hit
#: all the way, both fading with the hold.
HIGHLIGHT_MIN = 0.55
#: Over-bright: the highlight is a flash, brighter than any steady tint, so a car lit orange by a wall
#: does not read as a rival's amber.
HIGHLIGHT_GAIN = 1.8


def _block(fr: dict) -> Optional[dict]:
    """The frame's contact block, checked for the shapes everything below indexes -- or None.

    None for a frame without one (an older worker, a test stub) and for one that is malformed: a
    marker is never worth an exception inside a paint handler.
    """
    blk = fr.get("contact") if isinstance(fr, dict) else None
    if not isinstance(blk, dict):
        return None
    try:
        n = int(fr["n"])
        out = {
            "held_s": np.asarray(blk["held_s"], np.float64).reshape(-1)[:n],
            "held_kind": np.asarray(blk["held_kind"]).astype(np.int64).reshape(-1)[:n],
            "held_point": np.asarray(blk["held_point"], np.float64).reshape(-1, 2)[:n],
            "held_severity": np.asarray(blk["held_severity"], np.float64).reshape(-1)[:n],
            "onsets": np.asarray(blk["onsets"]).astype(np.int64).reshape(-1, 3)[:n],
            "hard": np.asarray(blk["hard"]).astype(np.int64).reshape(-1, 3)[:n],
            "hold_s": float(blk.get("hold_s", 0.3)),
            "hard_mps": float(blk.get("hard_mps", 1.0)),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if any(len(out[k]) != n for k in ("held_s", "held_kind", "held_point", "held_severity", "onsets", "hard")):
        return None
    if out["hold_s"] <= 0.0:
        return None
    out["held_kind"] = np.where((out["held_kind"] >= KIND_WALL) & (out["held_kind"] <= KIND_CAR),
                                out["held_kind"], KIND_NONE)
    return out


def hold_left(fr: dict, blk: Optional[dict] = None) -> Optional[np.ndarray]:
    """(n,) seconds of the hold left at the time this frame is drawn.

    The block's ``held_s`` is at ``t_frame`` (the frame the block came from); an interpolated frame is
    drawn up to one frame earlier, when the hold had that much more to run. A hold that has already
    run out at ``t_frame`` stays out: when it ended is not known, so it is not extended.
    """
    blk = _block(fr) if blk is None else blk
    if blk is None:
        return None
    try:
        t = float(fr.get("t", 0.0))
        t_frame = float(fr.get("t_frame", t))
    except (TypeError, ValueError):
        t = t_frame = 0.0
    lead = max(0.0, t_frame - t)
    held = blk["held_s"]
    left = np.where(held > 0.0, np.minimum(held + lead, blk["hold_s"]), 0.0)
    return np.where(blk["held_kind"] > KIND_NONE, left, 0.0)


def severity_scale(sev) -> np.ndarray:
    """0 (a standing touch) .. 1 (`MARK_FULL_MPS` or harder). Square root, so a graze is small but
    visible rather than a dot."""
    s = np.clip(np.asarray(sev, np.float64) / MARK_FULL_MPS, 0.0, 1.0)
    return np.sqrt(s)


def contact_marks(fr: dict) -> Optional[np.ndarray]:
    """(k, len(MARK_COLUMNS)) float32: one marker per held contact, for `Scene.set_contact_marks`.

    Colour by kind; diameter, on-screen minimum and brightness by severity; ``phase`` runs 0 -> 1 over
    the hold (the shader's ring expands with it) and the alpha fades 1 -> 0; ``hard`` is 1 at or above
    the simulator's `hard_mps`. None when there is nothing to draw.
    """
    blk = _block(fr)
    if blk is None:
        return None
    left = hold_left(fr, blk)
    live = np.flatnonzero(left > 0.0)
    if not len(live):
        return None
    hold = blk["hold_s"]
    kind = blk["held_kind"][live]
    sev = blk["held_severity"][live]
    pt = blk["held_point"][live]
    fade = left[live] / hold
    s = severity_scale(sev)
    rows = np.zeros((len(live), len(MARK_COLUMNS)), np.float32)
    rows[:, 0:2] = pt
    rows[:, 2] = MARK_Z
    rows[:, 3:6] = KIND_RGB[kind]
    rows[:, 6] = fade * (MARK_ALPHA_MIN + (1.0 - MARK_ALPHA_MIN) * s)
    rows[:, 7] = MARK_D[0] + (MARK_D[1] - MARK_D[0]) * s
    rows[:, 8] = MARK_PX[0] + (MARK_PX[1] - MARK_PX[0]) * s
    rows[:, 9] = 1.0 - fade
    rows[:, 10] = (sev >= blk["hard_mps"]).astype(np.float32)
    ok = np.isfinite(rows).all(1)
    rows, kind, sev = rows[ok], kind[ok], sev[ok]
    if not len(rows):
        return None
    # One contact, one marker. The cars of a car-car contact both hold it, at the same point; drawn
    # twice it would be twice as bright as the same hit on a wall. Harder (then fresher) first, and a
    # marker is dropped when one of its kind already stands within MERGE_M.
    order = np.lexsort((-rows[:, 6], -sev))
    keep = []
    for j in order:
        if not any(kind[i] == kind[j] and np.hypot(*(rows[i, :2] - rows[j, :2])) < MERGE_M for i in keep):
            keep.append(j)
    return rows[np.sort(np.asarray(keep, np.int64))]


def car_highlight(fr: dict) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(hot (n, 3), colour (n, 3), weight (n,)) -- how each car is lit by the contact it holds.

    ``hot`` is the over-bright tint a car's body is pulled toward, ``colour`` the kind's plain colour
    (for the ground ring), ``weight`` 0 (nothing held) .. 1 (a full-scale hit, just now): it fades with
    the hold, so a hit stays visible for `hold_s` after the car has backed off. None without a block.
    """
    blk = _block(fr)
    if blk is None:
        return None
    left = hold_left(fr, blk)
    kind = blk["held_kind"]
    fade = left / blk["hold_s"]
    s = severity_scale(blk["held_severity"])
    w = np.where(kind > KIND_NONE, fade * (HIGHLIGHT_MIN + (1.0 - HIGHLIGHT_MIN) * s), 0.0)
    w = np.where(np.isfinite(w), np.clip(w, 0.0, 1.0), 0.0)
    colour = KIND_RGB[kind].astype(np.float32)
    return (colour * HIGHLIGHT_GAIN).astype(np.float32), colour, w.astype(np.float32)


# ---------------------------------------------------------------- the panel's numbers
@dataclass
class ContactSummary:
    """One car's contacts for the panel: the episode's count per kind, the hard ones, and what it
    holds right now."""
    onsets: Tuple[int, int, int]          # wall, prop, car -- this episode
    hard: Tuple[int, int, int]
    hard_mps: float
    held_kind: int = KIND_NONE            # the hold's hardest contact, while one is held
    held_severity: float = 0.0
    held_left: float = 0.0                # [s]
    t_contact: Optional[float] = None     # [s, sim] when the held contact last touched

    @property
    def total(self) -> int:
        return int(sum(self.onsets))

    @property
    def total_hard(self) -> int:
        return int(sum(self.hard))

    @property
    def level(self) -> str:
        """"none", "graze" (contacts, none of them hard) or "hard" -- the question the panel answers."""
        if self.total_hard:
            return "hard"
        return "graze" if self.total else "none"


def focus_summary(fr: dict, f: Optional[int] = None) -> Optional[ContactSummary]:
    """`ContactSummary` of drawn car `f` (default: the frame's focus), or None without a block."""
    blk = _block(fr)
    if blk is None:
        return None
    try:
        f = int(fr.get("focus", 0)) if f is None else int(f)
    except (TypeError, ValueError):
        return None
    if not 0 <= f < len(blk["held_s"]):
        return None
    left = hold_left(fr, blk)
    out = ContactSummary(onsets=tuple(int(v) for v in blk["onsets"][f]),
                         hard=tuple(int(v) for v in blk["hard"][f]), hard_mps=blk["hard_mps"])
    if left is not None and left[f] > 0.0:
        out.held_kind = int(blk["held_kind"][f])
        out.held_severity = float(blk["held_severity"][f])
        out.held_left = float(left[f])
        try:
            t = float(fr.get("t", 0.0))
        except (TypeError, ValueError):
            t = 0.0
        out.t_contact = t - (blk["hold_s"] - out.held_left)
    return out


def kind_html(kind: int, text: Optional[str] = None) -> str:
    """The kind's name (or `text`) in its marker colour -- the panel's legend for the scene."""
    return f"<span style='color:{C[KIND_TOKEN[kind]]}'>{text or KIND_LABEL[kind]}</span>"


def breakdown_html(s: ContactSummary) -> str:
    """"벽 3 (세게 1) · 장애물 0 · 차 1", each name in its marker colour."""
    parts = []
    for j, k in enumerate((KIND_WALL, KIND_PROP, KIND_CAR)):
        part = f"{kind_html(k)} {s.onsets[j]}"
        if s.hard[j]:
            part += f" <span style='color:{C['danger']}'>(세게 {s.hard[j]})</span>"
        parts.append(part)
    return " · ".join(parts)


def tile_text(s: ContactSummary) -> str:
    """The tile's number: contacts this episode, and how many were hard."""
    return f"{s.total}" if not s.total else f"{s.total} · 세게 {s.total_hard}"


def hard_label(hard_mps: float) -> str:
    """The threshold, printed where the count is: "세게 ≥1.0 m/s"."""
    return f"세게 ≥{hard_mps:.1f} m/s"


class LastContact:
    """The most recent contact of the watched car, kept after its marker has faded.

    The marker says how hard a hit was for `hold_s`; this keeps the number until the next one, so a
    hit seen out of the corner of an eye can still be read. Forgotten when the watched car, the
    session or the episode changes (the episode's count going down is its reset).
    """

    def __init__(self):
        self.key = None
        self._clear()

    def _clear(self) -> None:
        self.kind = KIND_NONE
        self.severity = 0.0
        self.hard = False
        self.t_contact: Optional[float] = None
        self.now = False                  # still held: touching, or within the hold of the last touch
        self._total = 0

    def update(self, key, s: Optional[ContactSummary]) -> None:
        """`key` names the watched car in its session; `s` is its `focus_summary` (None: nothing to
        read, which forgets the last contact as well)."""
        if s is None or key != self.key or s.total < self._total:
            self._clear()
            self.key = key
        if s is None:
            return
        self._total = s.total
        self.now = s.held_kind != KIND_NONE and s.t_contact is not None
        if self.now:
            # While a hold runs its record is the hold's hardest contact, which is what "how hard
            # was it" asks; the time is the hold's latest contact step.
            self.kind, self.severity = s.held_kind, s.held_severity
            self.hard = s.held_severity >= s.hard_mps
            self.t_contact = s.t_contact

    def text(self, t_now: Optional[float]) -> str:
        if self.kind == KIND_NONE or self.t_contact is None:
            return "—"
        what = "세게" if self.hard else "스침"
        if self.now:
            age = " · 지금"
        elif t_now is None:
            age = ""
        else:
            age = f" · {max(0.0, float(t_now) - self.t_contact):.1f}초 전"
        return f"{kind_html(self.kind)} {self.severity:.2f} m/s · {what}{age}"
