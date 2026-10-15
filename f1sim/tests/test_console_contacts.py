"""The console's contact display: a marker where a car touched, the car lit in the colour of what it
touched, and the 접촉 panel -- from the frame's contact block to what the viewport hands the scene and
what the panel says.

The block is built with the worker's own packing (`sim_worker.contact_frame_dict` over
`Simulator.contact_frame`'s columns), so a change of format shows up here. No GL: the viewport draws
into a recording stand-in for `gl_scene.Scene`; the GL half is `test_console_gl_lifecycle.py`'s. The
Qt widgets need a platform: run with ``QT_QPA_PLATFORM=minimal`` (`offscreen` aborts the interpreter
on this machine).
"""
from __future__ import annotations

import math
import os
import types

import numpy as np
import pytest

from f1sim.viewer.console import contacts as CT
from f1sim.viewer.console import frames as frames_mod
from f1sim.viewer.console.frames import FrameBuffer

DT = 0.025                       # the control step the worker sends frames at
WALL, PROP, CAR = CT.KIND_WALL, CT.KIND_PROP, CT.KIND_CAR


def block(n, held=None, onsets=None, hard=None, now=None):
    """A frame's contact block for `n` cars, packed the way the worker packs it.

    held:   {car: (kind, (x, y), severity, seconds left)}
    onsets / hard: {car: (wall, prop, car)}
    now:    {car: (kind, (x, y), severity)} -- this step's contact
    """
    from f1sim.sim import CONTACT_FRAME_COLUMNS
    from f1sim.viewer.sim_worker import contact_frame_dict
    col = {k: j for j, k in enumerate(CONTACT_FRAME_COLUMNS)}
    b = np.zeros((n, len(CONTACT_FRAME_COLUMNS)), np.float32)
    for i, (kind, (x, y), sev, left) in (held or {}).items():
        b[i, [col["held_s"], col["held_kind"], col["held_x"], col["held_y"], col["held_severity"]]] = \
            [left, kind, x, y, sev]
    for i, (kind, (x, y), sev) in (now or {}).items():
        b[i, [col["kind"], col["x"], col["y"], col["severity"]]] = [kind, x, y, sev]
    for i, v in (onsets or {}).items():
        b[i, [col["onsets_wall"], col["onsets_prop"], col["onsets_car"]]] = v
    for i, v in (hard or {}).items():
        b[i, [col["hard_wall"], col["hard_prop"], col["hard_car"]]] = v
    return contact_frame_dict(b)


def a_frame(n=3, t=1.0, seq=0, gen=0, focus=0, contact=None, **extra):
    fr = {"gen": gen, "seq": seq, "t": t, "n": n, "focus": focus, "ids": np.arange(n, dtype=np.int32),
          "x": np.linspace(2.0, 6.0, n).astype(np.float32), "y": np.full(n, 3.0, np.float32),
          "yaw": np.zeros(n, np.float32), "vx": np.full(n, 0.5, np.float32),
          "steer": np.zeros(n, np.float32), "roll": np.zeros(n, np.float32),
          "pitch": np.zeros(n, np.float32), "lap": np.zeros(n, np.float32),
          "coll": np.zeros(n, np.float32), "s": np.zeros(n, np.float32),
          "wall": np.full(n, 0.4, np.float32), "rear": np.tile(np.float32([0.05, 0.2, 0.0, 0.1]), (n, 1)),
          "len": np.full(n, 0.5, np.float32), "control_dt": DT, "sim_rate": 1.0}
    if contact is not None:
        fr["contact"] = contact
    fr.update(extra)
    return fr


def col(rows, name):
    return rows[:, CT.MARK_COLUMNS.index(name)]


# ================================================================ the markers
def test_each_kind_is_drawn_in_its_own_colour():
    """Orange a wall or duct hose, yellow a crate, red another car -- the theme's tokens, three apart."""
    fr = a_frame(contact=block(3, held={0: (WALL, (1.0, 2.0), 1.6, 0.3), 1: (PROP, (4.0, 2.0), 1.6, 0.3),
                                        2: (CAR, (7.0, 2.0), 1.6, 0.3)}))
    rows = CT.contact_marks(fr)
    assert rows.shape == (3, len(CT.MARK_COLUMNS))
    by_x = {float(r[0]): r[3:6] for r in rows}
    np.testing.assert_allclose(by_x[1.0], CT.KIND_RGB[WALL])
    np.testing.assert_allclose(by_x[4.0], CT.KIND_RGB[PROP])
    np.testing.assert_allclose(by_x[7.0], CT.KIND_RGB[CAR])
    rgb = CT.KIND_RGB[1:]
    assert min(np.abs(rgb[i] - rgb[j]).max() for i in range(3) for j in range(i + 1, 3)) > 0.25
    # where it touched, at bumper height
    assert sorted(map(tuple, rows[:, :2].tolist())) == [(1.0, 2.0), (4.0, 2.0), (7.0, 2.0)]
    assert np.allclose(rows[:, 2], CT.MARK_Z)


def test_a_graze_is_small_and_faint_and_a_hard_hit_large_and_bright():
    sev = [0.05, 0.3, 0.6, 1.0, CT.MARK_FULL_MPS, 4.0]
    fr = a_frame(n=len(sev), contact=block(len(sev), held={i: (WALL, (float(i), 0.0), s, 0.3)
                                                           for i, s in enumerate(sev)}))
    rows = CT.contact_marks(fr)
    rows = rows[np.argsort(rows[:, 0])]
    for name, ratio in (("diameter_m", 2.0), ("min_px", 2.0), ("a", 1.4)):
        v = col(rows, name)
        assert np.all(np.diff(v) >= -1e-6), (name, v)              # never smaller for a harder hit
        assert v[-1] > ratio * v[0], (name, v)                        # a hit is clearly not a graze
        assert v[-1] == pytest.approx(v[-2]), (name, v)               # full size from MARK_FULL_MPS on
    assert col(rows, "a")[0] < 0.7 and col(rows, "a")[-1] == pytest.approx(1.0)
    # "hard" is the simulator's threshold (what the panel counts), not the display scale
    hard_mps = fr["contact"]["hard_mps"]
    np.testing.assert_array_equal(col(rows, "hard"), [float(s >= hard_mps) for s in sev])


def test_a_marker_fades_over_the_hold_and_is_gone_when_it_ends():
    alpha, phase = [], []
    for left in (0.3, 0.2, 0.1, 0.025):
        rows = CT.contact_marks(a_frame(contact=block(3, held={0: (WALL, (1.0, 2.0), 2.0, left)})))
        assert rows is not None and len(rows) == 1
        alpha.append(float(col(rows, "a")[0]))
        phase.append(float(col(rows, "phase")[0]))
    assert alpha == sorted(alpha, reverse=True) and alpha[0] == pytest.approx(1.0)
    assert alpha[-1] == pytest.approx(0.025 / 0.3, rel=1e-3)
    assert phase == sorted(phase) and phase[0] == pytest.approx(0.0)
    # the simulator keeps the held record after the hold runs out; nothing is drawn from it
    assert CT.contact_marks(a_frame(contact=block(3, held={0: (WALL, (1.0, 2.0), 2.0, 0.0)}))) is None


def test_no_block_no_hold_or_a_malformed_block_draws_nothing_and_never_raises():
    assert CT.contact_marks(a_frame()) is None                                   # an older worker
    assert CT.contact_marks(a_frame(contact=block(3))) is None                   # nobody touching
    bad = block(3, held={0: (WALL, (1.0, 2.0), 2.0, 0.3)})
    bad["held_s"] = bad["held_s"][:2]                                           # one car short
    assert CT.contact_marks(a_frame(contact=bad)) is None
    assert CT.car_highlight(a_frame(contact=bad)) is None
    assert CT.focus_summary(a_frame(contact=bad)) is None
    assert CT.contact_marks(a_frame(contact={"held_s": "nonsense"})) is None
    nan = block(3, held={0: (WALL, (float("nan"), 2.0), 2.0, 0.3)})
    assert CT.contact_marks(a_frame(contact=nan)) is None


def test_the_two_cars_of_a_car_contact_draw_one_marker():
    """Both cars hold the same contact at the same point: one marker, not one twice as bright."""
    q = (5.0, 3.1)
    fr = a_frame(n=4, contact=block(4, held={1: (CAR, q, 0.8, 0.3), 2: (CAR, q, 0.8, 0.3),
                                             0: (WALL, q, 1.2, 0.3), 3: (WALL, (9.0, 1.0), 0.4, 0.2)}))
    rows = CT.contact_marks(fr)
    kinds = [int(np.argmin(np.abs(CT.KIND_RGB - r[3:6]).sum(1))) for r in rows]
    assert sorted(kinds) == [WALL, WALL, CAR]                    # the wall at q is a contact of its own
    assert sum(1 for r, k in zip(rows, kinds) if k == CAR) == 1


# ================================================================ playback
def test_interpolated_playback_keeps_a_marker_where_it_was_once_and_fades_it_smoothly(monkeypatch):
    """Frames every 25 ms, drawn every 5 ms through `FrameBuffer`'s interpolation.

    Car 0 hits a hose at P (1.6 m/s); cars 1 and 2 touch each other at Q (0.8 m/s). In every drawn
    frame a marker is exactly at P or at Q -- never between them, never twice -- and P's fades out over
    the hold without a step at the frame boundaries (the block is the newer frame's, the fade runs to
    the drawn time through ``t_frame``).
    """
    clock = [100.0]
    monkeypatch.setattr(frames_mod, "time", types.SimpleNamespace(monotonic=lambda: clock[0]))
    P, Q = (3.5, 2.75), (7.25, 3.0)
    k_p, k_q = 4, 10

    def frame_at(j):
        held, now = {}, {}
        if j >= k_p:
            held[0] = (WALL, P, 1.6, max(0.0, 0.3 - DT * (j - k_p)))
        if j >= k_q:
            held[1] = held[2] = (CAR, Q, 0.8, max(0.0, 0.3 - DT * (j - k_q)))
        if j == k_p:
            now[0] = (WALL, P, 1.6)                         # the touch on the step of the hit is the hit
        if j == k_q:
            now[1] = now[2] = (CAR, Q, 0.8)
        return a_frame(t=DT * j, seq=j, contact=block(3, held=held, now=now))

    feed = [frame_at(j) for j in range(40)]
    stamp = CT.ImpactClock()                                # as SessionController does on arrival
    b = FrameBuffer(capacity=4)
    b.set_generation(0)
    nxt, draws = 0, []
    for _ in range(int(40 * DT / 0.005) + 20):
        clock[0] += 0.005
        while nxt < len(feed) and feed[nxt]["t"] <= clock[0] - 100.0 + 1e-9:
            stamp.stamp(feed[nxt])
            assert b.push(feed[nxt])
            nxt += 1
        out = b.frame_to_draw(interpolate=True)
        if out is None or b.last_mode == "raw:clock_start":
            continue
        draws.append((b.last_mode, float(out["t"]), CT.contact_marks(out)))
    assert any(mode == "interp" for mode, _, _ in draws)
    a_p = []
    for mode, t, rows in draws:
        pts = [] if rows is None else [tuple(map(float, r[:2])) for r in rows]
        assert set(pts) <= {P, Q}, (mode, t, pts)                       # never between two contacts
        assert pts.count(P) <= 1 and pts.count(Q) <= 1, (mode, t, pts)  # never twice
        a_p.append(next((float(r[6]) for r in rows if tuple(map(float, r[:2])) == P), 0.0)
                   if rows is not None else 0.0)
    a_p = np.asarray(a_p)
    on = np.flatnonzero(a_p > 0)
    assert len(on) and np.all(np.diff(on) == 1), "P's marker came and went more than once"
    seq = a_p[on[0]:on[-1] + 2]
    assert np.all(np.diff(seq) <= 1e-6), seq                  # it only fades
    assert np.max(-np.diff(seq)) < 0.1, seq                   # and smoothly: no step at a frame boundary
    shown = (on[-1] - on[0] + 1) * 0.005
    assert 0.3 - 2 * DT <= shown <= 0.3 + 2 * DT, shown       # for the hold, give or take a frame


def test_a_hit_fades_from_when_it_happened_and_a_long_push_is_drawn_where_it_touches():
    """A 2 m/s hit at P, then the car keeps pushing for a second while the contact slides away from P.

    Every contact step restarts the hold and the hit stays the hold's record, so a hit timed by the hold
    looked fresh -- and stood at P -- for as long as the car touched. Timed from when it happened
    (`ImpactClock`), it is gone after `hold_s`; the touch is drawn where the car touches now.
    """
    P = (3.0, 2.0)
    stamp = CT.ImpactClock()
    frames = []
    for j in range(60):
        held, now = {}, {}
        if j >= 4:
            pushing = j < 44
            left = 0.3 if pushing else max(0.0, 0.3 - DT * (j - 43))
            held[0] = (WALL, P, 2.0, left)
            if pushing:
                now[0] = (WALL, (P[0] + 0.05 * (j - 4), P[1]), 2.0 if j == 4 else 0.0)
        fr = a_frame(t=DT * j, seq=j, contact=block(3, held=held, now=now))
        stamp.stamp(fr)
        frames.append(fr)
    assert frames[30]["contact"]["impact_t"][0] == pytest.approx(DT * 4, abs=1e-6)   # the hit's own time

    def marks(j):
        """{point rounded to the mm: (alpha, diameter)} of the markers drawn for frame j."""
        rows = CT.contact_marks(frames[j])
        return {} if rows is None else {(round(float(r[0]), 3), round(float(r[1]), 3)): (float(r[6]), float(r[7]))
                                        for r in rows}

    assert list(marks(4)) == [P]                                               # the hit, once
    at10 = marks(10)
    assert at10[P][0] == pytest.approx(1.0 - 6 * DT / 0.3, abs=1e-3)           # fading from the hit
    touch = (round(P[0] + 0.05 * 6, 3), P[1])
    assert touch in at10 and at10[touch][1] < at10[P][1]                       # and the push, smaller
    assert list(marks(30)) == [(round(P[0] + 0.05 * 26, 3), P[1])]            # the hit has gone; the push is where it is
    assert marks(50) == {}                                                     # let go: nothing on the track
    w = [float(CT.car_highlight(frames[j])[2][0]) for j in (4, 30, 46, 50)]
    assert w[0] == pytest.approx(1.0)                                          # a flash
    assert w[1] == pytest.approx(CT.HIGHLIGHT_MIN)                             # a steady glow while pushing
    assert w[1] > w[2] > w[3] > 0.0                                            # fading once it let go


def test_a_car_pressed_where_it_hit_never_loses_its_marker():
    """Hit at P, then pressed against P: the hit alone at first (the touch is the hit), the fading hit
    and the steady touch once the hit is the fainter, the touch alone after the hold -- never nothing."""
    P = (3.0, 2.0)
    stamp = CT.ImpactClock()
    counts, touch_alpha = [], CT.MARK_ALPHA_MIN
    for j in range(30):
        fr = a_frame(t=DT * j, seq=j, contact=block(3, held={0: (WALL, P, 2.0, 0.3)},
                                                    now={0: (WALL, P, 2.0 if j == 0 else 0.0)}))
        stamp.stamp(fr)
        rows = CT.contact_marks(fr)
        assert rows is not None and np.allclose(rows[:, :2], P)
        counts.append(len(rows))
        if j == 0:
            assert len(rows) == 1 and rows[0, 6] == pytest.approx(1.0)
        if j * DT >= 0.3 + DT:                            # (float32 holds: a step of slack)
            assert len(rows) == 1 and rows[0, 6] == pytest.approx(touch_alpha)
    assert 2 in counts                                     # both, while the hit fades through the touch


def test_the_impact_clock_keeps_a_records_time_and_starts_over_with_a_new_one_or_session():
    clock = CT.ImpactClock()

    def stamped(t, held, gen=0):
        fr = a_frame(t=t, gen=gen, contact=block(3, held=held))
        clock.stamp(fr)
        return fr["contact"]["impact_t"]

    t0 = stamped(1.0, {0: (WALL, (1.0, 1.0), 1.2, 0.3)})
    assert t0[0] == pytest.approx(1.0) and np.isnan(t0[1:]).all()
    assert stamped(1.1, {0: (WALL, (1.0, 1.0), 1.2, 0.3)})[0] == pytest.approx(1.0)      # unchanged: kept
    assert stamped(1.2, {0: (WALL, (1.0, 1.0), 1.2, 0.2)})[0] == pytest.approx(1.0)
    # a harder contact takes the record: its own time, the hold's latest contact step
    assert stamped(1.3, {0: (CAR, (2.0, 1.0), 2.5, 0.25)})[0] == pytest.approx(1.25)
    assert np.isnan(stamped(1.4, {})[0])                                                 # nothing held
    # another session: nothing carries over, even an identical record
    stamped(2.0, {0: (WALL, (1.0, 1.0), 1.2, 0.3)})
    assert stamped(2.5, {0: (WALL, (1.0, 1.0), 1.2, 0.3)}, gen=1)[0] == pytest.approx(2.5)
    clock.stamp(a_frame())                                                               # no block: no-op


def test_a_frame_drawn_raw_reads_its_own_hold():
    fr = a_frame(t=2.0, contact=block(3, held={0: (WALL, (1.0, 2.0), 2.0, 0.15)}))
    assert CT.hold_left(fr)[0] == pytest.approx(0.15)
    fr["t_frame"], fr["t"] = 2.0, 1.99                  # drawn 10 ms before its frame: 10 ms more left
    assert CT.hold_left(fr)[0] == pytest.approx(0.16)
    fr["t"] = 1.5                                       # never more than the hold
    assert CT.hold_left(fr)[0] == pytest.approx(0.3)


# ================================================================ the car's own colour
def test_a_car_is_lit_in_the_colour_of_what_it_touched_fading_with_the_hold():
    fr = a_frame(contact=block(3, held={0: (WALL, (1.0, 2.0), 2.0, 0.3), 1: (PROP, (4.0, 2.0), 0.1, 0.3)}))
    hot, colour, w = CT.car_highlight(fr)
    np.testing.assert_allclose(colour[0], CT.KIND_RGB[WALL])
    np.testing.assert_allclose(colour[1], CT.KIND_RGB[PROP])
    assert w[0] == pytest.approx(1.0) and 0.5 < w[1] < w[0] and w[2] == 0.0      # a graze less, nothing none
    assert hot[0].max() > 1.0                                                     # a flash, over-bright
    later = a_frame(contact=block(3, held={0: (WALL, (1.0, 2.0), 2.0, 0.1)}))
    assert CT.car_highlight(later)[2][0] == pytest.approx(0.1 / 0.3)


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets
    from f1sim.viewer.console import app as console_app
    return QtWidgets.QApplication.instance() or console_app.create_app(["test"])


class RecordingScene:
    """What `ViewportWidget._draw` calls on `gl_scene.Scene`, recorded instead of drawn."""

    def __init__(self, w=640, h=400):
        self.width, self.height = w, h
        eye = np.eye(4, dtype=np.float32)
        self.car_pivots = {k: eye for k in ("lidar", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")}
        self.calls = {}

    def __getattr__(self, name):
        if name.startswith("set_") or name in ("push_trail_points", "draw_frame"):
            def record(*a, **kw):
                self.calls[name] = (a, kw)
            return record
        raise AttributeError(name)


@pytest.fixture
def viewport(qapp):
    from f1sim.viewer.console.viewport import ViewportWidget
    vp = ViewportWidget()
    vp.scene = RecordingScene()
    vp.camera = "top"                          # needs no map
    yield vp
    vp.scene = None
    vp.close()


def test_the_viewport_hands_the_scene_the_markers_and_lights_the_cars(viewport):
    fr = a_frame(n=3, focus=0, contact=block(3, held={0: (WALL, (1.9, 2.8), 2.0, 0.3),
                                                      2: (CAR, (6.1, 3.0), 0.5, 0.2)}))
    viewport._draw(fr)
    calls = viewport.scene.calls
    np.testing.assert_array_equal(calls["set_contact_marks"][0][0], CT.contact_marks(fr))
    tint = calls["set_car_instances"][0][1]
    orange, red = CT.KIND_RGB[WALL], CT.KIND_RGB[CAR]
    assert tint[0, 0] > 1.0 and tint[0, 0] > tint[0, 1] > tint[0, 2]               # the watched car, flashing orange
    np.testing.assert_allclose(tint[1], viewport.TINT_OTHER)                        # untouched: as before
    assert tint[2, 0] > tint[2, 1] and tint[2, 0] > viewport.TINT_OTHER[0]           # the other one, red
    ring = calls["set_focus_ring"][1]["colour"]
    assert ring == pytest.approx(viewport.RING_FOCUS)                                # the watched car stays cyan
    (xy, cols), kw = calls["set_contact_rings"]
    np.testing.assert_allclose(xy, [[fr["x"][0], fr["y"][0]], [fr["x"][2], fr["y"][2]]])   # a ring per contact
    np.testing.assert_allclose(cols[:, :3], [orange, red])
    assert kw["radius"] > 0.34                                                       # outside the watched car's
    # the markers are an overlay; the car's own colour is not
    viewport.show_contacts = False
    viewport._draw(fr)
    assert viewport.scene.calls["set_contact_marks"][0][0] is None
    assert viewport.scene.calls["set_car_instances"][0][1][0, 0] > 1.0


def test_a_frame_without_a_contact_block_still_shows_the_collision_flag(viewport):
    """An older worker's frame: the one-frame red flag it always had."""
    fr = a_frame(n=2, focus=0, coll=np.array([1.0, 0.0], np.float32))
    viewport._draw(fr)
    calls = viewport.scene.calls
    np.testing.assert_allclose(calls["set_car_instances"][0][1][0], viewport.TINT_COLLIDED)
    assert calls["set_focus_ring"][1]["colour"] == pytest.approx(viewport.RING_COLLIDED)
    assert calls["set_contact_marks"][0][0] is None
    assert calls["set_contact_rings"][0] == (None, None)


# ================================================================ the panel
def test_the_summary_tells_grazes_only_from_hard_hits():
    s = CT.focus_summary(a_frame(contact=block(3, onsets={0: (3, 1, 0)})))
    assert (s.total, s.total_hard, s.level) == (4, 0, "graze")
    s = CT.focus_summary(a_frame(contact=block(3, onsets={0: (3, 1, 2)}, hard={0: (1, 0, 1)})))
    assert (s.total, s.total_hard, s.level) == (6, 2, "hard")
    assert CT.tile_text(s) == "6 · 세게 2"
    html = CT.breakdown_html(s)
    for name, n in (("벽", 3), ("장애물", 1), ("차", 2)):
        assert f"{name}</span> {n}" in html
    assert html.count("세게") == 2                                # the wall's hard one and the car's
    s = CT.focus_summary(a_frame(contact=block(3)))
    assert (s.total, s.level, CT.tile_text(s)) == (0, "none", "0")
    assert CT.hard_label(s.hard_mps) == f"세게 ≥{s.hard_mps:.1f} m/s"


def test_the_last_contact_outlives_its_marker_and_forgets_on_reset_or_another_car():
    last = CT.LastContact()
    hit = a_frame(t=5.0, contact=block(3, held={0: (WALL, (1.0, 2.0), 1.7, 0.3)}, onsets={0: (1, 0, 0)},
                                       hard={0: (1, 0, 0)}))
    last.update((1, 0), CT.focus_summary(hit))
    assert last.text(5.0).endswith("지금")                          # while it is held
    later = a_frame(t=6.0, contact=block(3, held={0: (WALL, (1.0, 2.0), 1.7, 0.0)}, onsets={0: (1, 0, 0)},
                                         hard={0: (1, 0, 0)}))
    last.update((1, 0), CT.focus_summary(later))
    text = last.text(6.0)
    assert "벽" in text and "1.70 m/s" in text and "세게" in text and "1.0초 전" in text
    reset = a_frame(t=6.5, contact=block(3))                       # the episode's count went back to 0
    last.update((1, 0), CT.focus_summary(reset))
    assert last.text(6.5) == "—"
    last.update((1, 0), CT.focus_summary(hit))
    last.update((1, 2), CT.focus_summary(a_frame(t=5.0, focus=2, contact=hit["contact"])))
    assert last.text(5.0) == "—"                                   # another car: its own last contact


@pytest.fixture(scope="module")
def window(qapp):
    from f1sim.viewer.console.window import ConsoleWindow
    w = ConsoleWindow()
    yield w
    w.close()
    w.deleteLater()


def _telemetry(window, fr):
    from f1sim.viewer.console.frames import Freshness
    window.update_telemetry(fr, Freshness(0.0, 0.0, 0.0, 0, True), {}, 1.0, 60.0, None)


def test_the_contact_tile_counts_the_watched_cars_episode(window):
    from f1sim.viewer.console.theme import C
    fr = a_frame(t=3.0, focus=1, contact=block(3, held={1: (CAR, (4.0, 3.0), 1.3, 0.2)},
                                               onsets={0: (9, 9, 9), 1: (2, 0, 1)}, hard={1: (0, 0, 1)}))
    _telemetry(window, fr)
    assert window.m_coll.value.text() == "3 · 세게 1"
    assert window.m_coll._colour == C["danger"]
    assert "세게 ≥1.0 m/s" in window.m_coll.name_label.text()      # the threshold, where the count is
    kinds = window.drive_info.value_label("접촉 종류").text()
    assert "벽</span> 2" in kinds and "장애물</span> 0" in kinds and "차</span> 1" in kinds
    assert "1.30 m/s" in window.drive_info.value_label("최근 접촉").text()
    # grazes only: amber, not red
    _telemetry(window, a_frame(t=4.0, focus=0, contact=block(3, onsets={0: (4, 0, 0)})))
    assert (window.m_coll.value.text(), window.m_coll._colour) == ("4 · 세게 0", C["warn"])
    _telemetry(window, a_frame(t=4.0, focus=2, contact=block(3)))
    assert (window.m_coll.value.text(), window.m_coll._colour) == ("0", None)
    # an older worker: the old count of cars flagged in this frame
    _telemetry(window, a_frame(t=4.0, coll=np.array([1.0, 0.0, 1.0], np.float32)))
    assert window.m_coll.value.text() == "2 / 3" and window.m_coll.name_label.text() == "충돌"
