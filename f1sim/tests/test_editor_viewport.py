"""The environment editor's viewport, on a real GL context.

What is being checked is the contract the editor page builds on: the free camera frames what it
is given, a point projected on screen unprojects to where it started, mouse input turns into
world-space signals, camera gestures move the camera and nothing else, and layer visibility
changes what is drawn. These need real matrices and a real framebuffer, so they run under the
same software-GL contract as `test_console_gl_lifecycle.py` (Xvfb + llvmpipe) and skip, rather
than fail, where no GL context can be made.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytest.importorskip("PyQt5")
pytest.importorskip("moderngl")

from PyQt5 import QtCore, QtGui, QtWidgets                              # noqa: E402

from f1sim.viewer import gl_scene as G                                  # noqa: E402
from f1sim.viewer.console import app as console_app                     # noqa: E402
from f1sim.viewer.console.editor_viewport import EditorViewport, point_in_polygon   # noqa: E402
from f1sim.viewer.console.viewport import TrackGeometry                 # noqa: E402
from console_fake_worker import _geometry                                # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        pytest.skip("no display")
    existing = QtWidgets.QApplication.instance()
    yield existing or console_app.create_app(["test"])


@pytest.fixture
def widget(qapp):
    w = EditorViewport()
    w.resize(640, 400)
    w.show()
    for _ in range(40):
        qapp.processEvents()
        if w.ctx is not None or w._gl_error:
            break
    if w.ctx is None:
        w.close()
        pytest.skip(f"no usable GL context: {w._gl_error}")
    # A paint that throws is caught by `paintGL` and reported through `gl_failed` while the frame
    # is cleared, so a broken overlay path would otherwise pass a "pixels changed" check with a
    # blank picture. Every test fails on any paint error.
    errors = []
    w.gl_failed.connect(errors.append)
    yield w
    w.close()
    qapp.processEvents()
    assert not errors, f"paint raised: {errors}"


def fake_geometry(name="fake:oval"):
    return TrackGeometry(**_geometry(name))


def box_batch(x, y, size=(0.5, 0.4, 0.35), material="plastic"):
    v, n, idx = G._unit_cube()
    pos = v * np.asarray(size, np.float32) + np.array([x, y, 0.5 * size[2]], np.float32)
    col = np.tile([0.72, 0.55, 0.32, 0.0], (len(v), 1)).astype(np.float32)
    return {"material": material, "pos": pos.astype(np.float32), "nrm": n, "col": col, "idx": idx,
            "n_props": 1, "n_tris": len(idx) // 3}


def rich_geometry():
    """The fake oval plus a wall ring, a pillar and one prop, so every layer has a mesh."""
    d = _geometry("fake:rich")
    x0, y0, x1, y1 = d["bounds"]
    ring = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    pillar = np.array([[1.0, 1.0], [1.8, 1.0], [1.8, 1.8], [1.0, 1.8]], np.float64)
    d["walls"] = G.wall_mesh_arrays([ring, pillar], height=1.0)
    d["props"] = (box_batch(6.0, 8.0),)
    return TrackGeometry(**d)


def paint(widget, qapp, times=2):
    for _ in range(times):
        widget.update()
        qapp.processEvents()


def grab(widget):
    img = widget.grabFramebuffer().convertToFormat(QtGui.QImage.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(img.byteCount())
    return np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())[:, :w * 3].reshape(h, w, 3).copy()


def press(widget, px, py, button=QtCore.Qt.LeftButton, mods=QtCore.Qt.NoModifier):
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress, QtCore.QPointF(px, py), button, button, mods)
    QtWidgets.QApplication.sendEvent(widget, ev)


def move(widget, px, py, buttons=QtCore.Qt.LeftButton, mods=QtCore.Qt.NoModifier):
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(px, py), QtCore.Qt.NoButton, buttons, mods)
    QtWidgets.QApplication.sendEvent(widget, ev)


def release(widget, px, py, button=QtCore.Qt.LeftButton, mods=QtCore.Qt.NoModifier):
    ev = QtGui.QMouseEvent(QtCore.QEvent.MouseButtonRelease, QtCore.QPointF(px, py), button, QtCore.Qt.NoButton, mods)
    QtWidgets.QApplication.sendEvent(widget, ev)


def wheel(widget, px, py, delta=120, mods=QtCore.Qt.NoModifier):
    ev = QtGui.QWheelEvent(QtCore.QPointF(px, py), QtCore.QPointF(px, py), QtCore.QPoint(0, 0),
                           QtCore.QPoint(0, delta), QtCore.Qt.NoButton, mods, QtCore.Qt.NoScrollPhase, False)
    QtWidgets.QApplication.sendEvent(widget, ev)


def key(widget, k, mods=QtCore.Qt.NoModifier):
    QtWidgets.QApplication.sendEvent(widget, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, k, mods))
    QtWidgets.QApplication.sendEvent(widget, QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, k, mods))


# ---------------------------------------------------------------- pure
def test_point_in_polygon_handles_convex_concave_and_edges():
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    assert point_in_polygon(1.0, 1.0, square)
    assert not point_in_polygon(2.5, 1.0, square)
    assert not point_in_polygon(1.0, -0.1, square)
    # a U shape: the notch is outside even though the bbox contains it
    u = np.array([[0, 0], [3, 0], [3, 3], [2, 3], [2, 1], [1, 1], [1, 3], [0, 3]], float)
    assert point_in_polygon(0.5, 2.0, u) and point_in_polygon(2.5, 2.0, u)
    assert not point_in_polygon(1.5, 2.0, u)
    assert not point_in_polygon(1.0, 1.0, np.zeros((2, 2)))


# ---------------------------------------------------------------- construction and upload
def test_constructs_with_gl_and_editor_empty_text(widget, qapp):
    assert widget.ctx is not None and widget.scene is not None
    assert widget._car_loaded is False, "an editor has no cars to load"
    assert "환경" in widget._empty.text()
    paint(widget, qapp)
    assert widget._empty.isVisible(), "no geometry: the empty-state text shows"
    assert widget.last_static_drawn == 0
    assert widget.ctx.error == "GL_NO_ERROR"


def test_geometry_upload_tags_layers_and_frames_the_map(widget, qapp):
    uploads = []
    widget.upload_timed.connect(lambda what, ms: uploads.append(what))
    geom = fake_geometry()
    widget.set_geometry(geom)
    paint(widget, qapp)
    assert widget.geometry_data is geom
    assert any(w.startswith("map:") for w in uploads)
    assert sorted(getattr(m, "layer", "?") for m in widget.scene.static) == ["ducts", "floor"]
    assert widget.last_static_drawn == 2
    assert not widget._empty.isVisible()
    x0, y0, x1, y1 = geom.bounds
    assert widget.pivot == pytest.approx([(x0 + x1) / 2, (y0 + y1) / 2])
    assert widget.ctx.error == "GL_NO_ERROR"


def test_re_uploading_geometry_leaves_the_camera_alone(widget, qapp):
    widget.set_geometry(fake_geometry())
    paint(widget, qapp)
    widget.look_at_point(2.0, 2.0)
    widget.dist = 3.0
    widget.set_geometry(fake_geometry("fake:edited"))   # every edit re-sends geometry
    paint(widget, qapp)
    assert widget.pivot == pytest.approx([2.0, 2.0]) and widget.dist == 3.0
    widget.set_geometry(None)
    paint(widget, qapp)
    assert widget.geometry_data is None and widget._empty.isVisible()
    widget.set_geometry(fake_geometry())                # a new scene after a blank: framed again
    paint(widget, qapp)
    assert widget.pivot == pytest.approx([6.0, 4.5])


# ---------------------------------------------------------------- camera
@pytest.mark.parametrize("top,ortho", [(False, False), (True, True), (True, False)])
def test_frame_all_puts_every_corner_on_screen(widget, qapp, top, ortho):
    geom = fake_geometry()
    widget.set_geometry(geom)
    paint(widget, qapp)
    widget.ortho_top = ortho
    widget.set_top_view(top)
    widget.az, widget.el = math.radians(-60.0), math.radians(40.0)
    widget.frame_all()
    x0, y0, x1, y1 = geom.bounds
    W, H = widget.width(), widget.height()
    worst = 0.0
    for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        p = widget.project(cx, cy)
        assert p is not None
        assert 0 <= p[0] <= W and 0 <= p[1] <= H, f"corner ({cx},{cy}) at {p} is off screen"
        worst = max(worst, abs(2 * p[0] / W - 1), abs(2 * p[1] / H - 1))
    assert worst > 0.6, f"the map only reaches {worst:.2f} of the half-extent: framed too far away"


@pytest.mark.parametrize("pose", [
    dict(top=False, ortho=False, az=-90.0, el=55.0, dist=12.0),
    dict(top=False, ortho=False, az=30.0, el=20.0, dist=6.0),
    dict(top=False, ortho=False, az=200.0, el=80.0, dist=25.0),
    dict(top=True, ortho=True, az=-90.0, el=55.0, dist=9.0),
    dict(top=True, ortho=False, az=45.0, el=55.0, dist=9.0),
])
def test_projection_round_trip_within_a_centimetre(widget, qapp, pose):
    widget.set_geometry(fake_geometry())
    paint(widget, qapp)
    widget.ortho_top = pose["ortho"]
    widget.set_top_view(pose["top"])
    widget.az, widget.el, widget.dist = math.radians(pose["az"]), math.radians(pose["el"]), pose["dist"]
    widget.pivot = np.array([6.0, 4.5])
    for x, y in ((6.0, 4.5), (2.0, 1.0), (10.5, 7.8), (4.2, 6.6)):
        p = widget.project(x, y)
        assert p is not None
        g = widget.ground_point(*p)
        assert g is not None
        assert math.hypot(g[0] - x, g[1] - y) < 0.01, f"{pose}: ({x},{y}) -> {p} -> {g}"


def test_ground_point_misses_the_sky(widget, qapp):
    widget.set_geometry(fake_geometry())
    paint(widget, qapp)
    widget.set_top_view(False)
    widget.el, widget.dist = math.radians(5.0), 8.0
    assert widget.ground_point(widget.width() / 2, 0) is None, "a pixel at the top of a near-horizontal view is sky"
    assert widget.ground_point(widget.width() / 2, widget.height() - 1) is not None


def test_top_view_toggle_restores_the_perspective(widget, qapp):
    widget.set_geometry(fake_geometry())
    paint(widget, qapp)
    widget.az, widget.el = math.radians(33.0), math.radians(41.0)
    changes = []
    widget.camera_changed.connect(lambda: changes.append(1))
    widget.set_top_view(True)
    assert widget.top and widget.camera_state()["ortho"] is True
    _, _, eye, target, _ = widget._matrices()
    assert eye[2] > 0.99 * widget.dist, "top view looks straight down"
    widget.set_top_view(False)
    assert not widget.top
    assert (widget.az, widget.el) == pytest.approx((math.radians(33.0), math.radians(41.0)))
    assert len(changes) == 2


# ---------------------------------------------------------------- picking
def test_hit_test_on_squares_last_wins(widget):
    sq = lambda cx, cy, h: np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]])
    widget.set_pickables([("a", sq(2, 2, 1)), ("b", sq(3, 2, 1)), ("tiny", np.zeros((2, 2)))])
    assert widget.hit_test(1.5, 2.0) == "a"
    assert widget.hit_test(3.5, 2.0) == "b"
    assert widget.hit_test(2.5, 2.0) == "b", "overlap: the later pickable wins"
    assert widget.hit_test(5.0, 5.0) == ""
    widget.set_pickables([])
    assert widget.hit_test(2.0, 2.0) == ""


# ---------------------------------------------------------------- mouse -> world signals
def test_press_move_release_emit_world_coordinates_and_hit(widget, qapp):
    widget.set_geometry(fake_geometry())
    paint(widget, qapp)
    widget.set_top_view(True)
    got = {"press": [], "move": [], "release": [], "hover": []}
    widget.ground_pressed.connect(lambda x, y, b, m, hit: got["press"].append((x, y, b, m, hit)))
    widget.ground_moved.connect(lambda x, y, b, m: got["move"].append((x, y, b, m)))
    widget.ground_released.connect(lambda x, y, b, m: got["release"].append((x, y, b, m)))
    widget.hovered.connect(lambda x, y, hit: got["hover"].append((x, y, hit)))
    widget.set_pickables([("box", np.array([[5.5, 4.0], [6.5, 4.0], [6.5, 5.0], [5.5, 5.0]]))])

    p0 = widget.project(6.0, 4.5)          # inside the box
    p1 = widget.project(8.0, 6.0)
    press(widget, *p0)
    move(widget, *p1)
    release(widget, *p1)
    assert len(got["press"]) == 1 and len(got["move"]) == 1 and len(got["release"]) == 1
    x, y, b, m, hit = got["press"][0]
    assert (x, y) == pytest.approx((6.0, 4.5), abs=0.01) and b == int(QtCore.Qt.LeftButton) and hit == "box"
    assert got["move"][0][:2] == pytest.approx((8.0, 6.0), abs=0.01)
    assert got["move"][0][2] == int(QtCore.Qt.LeftButton)
    assert got["release"][0][:2] == pytest.approx((8.0, 6.0), abs=0.01)
    assert widget.pivot == pytest.approx([6.0, 4.5]), "an edit drag never moves the camera"

    move(widget, *p0, buttons=QtCore.Qt.NoButton)
    assert got["hover"] and got["hover"][-1][2] == "box"
    assert got["hover"][-1][:2] == pytest.approx((6.0, 4.5), abs=0.01)
    assert len(got["move"]) == 1, "a hover is not a ground_moved"


def test_double_click_emits_once_without_a_second_press(widget, qapp):
    widget.set_geometry(fake_geometry())
    paint(widget, qapp)
    widget.set_top_view(True)
    presses, dbl, releases = [], [], []
    widget.ground_pressed.connect(lambda *a: presses.append(a))
    widget.double_clicked.connect(lambda x, y: dbl.append((x, y)))
    widget.ground_released.connect(lambda *a: releases.append(a))
    p = widget.project(7.0, 5.0)
    # Qt's sequence: press, release, DoubleClick (in place of the second press), release
    press(widget, *p)
    release(widget, *p)
    QtWidgets.QApplication.sendEvent(widget, QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonDblClick, QtCore.QPointF(*p), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier))
    release(widget, *p)
    assert len(presses) == 1 and len(releases) == 1
    assert len(dbl) == 1 and dbl[0] == pytest.approx((7.0, 5.0), abs=0.01)


def test_orbit_and_pan_move_the_camera_not_the_document(widget, qapp):
    widget.set_geometry(fake_geometry())
    paint(widget, qapp)
    widget.set_top_view(False)
    widget.az, widget.el = math.radians(-90.0), math.radians(50.0)
    edits, changes = [], []
    for sig in (widget.ground_pressed, widget.ground_moved, widget.ground_released):
        sig.connect(lambda *a: edits.append(a))
    widget.camera_changed.connect(lambda: changes.append(1))
    az0, pivot0 = widget.az, widget.pivot.copy()
    # middle-drag orbits
    press(widget, 300, 200, button=QtCore.Qt.MiddleButton)
    move(widget, 340, 190, buttons=QtCore.Qt.MiddleButton)
    release(widget, 340, 190, button=QtCore.Qt.MiddleButton)
    assert widget.az != az0 and np.allclose(widget.pivot, pivot0)
    assert widget.el < math.radians(50.0), "dragging up tilts toward the ground"
    # right-drag pans: the ground point under the cursor stays under the cursor
    g_before = widget.ground_point(300, 220)
    press(widget, 300, 220, button=QtCore.Qt.RightButton)
    move(widget, 360, 250, buttons=QtCore.Qt.RightButton)
    g_after = widget.ground_point(360, 250)
    release(widget, 360, 250, button=QtCore.Qt.RightButton)
    assert not np.allclose(widget.pivot, pivot0)
    assert g_after == pytest.approx(g_before, abs=0.01)
    # Shift+left also pans, Alt+left orbits
    az1, pivot1 = widget.az, widget.pivot.copy()
    press(widget, 300, 200, mods=QtCore.Qt.ShiftModifier)
    move(widget, 320, 200, mods=QtCore.Qt.ShiftModifier)
    release(widget, 320, 200, mods=QtCore.Qt.ShiftModifier)
    assert not np.allclose(widget.pivot, pivot1) and widget.az == az1
    press(widget, 300, 200, mods=QtCore.Qt.AltModifier)
    move(widget, 330, 200, buttons=QtCore.Qt.LeftButton, mods=QtCore.Qt.AltModifier)
    release(widget, 330, 200, mods=QtCore.Qt.AltModifier)
    assert widget.az != az1
    assert not edits, "camera gestures never reach the page as edits"
    assert len(changes) >= 4


def test_wheel_zooms_toward_the_cursor_and_ctrl_wheel_is_handed_to_the_page(widget, qapp):
    widget.set_geometry(fake_geometry())
    paint(widget, qapp)
    widget.set_top_view(True)
    edits = []
    widget.wheel_edit.connect(lambda d, m: edits.append((d, m)))
    d0 = widget.dist
    anchor = widget.ground_point(500, 100)
    wheel(widget, 500, 100, delta=240)
    assert widget.dist < d0
    assert widget.ground_point(500, 100) == pytest.approx(anchor, abs=0.01), "zoom keeps the cursor's ground point"
    assert not edits
    d1 = widget.dist
    wheel(widget, 500, 100, delta=120, mods=QtCore.Qt.ControlModifier)
    wheel(widget, 500, 100, delta=-120, mods=QtCore.Qt.AltModifier)
    assert widget.dist == d1, "Ctrl/Alt+wheel is the page's, not the camera's"
    assert edits == [(120, int(QtCore.Qt.ControlModifier)), (-120, int(QtCore.Qt.AltModifier))]


def test_keys_frame_toggle_top_and_forward_the_rest(widget, qapp):
    widget.set_geometry(fake_geometry())
    paint(widget, qapp)
    keys = []
    widget.key_pressed.connect(lambda k, m: keys.append((k, m)))
    widget.look_at_point(0.0, 0.0)
    key(widget, QtCore.Qt.Key_F)
    assert widget.pivot == pytest.approx([6.0, 4.5])
    widget.look_at_point(0.0, 0.0)
    key(widget, QtCore.Qt.Key_Home)
    assert widget.pivot == pytest.approx([6.0, 4.5])
    key(widget, QtCore.Qt.Key_5)
    assert widget.top
    key(widget, QtCore.Qt.Key_5)
    assert not widget.top
    key(widget, QtCore.Qt.Key_B, QtCore.Qt.ControlModifier)
    key(widget, QtCore.Qt.Key_Escape)
    assert keys == [(int(QtCore.Qt.Key_B), int(QtCore.Qt.ControlModifier)), (int(QtCore.Qt.Key_Escape), 0)]
    # Space is a pan modifier here, not a command for the page
    QtWidgets.QApplication.sendEvent(widget, QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Space, QtCore.Qt.NoModifier))
    pivot0 = widget.pivot.copy()
    press(widget, 300, 200)
    move(widget, 330, 200)
    release(widget, 330, 200)
    QtWidgets.QApplication.sendEvent(widget, QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, QtCore.Qt.Key_Space, QtCore.Qt.NoModifier))
    assert not np.allclose(widget.pivot, pivot0)
    assert len(keys) == 2


# ---------------------------------------------------------------- drawing
def test_layer_visibility_changes_what_is_drawn(widget, qapp):
    widget.set_geometry(rich_geometry())
    paint(widget, qapp)
    assert sorted(getattr(m, "layer", "?") for m in widget.scene.static) == ["ducts", "floor", "props", "walls"]
    assert widget.last_static_drawn == 4
    full = grab(widget)
    widget.set_layer_visibility(ducts=False)
    paint(widget, qapp)
    assert widget.last_static_drawn == 3
    no_ducts = grab(widget)
    assert np.count_nonzero(np.any(full != no_ducts, axis=2)) > 500, "hiding the hoses changed no pixels"
    widget.set_layer_visibility(walls=False, props=False)
    paint(widget, qapp)
    assert widget.last_static_drawn == 2
    widget.set_layer_visibility(floor=False)
    paint(widget, qapp)
    assert widget.last_static_drawn == 3
    no_floor = grab(widget)
    assert np.count_nonzero(np.any(full != no_floor, axis=2)) > 500
    widget.set_layer_visibility()
    paint(widget, qapp)
    assert widget.last_static_drawn == 4
    assert np.array_equal(grab(widget), full), "restoring every layer restores the picture"
    assert widget.ctx.error == "GL_NO_ERROR"


def test_overlays_are_drawn_on_top_and_cleared(widget, qapp):
    widget.set_geometry(rich_geometry())
    paint(widget, qapp)
    widget.set_top_view(True)
    paint(widget, qapp)
    bare = grab(widget)
    box = np.array([[5.7, 7.75], [6.3, 7.75], [6.3, 8.25], [5.7, 8.25]])
    widget.set_cursor("brush", 3.0, 3.0, radius=0.5)
    widget.set_selection([box])
    widget.set_hover(box + 1.0)
    widget.set_preview(np.array([[2.0, 6.0], [4.0, 7.0], [5.0, 6.0]]), width=0.3)
    widget.set_gizmo(6.0, 8.0, yaw=0.7, radius=0.6)
    paint(widget, qapp)
    assert widget.ctx.error == "GL_NO_ERROR"
    with_ov = grab(widget)
    assert np.count_nonzero(np.any(bare != with_ov, axis=2)) > 200, "the overlays drew nothing"
    # the brush ring really is where it was asked for: pixels change on the ring, not at its centre
    cx, cy = widget.project(3.0, 3.0)
    rx, ry = widget.project(3.5, 3.0)
    ring_px = np.any(bare != with_ov, axis=2)[int(ry) - 2:int(ry) + 3, int(rx) - 2:int(rx) + 3]
    assert ring_px.any(), "no changed pixel where the brush ring should be"
    widget.set_cursor("none", 0, 0)
    widget.set_selection([])
    widget.set_hover(None)
    widget.set_preview(None, 0.0)
    widget.set_gizmo(0, 0, 0, on=False)
    paint(widget, qapp)
    assert np.array_equal(grab(widget), bare), "clearing every overlay restores the bare picture"


def test_screenshot_and_base_timing_signals_still_work(widget, qapp, tmp_path):
    draws = []
    widget.draw_timed.connect(draws.append)
    widget.set_geometry(rich_geometry())
    widget.set_gizmo(6.0, 8.0, 0.0)
    paint(widget, qapp)
    out = tmp_path / "editor.png"
    assert widget.grab_png(str(out)) and out.stat().st_size > 1000
    assert draws and all(d >= 0 for d in draws)
    widget.set_corner_text("피벗 6.0, 4.5")
    assert widget._corner.isVisible()
    assert widget.draw_percentiles() is not None
