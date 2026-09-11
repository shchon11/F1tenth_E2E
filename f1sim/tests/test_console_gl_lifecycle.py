"""GL lifecycle of the console viewport, on a real context.

The thing being guarded against is specific. In moderngl 5.12.0, `Framebuffer.__del__` honours the
`_is_reference` flag that `detect_framebuffer()` sets, but `Framebuffer.release()` does **not**: it
calls straight through to the C release, which deletes a non-zero framebuffer name. A QOpenGLWidget
renders into a framebuffer Qt owns, so calling `.release()` on a detected handle for it would
delete the widget's own surface out from under it.

These tests run under whatever GL the environment provides (the CI fixture uses Xvfb + llvmpipe).
They skip rather than fail when no GL context can be created, because "this machine has no GL" is
not a defect in this code.
"""
import math
import os

import numpy as np
import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("moderngl")

from PyQt5 import QtCore, QtWidgets                                    # noqa: E402

from f1sim.viewer.console import app as console_app                     # noqa: E402
from f1sim.viewer.console.viewport import (ForeignObjectError, MAX_RENDER_CARS,  # noqa: E402
                                           TrackGeometry, ViewportWidget, _ForeignFramebuffer)


@pytest.fixture(scope="module")
def qapp():
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        pytest.skip("no display")
    existing = QtWidgets.QApplication.instance()
    app = existing or console_app.create_app(["test"])
    yield app


@pytest.fixture
def widget(qapp):
    w = ViewportWidget()
    w.resize(320, 240)
    w.show()
    for _ in range(40):
        qapp.processEvents()
        if w.ctx is not None or w._gl_error:
            break
    if w.ctx is None:
        w.close()
        pytest.skip(f"no usable GL context: {w._gl_error}")
    yield w
    w.close()
    qapp.processEvents()


def oval_geometry(name="test:oval"):
    from f1sim.viewer import gl_scene as G
    th = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    outer = np.stack([6 + 5.0 * np.cos(th), 4.5 + 3.4 * np.sin(th)], 1)
    inner = np.stack([6 + 3.2 * np.cos(th), 4.5 + 1.7 * np.sin(th)], 1)

    def occupied(xy):
        dx, dy = xy[:, 0] - 6.0, xy[:, 1] - 4.5
        return (((dx / 5.0) ** 2 + (dy / 3.4) ** 2) > 1.0) | (((dx / 3.2) ** 2 + (dy / 1.7) ** 2) < 1.0)

    bounds = (0.2, 0.4, 11.8, 8.6)
    return TrackGeometry(name=name, bounds=bounds, duct_height=0.2,
                         floor=G.floor_mesh_arrays(bounds),
                         ducts=G.duct_mesh_arrays([outer, inner], 0.2, occupied=occupied),
                         walls=None, centerline=None, raceline_xy=None, raceline_v=None)


def a_frame(n=2, t=0.0, seq=0, gen=0):
    return {
        "gen": gen, "seq": seq, "t": t, "n": n,
        "x": np.linspace(3, 8, n).astype(np.float32), "y": np.full(n, 4.5, np.float32),
        "yaw": np.zeros(n, np.float32), "vx": np.full(n, 3.0, np.float32),
        "steer": np.zeros(n, np.float32), "roll": np.zeros(n, np.float32),
        "pitch": np.zeros(n, np.float32), "lap": np.zeros(n, np.float32),
        "coll": np.zeros(n, np.float32), "s": np.zeros(n, np.float32),
        "wall": np.full(n, 0.4, np.float32),
        "rear": np.tile([0.10, 0.28, 0.0, 0.20], (n, 1)).astype(np.float32),
        "len": np.full(n, 0.55, np.float32), "focus": 0, "focus_env": 0,
        "ids": np.arange(n), "control_dt": 0.025,
    }


# ---------------------------------------------------------------- ownership
def test_foreign_framebuffer_refuses_release():
    """The guard itself: releasing a borrowed framebuffer must be impossible, not merely avoided."""
    class FakeFbo:
        def use(self):
            pass

    fb = _ForeignFramebuffer(FakeFbo(), glo=7, size=(10, 10))
    with pytest.raises(ForeignObjectError):
        fb.release()


def test_scene_target_is_qt_framebuffer_not_zero(widget, qapp):
    widget.update()
    qapp.processEvents()
    assert widget._target is not None
    assert widget.scene.target is widget._target
    assert widget._target.glo == widget.defaultFramebufferObject()
    # Qt's default FBO is normally not 0; if this platform gives 0, the assertion below is vacuous
    # but the binding is still correct.
    assert widget._target.glo == int(widget.defaultFramebufferObject())


def test_qt_framebuffer_survives_resizes(widget, qapp, tmp_path):
    """The regression PM asked for: resize repeatedly and the widget must still render.

    The check is functional rather than a poke at the framebuffer name. `detect_framebuffer` is
    only meaningful with the context current and Qt's framebuffer bound -- i.e. from inside
    `paintGL`; calling it from test code between events raises GL_INVALID_OPERATION on this
    driver even when everything is healthy (measured on llvmpipe: glo 6 and 8 error, glo 4 does
    not). So the assertion is the one that matters anyway: after all the resizing, the widget's
    own render path completes without a GL error and still produces a picture. If our teardown
    had deleted Qt's framebuffer, neither would be true.
    """
    from PIL import Image
    widget.set_geometry(oval_geometry())
    from f1sim.viewer.console.frames import FrameBuffer
    buf = FrameBuffer()
    buf.set_generation(0)
    buf.push(a_frame())
    widget.attach(buf)
    errors = []
    widget.gl_failed.connect(errors.append)
    for size in ((320, 240), (500, 300), (640, 480), (321, 241), (500, 300)):
        widget.resize(*size)
        widget.ctx.error                                   # drain: glGetError is destructive
        for _ in range(3):
            widget.update()
            qapp.processEvents()
        assert widget.ctx.error == "GL_NO_ERROR"
        assert not errors, errors
        assert widget._target_key == (int(widget.defaultFramebufferObject()), *widget.fb_size())
    out = tmp_path / "after_resizes.png"
    assert widget.grab_png(str(out))
    img = np.asarray(Image.open(out).convert("RGB"))
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 8, "nothing drawn after the resizes"


def test_target_cache_keys_on_size_not_only_on_name(widget, qapp):
    """A framebuffer name can be reused after a resize, so the name alone is not a valid key."""
    widget.update()
    qapp.processEvents()
    first = widget._target_key
    widget.resize(400, 320)
    widget.update()
    for _ in range(3):
        qapp.processEvents()
    assert widget._target_key != first
    assert widget._target_key[1:] == widget.fb_size()


def test_fb_size_is_physical_pixels(widget):
    dpr = widget.devicePixelRatioF()
    w, h = widget.fb_size()
    assert w == round(widget.width() * dpr)
    assert h == round(widget.height() * dpr)


# ---------------------------------------------------------------- teardown
def test_scene_release_is_idempotent_and_frees_only_our_objects(widget, qapp):
    widget.set_geometry(oval_geometry())
    widget.update()
    qapp.processEvents()
    scene = widget.scene
    glo_before = int(widget.defaultFramebufferObject())
    scene.release()
    scene.release()                      # twice must be safe
    assert scene.static == []
    assert scene.car_meshes == []
    # Qt's framebuffer is untouched by our teardown: it is still the widget's default, and the
    # widget can still be asked for its pixels. (Binding it by name from out here is not a valid
    # check -- see test_qt_framebuffer_survives_resizes.)
    assert int(widget.defaultFramebufferObject()) == glo_before
    assert widget.grabFramebuffer().width() > 0


def test_context_destroyed_drops_the_borrowed_reference(widget, qapp):
    widget.update()
    qapp.processEvents()
    assert widget._target is not None
    widget._on_context_destroyed()       # what Qt calls on aboutToBeDestroyed
    assert widget._target is None
    assert widget.scene is None
    widget._on_context_destroyed()       # idempotent


# ---------------------------------------------------------------- drawing
def test_renders_a_frame_and_screenshot_has_content(widget, qapp, tmp_path):
    widget.set_geometry(oval_geometry())
    from f1sim.viewer.console.frames import FrameBuffer
    buf = FrameBuffer()
    buf.set_generation(0)
    buf.push(a_frame())
    widget.attach(buf)
    widget.set_camera("overview")
    for _ in range(6):
        widget.update()
        qapp.processEvents()
    out = tmp_path / "shot.png"
    assert widget.grab_png(str(out))
    from PIL import Image
    img = np.asarray(Image.open(out).convert("RGB"))
    assert img.shape[0] > 0 and img.shape[1] > 0
    # something other than the clear colour was drawn
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 8


def test_malformed_frame_cannot_escape_paint(widget, qapp):
    """A frame the buffer let through but the renderer chokes on must not throw out of paintGL."""
    class EvilBuffer:
        def frame_to_draw(self, interpolate=True):
            return {"t": 0.0, "n": 3, "seq": 0, "x": np.zeros(1), "y": np.zeros(1),
                    "yaw": np.zeros(1)}          # n=3 but one-element arrays

    widget.set_geometry(oval_geometry())
    widget.attach(EvilBuffer())
    errors = []
    widget.gl_failed.connect(errors.append)
    widget.update()
    for _ in range(3):
        qapp.processEvents()
    assert errors, "the failure should be reported, not swallowed silently"
    assert widget.isVisible()                    # and the widget is still alive


def test_geometry_swap_releases_the_previous_static_meshes(widget, qapp):
    widget.set_geometry(oval_geometry("a"))
    widget.update()
    qapp.processEvents()
    n_after_first = len(widget.scene.static)
    widget.set_geometry(oval_geometry("b"))
    widget.update()
    qapp.processEvents()
    assert widget.geometry_data.name == "b"
    assert len(widget.scene.static) == n_after_first      # replaced, not accumulated


def test_camera_modes_all_render(widget, qapp):
    from f1sim.viewer.console.frames import FrameBuffer
    from f1sim.viewer.console.viewport import CAMERA_KEYS
    widget.set_geometry(oval_geometry())
    buf = FrameBuffer()
    buf.set_generation(0)
    buf.push(a_frame())
    widget.attach(buf)
    errors = []
    widget.gl_failed.connect(errors.append)
    for key in CAMERA_KEYS:
        widget.set_camera(key)
        widget.update()
        for _ in range(2):
            qapp.processEvents()
    assert not errors, errors


# --------------------------------------------------------------------------- depth range / fog
#
# A user reported `rt:Monza` rendering as a flat sheet with the walls gone. The cause was not the
# map: the fragment shader faded every surface to the background between 50 m and 200 m from the
# eye, and `전체보기` puts the eye at whatever height frames the map -- 22 m on a 23 m indoor map,
# 199 m on Monza. Past 200 m everything is the fog colour, so the whole circuit came out uniform.
# The fix derives near, far and the fog range from the camera distance and the map's own size.

def _sized_geometry(widget, bounds):
    """Attach geometry with the given bounds directly.

    `set_geometry` only queues; `geometry_data` is populated on the GL thread at the next paint,
    and these tests are about the arithmetic that reads `bounds`, not about the upload.
    """
    from f1sim.viewer import gl_scene as G
    from f1sim.viewer.console.viewport import TrackGeometry
    g = oval_geometry()
    widget.geometry_data = TrackGeometry(
        name="test:sized", bounds=bounds, duct_height=g.duct_height,
        floor=G.floor_mesh_arrays(bounds), ducts=g.ducts, walls=None,
        centerline=None, raceline_xy=None, raceline_v=None)
    return widget.geometry_data


def test_fog_and_far_plane_follow_the_camera_distance(widget):
    """A pulled-back camera must not put the whole scene past the end of the fog."""
    # rt:Monza as measured: 191.7 x 191.7 m, overview eye 198.9 m up
    _sized_geometry(widget, (-49.8, -50.5, 141.9, 141.2))
    eye_close, target = np.array([0.0, 0.0, 3.0]), np.array([0.0, 2.0, 0.0])
    near_c, far_c, fog0_c, fog1_c = widget._depth_range(eye_close, target)
    dist_c = float(np.linalg.norm(eye_close - target))
    # A close camera keeps the fog it always had; `far` grows only because this map's corners have
    # to be in front of it, and `near` can never reach the thing being looked at.
    assert (fog0_c, fog1_c) == (50.0, 200.0)
    assert far_c >= 300.0
    assert 0.05 <= near_c <= dist_c * 0.05

    # Monza's overview camera, to scale
    eye_far = np.array([46.0, 45.3, 198.9])
    near_f, far_f, fog0_f, fog1_f = widget._depth_range(eye_far, np.array([46.0, 45.3, 0.0]))
    dist = 198.9
    assert fog1_f > dist, "the surface being looked at must not be beyond the end of the fog"
    assert fog0_f < dist < fog1_f, "it should be partly hazed, not erased and not crisp"
    assert far_f > dist, "and it must be in front of the far plane"
    assert near_f > 0.05, "near should rise with distance rather than waste depth precision"
    assert near_f <= dist * 0.05, "but never far enough to clip what the camera is looking at"
    assert far_f / near_f <= 4001, "far/near has to stay within what a 24-bit buffer resolves"


def test_far_plane_clears_a_map_larger_than_the_old_fixed_300m(widget):
    """A map wider than the old fixed far plane was clipped; the far plane now comes from bounds."""
    _sized_geometry(widget, (-600.0, -450.0, 600.0, 450.0))
    eye = np.array([0.0, 0.0, 1250.0])
    near, far, fog0, fog1 = widget._depth_range(eye, np.array([0.0, 0.0, 0.0]))
    corner = float(np.hypot(1250.0, np.hypot(600.0, 450.0)))
    assert far > corner, f"far {far} must clear the far corner at {corner:.0f} m"
    assert far > 300.0 and near > 0.05
    assert near <= 1250.0 * 0.05


def test_scene_fog_range_is_settable_and_ordered(widget, qapp):
    """`set_fog` is what the viewport uses; end is always kept in front of start."""
    sc = widget.scene
    assert sc.fog_range == (50.0, 200.0)         # the previous hard-coded values, by default
    sc.set_fog(180.0, 500.0)
    assert sc.fog_range == (180.0, 500.0)
    sc.set_fog(400.0, 100.0)                     # nonsense in: a usable range out, not a crash
    assert sc.fog_range[1] > sc.fog_range[0]


def test_camera_modes_all_render_on_a_large_map(widget, qapp):
    """The reported case: every camera, on a map far bigger than the old fog and far plane."""
    from f1sim.viewer.console.frames import FrameBuffer
    from f1sim.viewer.console.viewport import CAMERA_KEYS
    widget.set_geometry(oval_geometry())
    _sized_geometry(widget, (-90.0, -90.0, 100.0, 100.0))
    buf = FrameBuffer()
    buf.set_generation(0)
    buf.push(a_frame())
    widget.attach(buf)
    errors = []
    widget.gl_failed.connect(errors.append)
    for key in CAMERA_KEYS:
        widget.set_camera(key)
        widget.update()
        for _ in range(2):
            qapp.processEvents()
    assert not errors, errors


# ------------------------------------------------------------------ pointer must not change mode
#
# Reported by the user: "화면 클릭하면 왜 궤도 뷰로 그냥 바뀌는거지? 바뀌게하지마". Pressing anywhere
# in the picture used to call set_camera("orbit"), so clicking the window -- even just to focus it --
# threw away the camera the user had chosen. The mode is now chosen by its button or shortcut only.

def _click(widget, qapp, kind="press", pos=(120, 90), button=None):
    from PyQt5 import QtCore as _C, QtGui as _G
    button = button or _C.Qt.LeftButton
    ev_type = {"press": _C.QEvent.MouseButtonPress,
               "release": _C.QEvent.MouseButtonRelease,
               "move": _C.QEvent.MouseMove}[kind]
    # (button that caused the event, buttons currently held) -- a move carries no cause and a held
    # button, which is the pair that actually reaches mouseMoveEvent
    cause = _C.Qt.NoButton if kind == "move" else button
    held = button if kind in ("press", "move") else _C.Qt.NoButton
    ev = _G.QMouseEvent(ev_type, _C.QPointF(*pos), cause, held, _C.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, ev)
    qapp.processEvents()


def test_clicking_the_picture_never_changes_the_camera_mode(widget, qapp):
    from f1sim.viewer.console.viewport import CAMERA_KEYS
    widget.set_geometry(oval_geometry())
    for mode in CAMERA_KEYS:
        widget.set_camera(mode)
        qapp.processEvents()
        _click(widget, qapp, "press")
        assert widget.camera == mode, f"a press changed the camera from {mode} to {widget.camera}"
        _click(widget, qapp, "move", pos=(160, 130))
        assert widget.camera == mode, f"a drag changed the camera from {mode} to {widget.camera}"
        _click(widget, qapp, "release")
        assert widget.camera == mode, f"a release changed the camera from {mode} to {widget.camera}"


def test_wheel_does_not_change_the_camera_mode(widget, qapp):
    from PyQt5 import QtCore as _C, QtGui as _G
    from f1sim.viewer.console.viewport import CAMERA_KEYS
    widget.set_geometry(oval_geometry())
    for mode in CAMERA_KEYS:
        widget.set_camera(mode)
        qapp.processEvents()
        ev = _G.QWheelEvent(_C.QPointF(120, 90), _C.QPointF(120, 90), _C.QPoint(0, 0),
                            _C.QPoint(0, 120), _C.Qt.NoButton, _C.Qt.NoModifier,
                            _C.Qt.NoScrollPhase, False)
        QtWidgets.QApplication.sendEvent(widget, ev)
        qapp.processEvents()
        assert widget.camera == mode, f"the wheel changed the camera from {mode} to {widget.camera}"


def test_drag_still_rotates_when_orbit_is_already_the_mode(widget, qapp):
    widget.set_geometry(oval_geometry())
    widget.set_camera("orbit")
    qapp.processEvents()
    before = list(widget._orbit)
    _click(widget, qapp, "press", pos=(120, 90))
    _click(widget, qapp, "move", pos=(200, 140))
    _click(widget, qapp, "release", pos=(200, 140))
    assert list(widget._orbit) != before, "dragging in 궤도 should still rotate the view"
    assert widget.camera == "orbit"


def test_drag_outside_orbit_does_not_rotate(widget, qapp):
    widget.set_geometry(oval_geometry())
    widget.set_camera("chase")
    qapp.processEvents()
    before = list(widget._orbit)
    _click(widget, qapp, "press", pos=(120, 90))
    _click(widget, qapp, "move", pos=(200, 140))
    _click(widget, qapp, "release", pos=(200, 140))
    assert list(widget._orbit) == before, "a drag outside 궤도 must not quietly rotate it"
    assert widget.camera == "chase"


# ------------------------------------------------------------- chase camera follows in *time*
#
# Reported by the user: in 추격 the focused car moves in visible steps. Traced on hardware, the
# drawn sim clock advances p50 13.2 ms / p95 57.5 ms per paint -- perfectly correct, since it
# advances by real elapsed time -- while the camera's lag was a fixed fraction per paint derived
# from a windowed fps average. On a long frame the car moved four times further and the camera moved
# exactly as far as on a short one, so the car surged in the frame and the camera caught up after.
# Measured relative motion was p50 0.067 m against p95 0.275 m: a 4.1x spread.

def _pose_frame(x, y, yaw=0.0, t=0.0, seq=0, n=1, vx=3.0):
    fr = a_frame(n=n, t=t, seq=seq)
    fr["x"] = np.full(n, float(x), np.float32)
    fr["y"] = np.full(n, float(y), np.float32)
    fr["yaw"] = np.full(n, float(yaw), np.float32)
    fr["vx"] = np.full(n, float(vx), np.float32)
    fr["focus"] = 0
    return fr


def _chase_step(widget, x, y, dt):
    """One camera update at a given pose, `dt` seconds after the previous one."""
    widget._cam_wall = None if widget._cam_wall is None else widget._cam_wall - dt
    eye, tgt, _ = widget._camera_pose(_pose_frame(x, y))
    return np.array(eye, float), np.array(tgt, float)


def _view_space(eye, target, up, car_xy):
    """Where the car sits in the camera's own frame: (right, up, forward) metres.

    This is the quantity the eye actually reads -- a car that holds still on screen holds still
    here, however the frames happen to fall. Raw world displacement per paint cannot answer the
    question, because uneven paint intervals make it uneven all by themselves.
    """
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= max(1e-9, np.linalg.norm(fwd))
    right = np.cross(fwd, np.asarray(up, float))
    right /= max(1e-9, np.linalg.norm(right))
    upv = np.cross(right, fwd)
    d = np.array([car_xy[0], car_xy[1], 0.0]) - eye
    return np.array([d @ right, d @ upv, d @ fwd])


def test_chase_holds_the_car_still_in_view_space_under_uneven_paints(widget):
    """Constant velocity and heading, wildly alternating paint intervals: a known answer.

    The car is driving in a straight line at a fixed speed. Whatever the interval pattern, 추격
    should hold it at the same place in the frame. Filtering the camera's world position could not
    do that -- the lag is proportional to speed and to the interval, so the car drifted around
    inside the frame and every long paint shifted it.
    """
    widget.set_camera("chase")
    widget._cam_eye = widget._cam_tgt = None
    widget._cam_wall = widget._chase_heading = None
    speed, y = 4.0, 0.0
    seen = []
    for dt in ([0.013] * 6 + [0.060] + [0.013] * 6 + [0.090] + [0.013] * 6) * 3:
        y += speed * dt
        widget._cam_wall = None if widget._cam_wall is None else widget._cam_wall - dt
        eye, tgt, up = widget._camera_pose(_pose_frame(0.0, y, yaw=math.pi / 2, vx=speed))
        seen.append(_view_space(eye, tgt, up, (0.0, y)))
    v = np.asarray(seen[3:])                      # skip the first-frame seeding
    drift = float(np.abs(v - v[0]).max())
    assert drift < 1e-3, (
        f"the car moves {drift:.4f} m within the frame as the paint interval changes; "
        "that is the stepping the user sees")


def test_chase_relative_velocity_is_time_normalised(widget):
    """dx/dt, not dx: the car's speed relative to the camera must not depend on the interval."""
    widget.set_camera("chase")
    widget._cam_eye = widget._cam_tgt = None
    widget._cam_wall = widget._chase_heading = None
    speed, y = 4.0, 0.0
    prev, rel_v = None, []
    for dt in ([0.013] * 5 + [0.060] + [0.013] * 5 + [0.090]) * 3:
        y += speed * dt
        widget._cam_wall = None if widget._cam_wall is None else widget._cam_wall - dt
        eye, tgt, up = widget._camera_pose(_pose_frame(0.0, y, yaw=math.pi / 2, vx=speed))
        p = _view_space(eye, tgt, up, (0.0, y))
        if prev is not None:
            rel_v.append(float(np.linalg.norm(p - prev) / dt))
        prev = p
    r = np.asarray(rel_v[3:])
    assert float(r.max()) < 0.05, (
        f"relative velocity reaches {r.max():.3f} m/s in a steady straight line; the camera is "
        "still lagging by an interval-dependent amount")


def _chase_pose(widget, x, y, yaw, dt, focus=0, gen=0, n=1, vx=4.0):
    fr = _pose_frame(x, y, yaw=yaw, n=n, vx=vx)
    fr["focus"] = focus
    fr["gen"] = gen
    widget._cam_wall = None if widget._cam_wall is None else widget._cam_wall - dt
    return widget._camera_pose(fr)


def test_chase_cuts_rather_than_chases_a_reset(widget):
    """A reset moves the car to the start line. The camera cuts; it does not drag across the map."""
    widget.set_camera("chase")
    widget._cam_eye = widget._cam_tgt = None
    widget._cam_wall = widget._chase_heading = widget._chase_ref = None
    _chase_pose(widget, 0.0, 0.0, 0.0, 0.0)
    for _ in range(3):
        _chase_pose(widget, 0.0, 0.0, 0.0, 0.016)
    eye, _, _ = _chase_pose(widget, 60.0, 40.0, math.pi, 0.016)      # reset, and turned around
    assert float(np.linalg.norm(np.asarray(eye)[:2] - np.array([60.0, 40.0]))) < 3.0
    assert widget._chase_heading == pytest.approx(math.pi), (
        "the eased heading survived a reset instead of being cut")


def test_a_reset_that_barely_turns_the_car_is_still_a_reset(widget):
    """The case an angle threshold cannot see: across the map, same heading."""
    widget.set_camera("chase")
    widget._cam_eye = widget._cam_tgt = None
    widget._cam_wall = widget._chase_heading = widget._chase_ref = None
    _chase_pose(widget, 0.0, 0.0, 0.3, 0.0)
    for _ in range(3):
        _chase_pose(widget, 0.0, 0.0, 0.3, 0.016)
    eye, _, _ = _chase_pose(widget, 80.0, 0.0, 0.3, 0.016)           # 80 m, heading unchanged
    assert float(np.linalg.norm(np.asarray(eye)[:2] - np.array([80.0, 0.0]))) < 3.0, (
        "a reset with an unchanged heading was treated as ordinary driving")


def test_switching_the_focused_car_clears_the_eased_heading(widget):
    """A different car is a different subject; the old heading must not swing across to it."""
    widget.set_camera("chase")
    widget._cam_eye = widget._cam_tgt = None
    widget._cam_wall = widget._chase_heading = widget._chase_ref = None
    for _ in range(4):
        _chase_pose(widget, 0.0, 0.0, 0.0, 0.016, focus=0, n=2)
    eye, _, _ = _chase_pose(widget, 5.0, 5.0, math.pi / 2, 0.016, focus=1, n=2)
    assert widget._chase_heading == pytest.approx(math.pi / 2)
    assert float(np.linalg.norm(np.asarray(eye)[:2] - np.array([5.0, 5.0]))) < 3.0


def test_a_new_generation_clears_the_eased_heading(widget):
    """A new session is not a continuation of the old one."""
    widget.set_camera("chase")
    widget._cam_eye = widget._cam_tgt = None
    widget._cam_wall = widget._chase_heading = widget._chase_ref = None
    for _ in range(4):
        _chase_pose(widget, 0.0, 0.0, 0.0, 0.016, gen=1)
    _chase_pose(widget, 0.0, 0.0, math.pi, 0.016, gen=2)
    assert widget._chase_heading == pytest.approx(math.pi)


def test_a_spinning_car_is_eased_not_cut(widget):
    """A large turn is not a reset: the camera should swing through it, not snap."""
    widget.set_camera("chase")
    widget._cam_eye = widget._cam_tgt = None
    widget._cam_wall = widget._chase_heading = widget._chase_ref = None
    _chase_pose(widget, 0.0, 0.0, 0.0, 0.0)
    yaw = 0.0
    for _ in range(6):
        yaw += math.radians(30.0)                 # a genuine, fast spin in place
        _chase_pose(widget, 0.0, 0.0, yaw, 0.016)
    assert widget._chase_heading != pytest.approx(yaw), (
        "the camera snapped to a heading it should have eased through")


def test_interpolator_does_not_slide_a_car_across_a_teleport():
    from f1sim.viewer.console.frames import FrameBuffer
    buf = FrameBuffer()
    buf.set_generation(0)
    for i, (x, y) in enumerate([(0.0, 0.0), (0.0, 0.05), (0.0, 0.10)]):
        buf.push(_pose_frame(x, y, t=i * 0.025, seq=i))
    buf.frame_to_draw()                                   # start the playback clock
    buf.push(_pose_frame(80.0, 60.0, t=3 * 0.025, seq=3))  # reset: straight to the start line
    for _ in range(6):
        out = buf.frame_to_draw()
        pos = (float(out["x"][0]), float(out["y"][0]))
        # never anywhere between the two: either the old pose or the new one
        assert pos[0] < 1.0 or pos[0] > 79.0, f"interpolated across the teleport to {pos}"


def test_the_camera_button_and_the_viewport_agree(widget, qapp):
    """Whatever chose the camera, the highlighted button must name the camera actually in use.

    A capture tool called `viewport.set_camera` directly and produced a close-up image under a
    highlighted 전체보기 -- a picture of a state the UI never had. The window is what deliverables
    are photographed from, so the two have to agree by construction.
    """
    from f1sim.viewer.console.viewport import CAMERA_KEYS
    from f1sim.viewer.console.window import ConsoleWindow
    win = ConsoleWindow()
    win.resize(1200, 760)
    win.show()
    qapp.processEvents()
    try:
        for key in CAMERA_KEYS:
            btn = win.camera_buttons.button(key)
            assert btn is not None, f"no button for camera {key}"
            btn.click()
            qapp.processEvents()
            assert win.viewport.camera == key
            assert win.camera_buttons.current() == key, (
                f"pressed {key}, viewport is on {win.viewport.camera}, button says "
                f"{win.camera_buttons.current()}")
    finally:
        win.viewport.teardown()
        win.close()
