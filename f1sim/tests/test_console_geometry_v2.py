"""Console wiring for geometry v2.

The worker now measures how the track actually lies and how big the drawn content is, and sends an
apron plus a backdrop instead of a canvas-shaped quad. The console has to use those *for the camera
only* -- world coordinates, the occupancy grid and everything the policy sees are untouched -- and
it has to keep working when paired with a worker that sends none of it.

Numbers here are the ones map-redesign measured (`work/claude-map-redesign/evidence/`):
Korea's track lies at 82.95 degrees, Monza's at 60.43, and Monza's content is 38.9% of its canvas.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("PyQt5")
from PyQt5 import QtWidgets                                             # noqa: E402

from f1sim.viewer.console.viewport import TrackGeometry                 # noqa: E402


def v1_payload():
    """What an older worker sends: no version, no presentation, canvas bounds."""
    return {"name": "old:map", "bounds": (-12.0, -10.0, 11.0, 11.0), "duct_height": 0.2,
            "floor": None, "ducts": None, "walls": None,
            "centerline": None, "raceline_xy": None, "raceline_v": None, "build_ms": 3.0}


def v2_payload(angle_deg=82.95, centre=(0.5, 1.0), extent=(20.0, 60.0),
               canvas=(-60.0, -60.0, 60.0, 60.0), content=(-9.0, -28.0, 10.0, 30.0)):
    p = v1_payload()
    p.update({"name": "new:map", "bounds": canvas, "geometry_version": 2,
              "backdrop": (np.zeros((3, 3), np.float32), np.zeros((3, 3), np.float32),
                           np.zeros((3, 4), np.float32), np.zeros((1, 3), np.int32)),
              "content_bounds": content,
              "presentation": {"up_angle_deg": angle_deg, "center": centre, "extent": extent}})
    return p


# ----------------------------------------------------------------- the payload -> TrackGeometry
def test_a_v1_payload_still_produces_a_drawable_map():
    g = TrackGeometry(name="old:map", bounds=(-12.0, -10.0, 11.0, 11.0), duct_height=0.2)
    assert g.geometry_version == 1
    assert g.backdrop is None and g.presentation is None and g.content_bounds is None
    assert g.frame() is None, "a v1 payload has no measured framing to offer"
    assert g.drawn_bounds() == g.bounds, "with no content bounds the canvas is all we know"


def test_v2_fields_are_read_off_the_payload():
    p = v2_payload()
    g = TrackGeometry(
        name=p["name"], bounds=tuple(p["bounds"]), duct_height=p["duct_height"],
        geometry_version=int(p.get("geometry_version", 1)), backdrop=p.get("backdrop"),
        content_bounds=tuple(p["content_bounds"]), presentation=dict(p["presentation"]))
    assert g.geometry_version == 2
    ang, centre, extent = g.frame()
    assert math.degrees(ang) == pytest.approx(82.95)
    assert list(centre) == [0.5, 1.0] and list(extent) == [20.0, 60.0]
    assert g.drawn_bounds() == (-9.0, -28.0, 10.0, 30.0)
    assert g.drawn_bounds() != g.bounds, "content bounds must not silently fall back to the canvas"


def test_a_presentation_missing_its_parts_is_ignored_rather_than_half_used():
    g = TrackGeometry(name="x", bounds=(0, 0, 1, 1), duct_height=0.2,
                      geometry_version=2, presentation={"up_angle_deg": 30.0})
    assert g.frame() is None, "an angle with no centre or extent cannot frame anything"


def test_the_backdrop_counts_towards_the_uploaded_size():
    arrays = (np.zeros((100, 3), np.float32), np.zeros((100, 3), np.float32),
              np.zeros((100, 4), np.float32), np.zeros((50, 3), np.int32))
    without = TrackGeometry(name="x", bounds=(0, 0, 1, 1), duct_height=0.2, floor=arrays)
    with_bd = TrackGeometry(name="x", bounds=(0, 0, 1, 1), duct_height=0.2, floor=arrays,
                            backdrop=arrays)
    assert with_bd.nbytes() > without.nbytes()


# ----------------------------------------------------------------- the camera
@pytest.fixture(scope="module")
def qapp():
    # Scoped: `os.environ` is process-global and a module fixture that sets it without restoring
    # leaks the platform into every later test AND into any child process they spawn. That is not
    # hypothetical -- `test_console_launch_smoke` builds its child env from `os.environ`, so an
    # offscreen platform left here made the console open with no X window and its `xdotool` search
    # find nothing. Restored on teardown.
    _prev_qpa = os.environ.get("QT_QPA_PLATFORM")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from f1sim.viewer.console import app as A
    yield QtWidgets.QApplication.instance() or A.create_app(["test"])
    if _prev_qpa is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = _prev_qpa


@pytest.fixture
def widget(qapp):
    from f1sim.viewer.console.viewport import ViewportWidget
    w = ViewportWidget()
    w.resize(800, 480)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def a_frame(n=1):
    return {"gen": 0, "seq": 0, "t": 0.0, "n": n, "focus": 0,
            "x": np.zeros(n, np.float32), "y": np.zeros(n, np.float32),
            "yaw": np.zeros(n, np.float32), "vx": np.zeros(n, np.float32),
            "steer": np.zeros(n, np.float32), "roll": np.zeros(n, np.float32),
            "pitch": np.zeros(n, np.float32)}


def geom_from(p):
    return TrackGeometry(
        name=p["name"], bounds=tuple(p["bounds"]), duct_height=p["duct_height"],
        geometry_version=int(p.get("geometry_version", 1)), backdrop=p.get("backdrop"),
        content_bounds=tuple(p["content_bounds"]) if p.get("content_bounds") else None,
        presentation=dict(p["presentation"]) if p.get("presentation") else None)


def frame_corners(ang, centre, extent):
    """The four corners of the oriented content rectangle, in world x/y.

    `content_frame` measures extent[0] along `ang` and extent[1] across it, so the corners are
    centre +- half-along*t +- half-across*n. Deriving them here, from the contract rather than from
    the viewport's arithmetic, is the point: a test that reuses the implementation's own formula
    passes whatever axes that formula mixed up.
    """
    t = np.array([math.cos(ang), math.sin(ang)])
    n = np.array([-t[1], t[0]])
    ha, hc = extent[0] / 2.0, extent[1] / 2.0
    return [centre + sa * ha * t + sc * hc * n
            for sa in (-1, 1) for sc in (-1, 1)]


def visible(widget, eye, target, up, pt, fov_deg=55.0):
    """Is a world point inside the projected frustum of this camera? (fraction of half-extent)

    Returns (u, v) in [-1, 1] when on screen: u across the screen, v up it.
    """
    w, h = widget.fb_size()
    aspect = w / max(1, h)
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    upv = np.cross(right, fwd)
    d = np.array([pt[0], pt[1], 0.0]) - eye
    z = float(d @ fwd)
    tan = math.tan(math.radians(fov_deg) / 2.0)
    return float(d @ right) / (z * tan * aspect), float(d @ upv) / (z * tan)


def assert_frames_all_corners(widget, geom, slack=1.0):
    widget.geometry_data = geom
    widget.set_camera("overview")
    widget._cam_eye = widget._cam_tgt = None
    eye, target, up = widget._camera_pose(a_frame())
    ang, centre, extent = geom.frame()
    worst = 0.0
    for c in frame_corners(ang, centre, extent):
        u, v = visible(widget, eye, target, up, c)
        worst = max(worst, abs(u), abs(v))
        assert abs(u) <= slack and abs(v) <= slack, (
            f"corner {c} projects to ({u:.3f}, {v:.3f}) -- outside the frustum, so the map is "
            f"cropped on screen")
    return eye, worst


def test_overview_fits_every_corner_of_the_content(widget):
    """Korea, standing at 82.95 degrees. All four corners of the measured frame must be on screen."""
    geom = geom_from(v2_payload(angle_deg=82.95, centre=(0.5, 1.0), extent=(60.0, 20.0)))
    eye, worst = assert_frames_all_corners(widget, geom)
    assert (eye[0], eye[1]) == pytest.approx((0.5, 1.0)), "framed off the measured centre"
    assert worst > 0.5, (
        f"the content only fills {worst:.2f} of the half-extent -- the camera is much further back "
        "than it needs to be")


@pytest.mark.parametrize("angle_deg", [0.0, 7.05, 45.0, 60.43, 82.95, 120.0, 179.0])
@pytest.mark.parametrize("extent", [(60.0, 20.0), (20.0, 60.0), (40.0, 40.0)])
def test_every_orientation_and_shape_is_fully_framed(widget, angle_deg, extent):
    """Whichever way the track lies and whichever way round its extent is, nothing is cropped."""
    geom = geom_from(v2_payload(angle_deg=angle_deg, centre=(3.0, -2.0), extent=extent))
    assert_frames_all_corners(widget, geom)


def test_overview_picks_the_orientation_that_wastes_less_viewport(widget):
    """A landscape map on a landscape screen should be laid across it, not stood on end."""
    # 800x480 viewport: aspect 1.667. A 60 x 20 frame laid across needs less distance than stood up.
    geom = geom_from(v2_payload(angle_deg=0.0, centre=(0.0, 0.0), extent=(60.0, 20.0)))
    eye_landscape, worst = assert_frames_all_corners(widget, geom)
    # the same content described the other way round must end up at the same distance
    geom2 = geom_from(v2_payload(angle_deg=90.0, centre=(0.0, 0.0), extent=(20.0, 60.0)))
    eye_rotated, _ = assert_frames_all_corners(widget, geom2)
    assert float(eye_landscape[2]) == pytest.approx(float(eye_rotated[2]), rel=1e-6), (
        "the same rectangle framed two equivalent ways needs two different distances")
    assert worst > 0.9, (
        f"the long axis only reaches {worst:.2f} of the half-extent; the frame is not being laid "
        "along the screen's long axis")


def test_overview_frames_the_content_not_the_canvas(widget):
    """Monza's content is 38.9% of its canvas; framing on the canvas pushes the camera far back."""
    canvas = (-96.0, -96.0, 96.0, 96.0)
    content = (-30.0, -60.0, 30.0, 60.0)
    v2 = geom_from(v2_payload(angle_deg=60.43, centre=(0.0, 0.0), extent=(120.0, 60.0),
                              canvas=canvas, content=content))
    eye_v2, _ = assert_frames_all_corners(widget, v2)

    v1 = TrackGeometry(name="old", bounds=canvas, duct_height=0.2)   # same map, v1 payload
    widget.geometry_data = v1
    widget._cam_eye = widget._cam_tgt = None
    eye_v1, _, _ = widget._camera_pose(a_frame())
    assert eye_v2[2] < eye_v1[2], (
        f"v2 framing put the eye at {eye_v2[2]:.1f} m, no closer than the canvas framing "
        f"{eye_v1[2]:.1f} m -- the content framing is not being used")


def test_a_v1_payload_frames_exactly_as_it_used_to(widget):
    """The fallback path is the old arithmetic, unchanged."""
    b = (-12.0, -10.0, 11.0, 11.0)
    widget.geometry_data = TrackGeometry(name="old", bounds=b, duct_height=0.2)
    widget.set_camera("overview")
    eye, target, up = widget._camera_pose(a_frame())
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    assert (eye[0], eye[1]) == pytest.approx((cx, cy))
    assert (target[0], target[1]) == pytest.approx((cx, cy))


def test_chase_and_top_are_untouched_by_the_presentation(widget):
    """The user's camera meanings do not change: 추격 follows the car, 위에서 stays car-relative."""
    fr = a_frame()
    fr["x"][0], fr["y"][0], fr["yaw"][0] = 3.0, 4.0, 0.6
    for mode in ("chase", "top"):
        widget.geometry_data = TrackGeometry(name="old", bounds=(-9, -9, 9, 9), duct_height=0.2)
        widget.set_camera(mode)
        widget._cam_wall = widget._chase_heading = widget._chase_ref = None
        plain = widget._camera_pose(fr)
        widget.geometry_data = geom_from(v2_payload(angle_deg=82.95))
        widget.set_camera("overview")
        widget.set_camera(mode)
        widget._cam_wall = widget._chase_heading = widget._chase_ref = None
        with_v2 = widget._camera_pose(fr)
        for a, b in zip(plain, with_v2):
            assert np.allclose(np.asarray(a, float), np.asarray(b, float)), (
                f"{mode} changed when the map gained a presentation")


def test_depth_range_sizes_itself_on_the_drawn_content(widget):
    """The far plane has to clear what is drawn; the canvas can be several times larger."""
    canvas = (-300.0, -300.0, 300.0, 300.0)
    content = (-20.0, -20.0, 20.0, 20.0)
    widget.geometry_data = geom_from(v2_payload(canvas=canvas, content=content))
    near_v2, far_v2, _, _ = widget._depth_range(np.array([0.0, 0.0, 50.0]), np.zeros(3))
    widget.geometry_data = TrackGeometry(name="old", bounds=canvas, duct_height=0.2)
    near_v1, far_v1, _, _ = widget._depth_range(np.array([0.0, 0.0, 50.0]), np.zeros(3))
    assert far_v2 < far_v1, "the far plane is still being sized off the canvas"
    corner = float(math.hypot(50.0, math.hypot(20.0, 20.0)))
    assert far_v2 > corner, "and it must still clear the far corner of what is drawn"
