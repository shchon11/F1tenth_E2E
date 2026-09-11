"""Placed props: worker batching, console deserialisation, upload and catalogue.

`Track.props` holds placements, not meshes -- a style name, a pose and a seed. The worker turns
them into world-space vertex arrays merged by material, and the console uploads what it is given.
The split matters: building is numpy work, and the thread that would otherwise do it is the one
holding the GL context.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim import props as P                                            # noqa: E402
from f1sim.track import StaticProp                                      # noqa: E402
from f1sim.viewer.sim_worker import prop_batches                        # noqa: E402
from f1sim.viewer.console.session import _prop_batches                  # noqa: E402
from f1sim.viewer.console.viewport import TrackGeometry                 # noqa: E402


class FakeTrack:
    def __init__(self, placements):
        self.props = tuple(placements)


def a_box(x=1.0, y=2.0, yaw=0.0, seed=0, style="cardboard_box"):
    return StaticProp(style=style, x=x, y=y, yaw=yaw, seed=seed)


# ----------------------------------------------------------------- worker side
def test_a_track_with_no_props_produces_no_batches():
    assert prop_batches(FakeTrack([])) == []
    assert prop_batches(FakeTrack(None) if False else FakeTrack(())) == []


def test_every_style_in_the_catalogue_can_be_batched():
    placed = [a_box(x=float(i), style=s) for i, s in enumerate(P.STYLES)]
    batches = prop_batches(FakeTrack(placed))
    assert batches, "the whole catalogue produced nothing"
    total_props = sum(b["n_props"] for b in batches)
    assert total_props == len(P.STYLES)
    for b in batches:
        assert b["pos"].dtype == np.float32 and b["pos"].shape[1] == 3
        assert b["nrm"].shape == b["pos"].shape
        assert b["col"].shape == (len(b["pos"]), 4)
        assert b["idx"].dtype == np.int32 and len(b["idx"]) % 3 == 0
        assert int(b["idx"].max()) < len(b["pos"]), "an index points past the merged vertices"
        assert b["n_tris"] == len(b["idx"]) // 3


def test_batches_are_merged_by_material_not_per_prop():
    """Twenty boxes must not be twenty draw calls."""
    placed = [a_box(x=float(i)) for i in range(20)]
    batches = prop_batches(FakeTrack(placed))
    assert len(batches) == 1, f"20 identical props produced {len(batches)} batches"
    assert batches[0]["n_props"] == 20
    mixed = prop_batches(FakeTrack(placed + [a_box(style="steel_drum")]))
    assert len(mixed) == 2, "a second material should add exactly one batch"
    assert {b["material"] for b in mixed} == {"rubber", "metal"}


def test_a_prop_is_placed_where_it_was_put():
    """The mesh arrives in world space: the worker applies the pose, the console does not."""
    at_origin = prop_batches(FakeTrack([a_box(x=0.0, y=0.0)]))[0]["pos"]
    moved = prop_batches(FakeTrack([a_box(x=10.0, y=-4.0)]))[0]["pos"]
    assert np.allclose(moved[:, 0] - at_origin[:, 0], 10.0)
    assert np.allclose(moved[:, 1] - at_origin[:, 1], -4.0)
    assert np.allclose(moved[:, 2], at_origin[:, 2]), "z must not move with an x/y placement"


def test_yaw_rotates_the_mesh_and_keeps_it_rigid():
    a = prop_batches(FakeTrack([a_box(x=0.0, y=0.0, yaw=0.0)]))[0]
    b = prop_batches(FakeTrack([a_box(x=0.0, y=0.0, yaw=math.pi / 2)]))[0]
    # a rigid rotation preserves every pairwise distance
    da = np.linalg.norm(a["pos"][:50, None, :] - a["pos"][None, :50, :], axis=-1)
    db = np.linalg.norm(b["pos"][:50, None, :] - b["pos"][None, :50, :], axis=-1)
    assert np.allclose(da, db, atol=1e-5), "yaw changed the shape, not just the orientation"
    assert not np.allclose(a["pos"], b["pos"]), "yaw did nothing"
    # and the normals turn with it
    assert not np.allclose(a["nrm"], b["nrm"])
    assert np.allclose(np.linalg.norm(b["nrm"], axis=1), 1.0, atol=1e-4), "normals must stay unit"


def test_a_prop_that_cannot_be_built_stops_the_session_rather_than_vanishing():
    """The obstacle stays in the simulation whether or not it can be drawn.

    Skipping its mesh would leave a car colliding with something the LiDAR returns and the screen
    does not show. `StaticProp` exists precisely to avoid that, so the only honest outcome is to
    refuse to prepare the session.
    """
    from f1sim.viewer.sim_worker import StartConfigError
    placed = [a_box(), StaticProp(style="no_such_style", x=0.0, y=0.0), a_box(x=5.0)]
    with pytest.raises(StartConfigError) as e:
        prop_batches(FakeTrack(placed))
    msg = str(e.value)
    assert "no_such_style" in msg
    assert "충돌체" in msg, "the error should say why a missing mesh is not survivable"


def test_a_track_that_places_nothing_is_not_an_error():
    """No props at all is ordinary; only a prop we cannot draw is fatal."""
    assert prop_batches(FakeTrack([])) == []


# ----------------------------------------------------------------- console side
def test_the_console_accepts_what_the_worker_sends():
    sent = prop_batches(FakeTrack([a_box(), a_box(x=3.0, style="steel_drum")]))
    got = _prop_batches(sent)
    assert got is not None and len(got) == len(sent)
    assert {b["material"] for b in got} == {"rubber", "metal"}


@pytest.mark.parametrize("break_it", [
    lambda b: b.pop("pos"),
    lambda b: b.__setitem__("nrm", np.zeros((3, 3), np.float32)),
    lambda b: b.__setitem__("col", np.zeros((len(b["pos"]), 3), np.float32)),
    lambda b: b.__setitem__("idx", np.array([0, 1], np.int32)),
    lambda b: b.__setitem__("idx", np.array([0, 1, 10 ** 6], np.int32)),
    # numpy would wrap a negative index; GL reads it as an enormous unsigned one and draws
    # nothing or garbage. Checking only the maximum missed this.
    lambda b: b.__setitem__("idx", np.array([0, 1, -1], np.int32)),
    lambda b: b.__setitem__("idx", np.array([-3, -2, -1], np.int32)),
    # non-finite vertex data: the triangle collapses or flies off the map, and the obstacle
    # stops being visible while the simulator keeps colliding with it
    lambda b: b["pos"].__setitem__((0, 0), np.nan),
    lambda b: b["pos"].__setitem__((0, 2), np.inf),
    lambda b: b["nrm"].__setitem__((0, 1), np.nan),
    lambda b: b["col"].__setitem__((0, 3), np.nan),
])
def test_a_malformed_batch_rejects_the_payload_rather_than_being_dropped(break_it):
    """A batch we cannot draw is an obstacle we cannot show, not a batch to skip."""
    from f1sim.viewer.console.session import PropPayloadError
    # a fresh build per case: some of these mutate the arrays in place
    bad = dict(prop_batches(FakeTrack([a_box()]))[0])
    bad["pos"] = bad["pos"].copy()
    bad["nrm"] = bad["nrm"].copy()
    bad["col"] = bad["col"].copy()
    break_it(bad)
    with pytest.raises(PropPayloadError):
        _prop_batches([bad])
    good = prop_batches(FakeTrack([a_box()]))[0]
    # and a good batch alongside a bad one does not rescue it -- the map is still incomplete
    with pytest.raises(PropPayloadError):
        _prop_batches([dict(good), bad])


def test_no_props_stays_none_so_nothing_is_uploaded():
    assert _prop_batches(None) is None
    assert _prop_batches([]) is None, "a track that places nothing is valid, not an error"


def test_geometry_counts_and_sizes_include_the_props():
    sent = _prop_batches(prop_batches(FakeTrack([a_box(), a_box(x=2.0), a_box(style="steel_drum")])))
    bare = TrackGeometry(name="m", bounds=(0, 0, 1, 1), duct_height=0.2)
    with_props = TrackGeometry(name="m", bounds=(0, 0, 1, 1), duct_height=0.2, props=sent)
    assert bare.prop_counts() == (0, 0, 0)
    n_props, n_tris, n_batches = with_props.prop_counts()
    assert (n_props, n_batches) == (3, 2)
    assert n_tris > 0
    assert with_props.nbytes() > bare.nbytes(), "prop arrays must count towards the upload size"


# ----------------------------------------------------------------- catalogue
def test_the_props_group_is_offered_and_labelled_the_same_on_both_sides():
    from f1sim.viewer.console.catalog import GROUP_HINT, GROUP_ORDER
    from f1sim.learn import common
    label = "장애물 (상자·궤짝·드럼)"
    assert label in GROUP_ORDER and label in GROUP_HINT
    hint = GROUP_HINT[label]
    # The label must not suggest these are decoration: the car collides with them and the LiDAR
    # returns them, so a "보기 전용" style of wording would be actively misleading.
    assert "보기 전용" not in label and "보기 전용" not in hint
    assert "부딪" in hint and "LiDAR" in hint, (
        "the hint has to say the obstacles are real to the simulation")
    # The group is built over the eval maps rather than a hand-picked few, so every kind the
    # `+props` suffix supports is reachable. (An earlier version asserted against a
    # `PROP_SHOWCASE_MAPS` constant that no longer exists, because the demo list it named was
    # replaced by the eval set.)
    kinds = {n.split(":")[0] for n in common.EVAL_TRACKS if ":" in n}
    assert {"real", "rt", "gen"} <= kinds


@pytest.mark.parametrize("name", ["real:korea_2026_competition", "rt:Monza", "gen:competition:0"])
def test_the_offered_names_are_ones_the_loader_understands(name):
    """The catalogue must not offer a name that cannot be loaded."""
    from f1sim.learn import common
    tracks, _ = common.load_tracks([f"{name}+props3"], racelines=False, drop_infeasible=False)
    assert tracks and len(tracks[0].props) > 0, f"{name}+props3 loaded but placed nothing"


# ------------------------------------------------- the lifecycle, not just the validator
#
# PM's point: rejecting the payload is only half the job. If the console kept RUNNING after refusing
# to draw the props, the car would be on a track whose obstacles are in the simulation and not on
# the screen -- which is the state the whole design exists to prevent. So the refusal has to end the
# session through the paths that already end sessions.

@pytest.fixture(scope="module")
def qapp():
    # Scoped: `os.environ` is process-global and a module fixture that sets it without restoring
    # leaks the platform into every later test AND into any child process they spawn. That is not
    # hypothetical -- `test_console_launch_smoke` builds its child env from `os.environ`, so an
    # offscreen platform left here made the console open with no X window and its `xdotool` search
    # find nothing. Restored on teardown.
    _prev_qpa = os.environ.get("QT_QPA_PLATFORM")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from f1sim.viewer.console import app as A
    yield QtWidgets.QApplication.instance() or A.create_app(["test"])
    if _prev_qpa is None:
        os.environ.pop("QT_QPA_PLATFORM", None)
    else:
        os.environ["QT_QPA_PLATFORM"] = _prev_qpa


def _console(qapp):
    from f1sim.viewer.console.session import SessionController
    from f1sim.viewer.console.window import ConsoleWindow
    win = ConsoleWindow()
    win.resize(900, 600)
    ctl = SessionController(win)
    return win, ctl


def _geometry_msg(props):
    return {"name": "test:props", "bounds": (0.0, 0.0, 10.0, 10.0), "duct_height": 0.2,
            "floor": None, "ducts": None, "walls": None, "centerline": None,
            "raceline_xy": None, "raceline_v": None, "build_ms": 1.0,
            "geometry_version": 2, "props": props}


def test_an_undrawable_prop_payload_refuses_the_session(qapp):
    from f1sim.viewer.console.protocol import STATE_FAILED
    win, ctl = _console(qapp)
    try:
        ctl._preparing_gen = 4                      # as if a start were in flight
        good = prop_batches(FakeTrack([a_box()]))[0]
        bad = dict(good)
        bad["idx"] = np.array([0, 1, 10 ** 6], np.int32)
        sent = []
        ctl.send = lambda kind, for_gen=None, **f: sent.append((kind, f.get("gen")))

        ctl._apply_geometry(_geometry_msg([bad]))

        assert win.state == STATE_FAILED, f"console stayed {win.state} with undrawable obstacles"
        assert ("stop", 4) in sent, "the worker was not told to stop the session"
        assert ctl._preparing_gen == -1 and ctl._active_gen == -1
        assert win.viewport.geometry_data is None and not win.viewport._pending_geometry, (
            "a map we refused must be neither drawn nor queued for drawing")
        text = (win.status_text.text() + " " + win.error_text()) if hasattr(win, "error_text") \
            else win.status_text.text()
        assert "장애물" in text or "지오메트리" in text
    finally:
        ctl.shutdown()
        win.viewport.teardown()
        win.close()


def test_a_good_prop_payload_is_uploaded_and_the_session_continues(qapp):
    """The control: the same path with valid props must not refuse anything."""
    from f1sim.viewer.console.protocol import STATE_FAILED
    win, ctl = _console(qapp)
    try:
        ctl._preparing_gen = 4
        sent = []
        ctl.send = lambda kind, for_gen=None, **f: sent.append((kind, f.get("gen")))
        batches = prop_batches(FakeTrack([a_box(), a_box(x=4.0, style="steel_drum")]))

        ctl._apply_geometry(_geometry_msg(batches))

        assert win.state != STATE_FAILED
        assert not any(k == "stop" for k, _ in sent)
        # `set_geometry` queues; the upload happens on the GL thread at the next paint, so what is
        # observable here is what was handed over, not what has been drawn yet.
        gd = win.viewport._pending_geometry or win.viewport.geometry_data
        assert gd is not None and gd.prop_counts()[0] == 2
        assert {b["material"] for b in gd.props} == {"rubber", "metal"}
    finally:
        ctl.shutdown()
        win.viewport.teardown()
        win.close()


def _bad_batch():
    b = dict(prop_batches(FakeTrack([a_box()]))[0])
    b["idx"] = np.array([0, 1, 10 ** 6], np.int32)
    return b


def test_the_rejected_generation_is_resolved_not_merely_forgotten(qapp):
    """The stop we ask for has to be consumed when it comes back.

    Clearing `_preparing_gen`/`_active_gen` first left `_on_stopped` failing its own generation
    guard, so the stop this method sends was ignored and neither the start nor the stop pending
    record was ever cleared -- a console in FAILED, still reporting a command the worker had
    already answered.
    """
    from f1sim.viewer.console.protocol import STATE_FAILED
    win, ctl = _console(qapp)
    try:
        win._on_start()                                  # stamps the window's start ledger
        ctl._preparing_gen = 4
        assert "start" in win._pending
        sent = []
        ctl.send = lambda kind, for_gen=None, **f: sent.append((kind, f.get("gen")))

        ctl._apply_geometry(_geometry_msg([_bad_batch()]))
        assert win.state == STATE_FAILED
        assert ("stop", 4) in sent
        assert "start" not in win._pending, "the start we will never get a ready for is still pending"

        # the worker answers our stop
        ctl._on_control({"kind": "stopped", "gen": 4})
        assert "stop" not in win._pending, "the stop was ignored, so its ledger never cleared"
        assert win.state == STATE_FAILED, "the explanation must survive the termination"
        assert "장애물" in win.error_text() if hasattr(win, "error_text") else True
    finally:
        ctl.shutdown()
        win.viewport.teardown()
        win.close()


def test_a_late_ready_for_a_rejected_generation_is_refused(qapp):
    """It raced our stop. Its map is one we refused to draw, so it must not come up."""
    from f1sim.viewer.console.protocol import STATE_FAILED, STATE_RUNNING
    win, ctl = _console(qapp)
    try:
        ctl._preparing_gen = 4
        ctl.send = lambda kind, for_gen=None, **f: None
        ctl._apply_geometry(_geometry_msg([_bad_batch()]))
        ctl._on_control({"kind": "ready", "gen": 4, "facts": {"gen": 4, "map": "test:props"}})
        assert win.state == STATE_FAILED, "a refused map came up anyway"
        assert ctl._active_gen != 4
    finally:
        ctl.shutdown()
        win.viewport.teardown()
        win.close()


def test_rejecting_one_generation_does_not_disturb_a_newer_start(qapp):
    """A newer session must be unaffected: its generation was never rejected."""
    from f1sim.viewer.console.protocol import STATE_FAILED
    win, ctl = _console(qapp)
    try:
        ctl._preparing_gen = 4
        ctl.send = lambda kind, for_gen=None, **f: None
        ctl._apply_geometry(_geometry_msg([_bad_batch()]))
        assert win.state == STATE_FAILED

        # the user starts something else; then generation 4's stop finally arrives
        win._on_start()
        ctl._preparing_gen = 5
        ctl._on_control({"kind": "stopped", "gen": 4})
        assert ctl._preparing_gen == 5, "the older rejection tore down the newer start"
        assert "start" in win._pending, "the newer start's ledger was cleared by an older stop"
        assert 4 not in ctl._rejected_gens, "the rejected generation was never consumed"
    finally:
        ctl.shutdown()
        win.viewport.teardown()
        win.close()
