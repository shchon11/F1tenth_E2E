"""Static analytic props must be visible to offline teacher planning."""
from collections import OrderedDict

import numpy as np
import pytest

from f1sim.track import StaticProp, Track
from f1sim import props


def room():
    occ = np.zeros((100, 100), dtype=bool)
    occ[[0, -1], :] = True
    occ[:, [0, -1]] = True
    return Track.from_occupancy(occ, 0.05, centerline=np.array([[1., 1.], [4., 1.], [4., 4.], [1., 4.]]))


def test_authored_prop_projection_preserves_simulation_geometry():
    track = room()
    track.props = (StaticProp('cardboard_box', 2.0, 2.0, yaw=0.7),)
    original = track.occupancy.copy()
    planned = track.for_planning()
    assert not track.occupancy[40, 40]
    assert planned.occupancy[40, 40]
    np.testing.assert_array_equal(track.occupancy, original)
    assert len(track.props) == 1 and planned.props == ()
    assert planned.for_planning() is planned
    assert track.bare().for_planning().occupancy[40, 40] == False


@pytest.mark.parametrize('style', props.STYLES)
def test_every_collision_band_vertex_is_covered(style):
    track = room()
    placement = StaticProp(style, 2.5, 2.5, yaw=0.63)
    track.props = (placement,)
    planned = track.for_planning()
    c, s = np.cos(placement.yaw), np.sin(placement.yaw)
    rot = np.array([[c, -s], [s, c]])
    for band in props.sections(placement.build(), n_bands=4):
        if band['z0'] >= 0.27:
            continue
        polygon = np.asarray(band['polygon']) @ rot.T + [placement.x, placement.y]
        cells = np.floor(polygon / track.resolution).astype(int)
        assert planned.occupancy[cells[:, 1], cells[:, 0]].all()


def test_thin_prop_between_cell_centers_is_not_lost():
    track = room()
    track.props = (StaticProp('marker_post', 2.0, 2.0),)
    assert track.for_planning().occupancy[39:41, 39:41].all()


def test_clearance_includes_authored_prop():
    from f1sim.learn.common import raceline_clearance
    from types import SimpleNamespace
    track = room()
    line = SimpleNamespace(xy=np.array([[2., 2.]]))
    assert raceline_clearance(track, line) > 1
    track.props = (StaticProp('cardboard_box', 2., 2.),)
    assert raceline_clearance(track, line) == 0


@pytest.mark.parametrize('name', ['scene:hall', 'scene/hall'])
def test_viewer_reloads_edited_scene_in_both_grammars(monkeypatch, name):
    from f1sim.viewer.sim_worker import SimWorker
    from f1sim.learn import common
    worker = SimWorker.__new__(SimWorker)
    before, after = object(), object()
    worker._track_cache = OrderedDict([(name, before)])
    monkeypatch.setattr(common, 'load_tracks', lambda *a, **kw: ([after], None))
    assert worker.load_track(name) is after


def test_planning_cache_changes_when_analytic_prop_moves(tmp_path, monkeypatch):
    from f1sim.raceline import Raceline
    track = room()
    track.props = (StaticProp('cardboard_box', 2., 2.),)
    built = []
    def build(track, **kw):
        built.append(track)
        return Raceline.from_xy(track.centerline, np.ones(len(track.centerline)))
    monkeypatch.setattr(Raceline, 'build', staticmethod(build))
    # Signature is part of cache metadata, so retain its vehicle default.
    import inspect
    build.__signature__ = inspect.Signature([
        inspect.Parameter('track', inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter('vehicle', inspect.Parameter.KEYWORD_ONLY, default=None),
    ])
    Raceline.build_cached(track, cache_dir=str(tmp_path))
    Raceline.build_cached(track, cache_dir=str(tmp_path))
    track.props = (StaticProp('cardboard_box', 3., 2.),)
    Raceline.build_cached(track, cache_dir=str(tmp_path))
    assert len(built) == 2
    assert not np.array_equal(built[0].occupancy, built[1].occupancy)


def test_default_authored_scene_planning_sees_props_bare_does_not(tmp_path, monkeypatch):
    from f1sim import maps
    from f1sim.scene import SceneDoc
    monkeypatch.setenv('F1SIM_SCENES', str(tmp_path))
    scene = SceneDoc.new_blank('planner_props', 5.0, 5.0, resolution=0.05)
    scene.paint_border('tall', 0.1)
    scene.centerline = room().centerline
    scene.add_prop('cardboard_box', 2.0, 2.0, yaw=0.3)
    scene.save()
    authored = maps.load('scene:planner_props')
    bare = maps.load('scene:planner_props+bare')
    np.testing.assert_array_equal(authored.occupancy, bare.occupancy)
    assert authored.for_planning().occupancy[40, 40]
    assert not bare.for_planning().occupancy[40, 40]
