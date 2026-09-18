import numpy as np
import pytest
from scipy import ndimage

from f1sim.track import Track
from f1sim.planning_seed import obstacle_aware_seed, _densify


def loop(obstacle=None):
    occ = np.zeros((120, 160), bool)
    occ[[0, -1], :] = True
    occ[:, [0, -1]] = True
    # Infield guarantees the local router cannot replace the loop with its chord.
    occ[45:75, 50:110] = True
    if obstacle is not None:
        occ[obstacle] = True
    line = np.array([[1., 1.5], [7., 1.5], [7., 4.5], [1., 4.5]])
    return Track.from_occupancy(occ, .05, centerline=line)


def assert_safe(track, result, clearance=.2):
    q = _densify(result, .01)
    cells = np.floor(q / track.resolution).astype(int)
    dist = ndimage.distance_transform_edt(~track.occupancy) * track.resolution
    assert dist[cells[:, 1], cells[:, 0]].min() > clearance
    # The original clockwise lap still surrounds its infield.
    assert abs(np.sum(result[:, 0] * np.roll(result[:, 1], -1) - result[:, 1] * np.roll(result[:, 0], -1))) > 20


def test_clear_seed_is_exactly_unchanged():
    track = loop()
    result = obstacle_aware_seed(track, clearance=.2)
    np.testing.assert_array_equal(result, track.centerline)


def test_thin_obstacle_between_clear_vertices_has_two_passes():
    track = loop((slice(26, 35), slice(79, 81)))
    assert not track.occupancy[30, 20] and not track.occupancy[30, 140]
    result = obstacle_aware_seed(track, clearance=.2)
    assert_safe(track, result)
    # Opposite unblocked side of the loop remains in the route.
    assert np.max(result[:, 1]) == 4.5


def test_obstacle_across_closed_seam():
    track = loop((slice(59, 61), slice(15, 26)))
    result = obstacle_aware_seed(track, clearance=.2)
    assert_safe(track, result)


def test_blocked_corridor_is_rejected_instead_of_shortcut():
    track = loop((slice(0, 46), slice(79, 81)))
    with pytest.raises(ValueError, match='blocked corridor'):
        obstacle_aware_seed(track, clearance=.2)
