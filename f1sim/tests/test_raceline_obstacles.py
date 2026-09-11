"""`+rlobs` puts boxes on the racing line; `+obs` puts them against the wall.

The distinction matters for what the map tests. A box hugging the lane edge leaves the fast line
clear, so a policy that has learned one good line can ignore it. A box standing on the line forces a
deviation and back, which is what exercises seeing an obstacle and planning around it.
"""
import numpy as np
import pytest

from f1sim import maps
from f1sim.maps import _split_obstacle_suffix
from f1sim.raceline import Raceline
from f1sim.track import Track

BASE = "gen:competition:0"          # procedural: needs no map submodule


def _added_cells_world(base: Track, variant: Track) -> np.ndarray:
    rows, cols = np.nonzero(variant.occupancy & ~base.occupancy)
    return np.stack([cols * variant.resolution + variant.origin[0],
                     rows * variant.resolution + variant.origin[1]], 1)


def _distance_to_line(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    return np.sqrt(((points[:, None, :] - line[None, :, :]) ** 2).sum(-1)).min(1)


def test_suffix_parsing_prefers_the_longer_tag():
    # "+obs" is a substring of "+rlobs": splitting on it would cut "x+rlobs7" into "x+rl" and "7".
    assert _split_obstacle_suffix("korea+rlobs7") == ("korea", "rlobs", "7")
    assert _split_obstacle_suffix("korea+obs7") == ("korea", "obs", "7")
    assert _split_obstacle_suffix("korea") == ("korea", None, None)


def test_rlobs_boxes_sit_on_the_racing_line_and_obs_boxes_do_not():
    base = maps.load(BASE)
    line = Raceline.build_cached(base).xy

    on_line = _distance_to_line(_added_cells_world(base, maps.load(BASE + "+rlobs1")), line)
    at_wall = _distance_to_line(_added_cells_world(base, maps.load(BASE + "+obs1")), line)

    assert on_line.size and at_wall.size
    assert np.median(on_line) < 0.35                    # a box the car has to go around
    assert np.median(at_wall) > np.median(on_line) * 3  # the wall-hugging placement is far off-line


def _widest_free_run(track: Track, centre: np.ndarray, normal: np.ndarray, span: float = 4.0) -> float:
    """Longest uninterrupted free stretch across the lane at this station.

    Marching outwards from a single point is not enough: when the box straddles that point both
    directions report zero even though the lane is open on one side of it."""
    step = track.resolution
    offsets = np.arange(-span, span + step, step)
    height, width = track.occupancy.shape
    free = []
    for offset in offsets:
        q = centre + normal * offset
        col = int(round((q[0] - track.origin[0]) / track.resolution))
        row = int(round((q[1] - track.origin[1]) / track.resolution))
        free.append(0 <= row < height and 0 <= col < width and not track.occupancy[row, col])
    best = run = 0
    for is_free in free:
        run = run + 1 if is_free else 0
        best = max(best, run)
    return best * step


def test_every_rlobs_obstacle_leaves_a_passable_gap():
    base = maps.load(BASE)
    variant = maps.load(BASE + "+rlobs3")
    added = _added_cells_world(base, variant)
    assert added.size

    centerline = base.centerline
    car_width = 0.31
    for point in added[:: max(1, len(added) // 40)]:
        i = int(((centerline - point) ** 2).sum(1).argmin())
        tangent = centerline[(i + 1) % len(centerline)] - centerline[i - 1]
        tangent /= np.linalg.norm(tangent) + 1e-9
        normal = np.array([-tangent[1], tangent[0]])
        gap = _widest_free_run(variant, centerline[i], normal)
        assert gap > car_width, f"lane sealed at {centerline[i]}: widest free run {gap:.2f} m"


def test_different_seeds_place_different_obstacles():
    base = maps.load(BASE)
    a = maps.load(BASE + "+rlobs1").occupancy
    b = maps.load(BASE + "+rlobs2").occupancy
    assert not np.array_equal(a, b)
    assert (a & ~base.occupancy).sum() > 0 and (b & ~base.occupancy).sum() > 0


def test_rlobs_keeps_the_centerline_and_grid_of_the_base_map():
    base = maps.load(BASE)
    variant = maps.load(BASE + "+rlobs1")
    assert variant.resolution == base.resolution and variant.origin == base.origin
    np.testing.assert_allclose(variant.centerline, base.centerline)
    assert variant.occupancy.sum() > base.occupancy.sum()          # strictly adds obstacles
    assert bool((variant.tall & ~base.tall).any())                 # boxes are tall: beams do not pass over


def test_plain_names_are_unaffected():
    base = maps.load(BASE)
    again = maps.load(BASE)
    assert np.array_equal(base.occupancy, again.occupancy)


@pytest.mark.parametrize("direction", ["~rev", "~mir"])
def test_direction_modifiers_still_apply_on_top(direction):
    variant = maps.load(BASE + "+rlobs1" + direction)
    plain = maps.load(BASE + "+rlobs1")
    assert variant.occupancy.shape == plain.occupancy.shape
    assert variant.occupancy.sum() == plain.occupancy.sum()
