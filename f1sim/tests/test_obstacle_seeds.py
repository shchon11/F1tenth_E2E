import numpy as np

from f1sim import maps
from f1sim.track import Track


def test_procedural_obstacle_seed_changes_placement_not_track() -> None:
    first = maps.load("gen:competition:0+obs1")
    other = maps.load("gen:competition:0+obs99")
    repeated = maps.load("gen:competition:0+obs1")
    np.testing.assert_array_equal(first.centerline, other.centerline)
    np.testing.assert_array_equal(first.occupancy, repeated.occupancy)
    assert not np.array_equal(first.occupancy, other.occupancy)


def test_boolean_obstacles_request_random_count_not_one() -> None:
    base = Track.generate_random(0)
    expected = base.with_lane_obstacles(seed=0, n=4)
    generated = Track.generate_random(0, lane_obstacles=True)
    np.testing.assert_array_equal(generated.occupancy, expected.occupancy)


def test_explicit_obstacle_count_remains_exact_request() -> None:
    base = Track.generate_random(0)
    expected = base.with_lane_obstacles(seed=0, n=2)
    generated = Track.generate_random(0, lane_obstacles=2)
    np.testing.assert_array_equal(generated.occupancy, expected.occupancy)
