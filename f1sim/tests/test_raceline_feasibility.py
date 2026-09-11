"""A teacher label the car cannot steer is worse than no label.

`min_curvature_raceline` caps curvature by re-weighting rows, which is a preference the solve can
lose. Of 118 catalog tracks two came out below the car's 0.74 m full-lock radius -- gen:hallway:1100
~rev at R = 0.27 m with 7.7 m of a 39.8 m lap unfollowable, real:blackbox2021_3~rev at R = 0.45 m --
and both were ~10 % longer than the same track driven the other way. Those two were also the
privileged teacher's worst tracks in the training set by a wide margin (12.1 and 5.5 collisions/km
against 0.18 over the set), so the students were being taught from a line no car could follow.
"""
import math

import numpy as np
import pytest

from f1sim import maps
from f1sim.params import Config
from f1sim.raceline import Raceline, _enforce_turn_radius, curvature
from f1sim.track import resample_closed

VEHICLE = Config().vehicle
R_FULL_LOCK = (VEHICLE.lf + VEHICLE.lr) / math.tan(VEHICLE.s_max)      # 0.742 m


def _peak_curvature(xy: np.ndarray) -> float:
    return float(np.abs(curvature(xy)).max())


def _smoothed_min_radius(xy: np.ndarray) -> float:
    """Minimum radius over ~0.5 m of arc, so grid noise is not read as a corner."""
    ds = np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1)
    k = np.abs(curvature(xy))
    w = max(3, int(0.5 / max(ds.mean(), 1e-9)))
    smoothed = np.convolve(np.r_[k[-w:], k, k[:w]], np.ones(w) / w, "same")[w:-w]
    return 1.0 / max(smoothed.max(), 1e-9)


def test_car_cannot_turn_tighter_than_the_cap_the_builder_targets():
    # The default kappa_max must stay inside what the steering can do, or the repair is pointless.
    assert 1.0 / 1.1 > R_FULL_LOCK


def test_enforce_turn_radius_leaves_a_feasible_line_alone():
    # A circle well inside the limit must come back unchanged.
    angle = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.stack([5 * np.cos(angle), 5 * np.sin(angle)], 1)
    repaired = _enforce_turn_radius(circle.copy(), kappa_max=1.1)
    np.testing.assert_allclose(repaired, circle)


def test_enforce_turn_radius_flattens_a_corner_the_car_cannot_take():
    # A rounded square with 0.25 m corners: far tighter than the car can steer.
    corners = np.array([[-4., -4.], [4., -4.], [4., 4.], [-4., 4.]])
    dense = resample_closed(np.concatenate([
        np.linspace(corners[i], corners[(i + 1) % 4], 200, endpoint=False) for i in range(4)]), 800)
    assert _peak_curvature(dense) > 1.1

    repaired = _enforce_turn_radius(dense, kappa_max=1.1)
    assert _peak_curvature(repaired) <= 1.1 + 1e-6
    # and it stays a closed loop of roughly the same size, not a collapsed circle
    assert 0.6 < np.ptp(repaired[:, 0]) / np.ptp(dense[:, 0]) < 1.05


def test_the_track_that_produced_an_unfollowable_line_is_now_followable():
    # gen:hallway:1100~rev is procedural, so this needs no map submodule.
    track = maps.load("gen:hallway:1100~rev")
    line = Raceline.build(track)                       # build, not build_cached: ignore stale CSVs

    assert _smoothed_min_radius(line.xy) > R_FULL_LOCK
    ds = np.linalg.norm(np.roll(line.xy, -1, 0) - line.xy, axis=1)
    k = np.abs(curvature(line.xy))
    w = max(3, int(0.5 / max(ds.mean(), 1e-9)))
    smoothed = np.convolve(np.r_[k[-w:], k, k[:w]], np.ones(w) / w, "same")[w:-w]
    assert float(ds[smoothed > 1.0 / R_FULL_LOCK].sum()) == 0.0      # was 7.70 m of a 39.8 m lap


def test_repaired_line_still_clears_the_body_and_matches_the_forward_direction():
    reverse = Raceline.build(maps.load("gen:hallway:1100~rev"))
    forward = Raceline.build(maps.load("gen:hallway:1100"))

    # Reversing the travel direction flips the sign of curvature; it must not change the curve.
    assert reverse.length == pytest.approx(forward.length, rel=0.05)   # was 39.8 vs 35.0 (+14 %)

    track = maps.load("gen:hallway:1100~rev")
    col = ((reverse.xy[:, 0] - track.origin[0]) / track.resolution).astype(int)
    row = ((reverse.xy[:, 1] - track.origin[1]) / track.resolution).astype(int)
    col = col.clip(0, track.occupancy.shape[1] - 1); row = row.clip(0, track.occupancy.shape[0] - 1)
    assert float((track.edt[row, col] - VEHICLE.width / 2).min()) > 0.0
