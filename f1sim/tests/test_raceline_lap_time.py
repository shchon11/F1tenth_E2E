"""The planning profile must obey the same finite grip and motor budget on every lap edge."""
import numpy as np
import pytest

from f1sim.params import VehicleParams
from f1sim.raceline import Raceline, _refine_lap_time, curvature, speed_profile
from f1sim.track import Track
from f1sim.track import resample_closed


def _oval(n=160):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([8 * np.cos(theta), 3 * np.sin(theta)])


def _annulus(n=120):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    center = 6 * np.column_stack([np.cos(theta), np.sin(theta)])
    xx, yy = np.meshgrid(np.arange(-9, 9, .05), np.arange(-9, 9, .05))
    radius = np.hypot(xx, yy)
    return Track.from_occupancy((radius <= 4) | (radius >= 8), .05,
                                origin=(-9., -9.), centerline=center, name="annulus")


def _assert_physical_profile(xy, v, mu, vehicle):
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    ax = (np.roll(v * v, -1) - v * v) / (2 * ds)
    ay = v * v * np.abs(curvature(xy))
    wheelbase = vehicle.lf + vehicle.lr
    for share, load, transfer, grip in [
        (1 - vehicle.drive_split_r, vehicle.lr / wheelbase, -vehicle.h / wheelbase, mu * vehicle.mu_f_scale),
        (vehicle.drive_split_r, vehicle.lf / wheelbase, vehicle.h / wheelbase, mu * vehicle.mu_r_scale),
    ]:
        normal = 9.81 * load + transfer * ax
        assert np.min(normal) >= 0
        for lateral, speed in [(ay, v), (np.roll(ay, -1), np.roll(v, -1))]:
            tire_ax = ax + vehicle.c_roll + vehicle.c_drag * speed ** 2
            ratio = ((share * tire_ax) ** 2 + (load * lateral) ** 2) / (grip * normal) ** 2
            assert float(ratio.max()) <= 1 + 2e-6
    edge_v = np.maximum(v, np.roll(v, -1))
    power = vehicle.a_max * np.minimum(1, vehicle.v_switch / np.maximum(edge_v, 1e-9))
    assert np.max(ax + vehicle.c_roll + vehicle.c_drag * edge_v ** 2 - power) < 2e-6
    low_v = np.minimum(v, np.roll(v, -1))
    assert np.min(ax + vehicle.c_roll + vehicle.c_drag * low_v ** 2) >= -vehicle.a_brake - 2e-6
    steer = np.arctan(wheelbase * curvature(xy))
    slew = np.abs(np.roll(steer, -1) - steer) * (v + np.roll(v, -1)) / (2 * ds)
    assert np.max(slew) <= vehicle.sv_max + 1e-6


@pytest.mark.parametrize("mu", [0.20, 0.7, 1.0489])
def test_cyclic_profile_respects_both_axles_and_actuator_limits(mu):
    xy = _oval()
    vehicle = VehicleParams()
    v = speed_profile(xy, v_max=12, mu=mu, vehicle=vehicle)
    _assert_physical_profile(xy, v, mu, vehicle)
    assert np.all(v > 0)


def test_speed_floor_never_overrides_friction_or_speed_cap():
    theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    xy = .2 * np.column_stack([np.cos(theta), np.sin(theta)])
    v = speed_profile(xy, v_max=.3, a_lat=.1, v_min=1.)
    assert v.max() <= .3
    assert np.max(v * v * np.abs(curvature(xy))) <= .1 + 1e-8


def test_profile_is_independent_of_where_the_closed_lap_starts():
    xy = _oval()
    v = speed_profile(xy, mu=.7)
    shifted = speed_profile(np.roll(xy, 53, axis=0), mu=.7)
    np.testing.assert_allclose(np.roll(v, 53), shifted, rtol=0, atol=1e-7)


def test_motor_cap_is_not_multiplied_by_unused_tyre_fraction():
    # At high grip the engine, rather than the tire circle, limits most accelerating edges.
    xy = _oval(320)
    v = speed_profile(xy, v_max=12, a_acc=1., mu=2.)
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    ax = (np.roll(v * v, -1) - v * v) / (2 * ds)
    vehicle = VehicleParams()
    motor_demand = ax + vehicle.c_roll + vehicle.c_drag * np.maximum(v, np.roll(v, -1)) ** 2
    assert np.count_nonzero(np.isclose(motor_demand, 1., atol=1e-6)) > 10


def test_lap_time_uses_constant_acceleration_segment_time():
    xy = np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]])
    v = np.array([1., 3., 2., 4.])
    expected = np.sum(4 / (v + np.roll(v, -1)))
    assert Raceline.from_xy(xy, v).lap_time == pytest.approx(expected)


def test_long_straights_reach_the_cap_and_brake_before_the_corner():
    from scipy.ndimage import gaussian_filter1d
    arc = np.linspace(-np.pi / 2, np.pi / 2, 120, endpoint=False)
    xy = np.vstack([
        np.column_stack([15 + 3 * np.cos(arc), 3 * np.sin(arc)]),
        np.column_stack([np.linspace(15, -15, 240, endpoint=False), np.full(240, 3.)]),
        np.column_stack([-15 - 3 * np.cos(arc), -3 * np.sin(arc)]),
        np.column_stack([np.linspace(-15, 15, 240, endpoint=False), np.full(240, -3.)]),
    ])
    xy = gaussian_filter1d(resample_closed(xy, 640), 3., axis=0, mode="wrap")
    v = speed_profile(xy, v_max=9., mu=1.0489)
    middle_straight = (np.abs(xy[:, 0]) < 6) & (np.abs(xy[:, 1]) > 2.99)
    assert np.max(v[middle_straight]) == pytest.approx(9., abs=1e-6)
    assert np.max(v[np.abs(xy[:, 0]) > 16.5]) < 6
    _assert_physical_profile(xy, v, 1.0489, VehicleParams())


def test_rear_drive_and_absolute_axle_coefficients_are_respected():
    vehicle = VehicleParams(drive_split_r=1., mu_f_scale=.65, mu_r_scale=.8)
    xy = _oval()
    v = speed_profile(xy, mu=1., mu_front=.65, mu_rear=.8, vehicle=vehicle)
    _assert_physical_profile(xy, v, 1., vehicle)


def test_explicit_caps_cannot_raise_the_vehicle_limits():
    vehicle = VehicleParams(v_max=6., a_max=1.5, a_brake=1.2)
    xy = _oval()
    v = speed_profile(xy, v_max=50., a_lat=40., a_acc=50., a_brake=50.,
                      mu=2., vehicle=vehicle)
    assert np.max(v) <= vehicle.v_max
    _assert_physical_profile(xy, v, 2., vehicle)


def test_lap_time_refinement_takes_a_shorter_smooth_feasible_circle():
    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
    seed = 6 * np.column_stack([np.cos(theta), np.sin(theta)])
    resolution = .05
    xx, yy = np.meshgrid(np.arange(-9, 9, resolution), np.arange(-9, 9, resolution))
    radius = np.hypot(xx, yy)
    track = Track.from_occupancy((radius <= 4) | (radius >= 8), resolution,
                                 origin=(-9., -9.), centerline=seed, name="annulus")
    kwargs = dict(mu=1.0489)
    line = _refine_lap_time(seed, track, .31, .25, kwargs)
    before = Raceline.from_xy(seed, speed_profile(seed, **kwargs))
    after = Raceline.from_xy(line, speed_profile(line, **kwargs))
    assert after.lap_time < .99 * before.lap_time
    assert np.abs(after.kappa).max() <= 1.1
    assert np.min(after.kappa) > 0             # no spurious S wave on a constant turn
    assert np.min(np.linalg.norm(line, axis=1)) > 4 + .31 / 2 + .25 - .02
    _assert_physical_profile(line, after.v, 1.0489, VehicleParams())


def test_infeasible_seed_is_not_returned_when_refinement_budget_is_exhausted():
    theta = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    seed = 4.1 * np.column_stack([np.cos(theta), np.sin(theta)])
    xx, yy = np.meshgrid(np.arange(-9, 9, .05), np.arange(-9, 9, .05))
    radius = np.hypot(xx, yy)
    track = Track.from_occupancy((radius <= 4) | (radius >= 8), .05,
                                 origin=(-9., -9.), centerline=seed, name="blocked-seed")
    with pytest.raises(ValueError, match="no raceline satisfies"):
        _refine_lap_time(seed, track, .31, .25, {}, max_evaluations=0)


def test_same_geometry_and_cache_key_when_wall_clock_advances_during_solves(monkeypatch, tmp_path):
    import time
    track = _annulus()
    normal_cache, busy_cache = tmp_path / "normal", tmp_path / "busy"
    normal = Raceline.build_cached(track, cache_dir=str(normal_cache))
    tick = 0.
    def busy_clock():
        nonlocal tick
        tick += 100.
        return tick
    # Simulate CPU descheduling between solver calls, without a slow/flaky real sleep.
    with monkeypatch.context() as delayed:
        delayed.setattr(time, "monotonic", busy_clock)
        busy = Raceline.build_cached(track, cache_dir=str(busy_cache))
    np.testing.assert_array_equal(normal.xy, busy.xy)
    np.testing.assert_array_equal(normal.v, busy.v)
    assert next(normal_cache.glob("*.csv")).name == next(busy_cache.glob("*.csv")).name


@pytest.mark.parametrize("stage", ["seed", "refinement"])
def test_explicit_deadline_aborts_without_caching_partial_geometry(monkeypatch, tmp_path, stage):
    import f1sim.raceline as module
    function_name = "min_curvature_raceline" if stage == "seed" else "_refine_lap_time"
    original = getattr(module, function_name)
    def expired(*args, **kwargs):
        kwargs["_max_seconds" if stage == "seed" else "max_seconds"] = 0.
        return original(*args, **kwargs)
    monkeypatch.setattr(module, function_name, expired)
    with pytest.raises(TimeoutError, match="partial geometry discarded"):
        Raceline.build_cached(_annulus(), cache_dir=str(tmp_path))
    assert not list(tmp_path.glob("*.csv"))
