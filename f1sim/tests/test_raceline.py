import numpy as np
from scipy import ndimage
from f1sim import maps
from f1sim.raceline import Raceline, curvature


def _clearance(t, xy):
    rc = np.stack([(xy[:, 1] - t.origin[1]) / t.resolution, (xy[:, 0] - t.origin[0]) / t.resolution])
    return ndimage.map_coordinates(t.edt, rc, order=1, mode="nearest")


def _turn_angles(xy):
    d = np.roll(xy, -1, 0) - xy
    h = np.arctan2(d[:, 1], d[:, 0])
    return np.abs(np.angle(np.exp(1j * (np.roll(h, -1) - h))))


def test_raceline_smooth_faster_and_clear():
    for name in ["gen:competition:0", "real:blackbox2021_1", "real:icra2022"]:
        t = maps.load(name)
        rl = Raceline.build(t)
        c = t.centerline
        # no kinks and nothing tighter than the car can steer (min turning radius ~0.74 m)
        assert _turn_angles(rl.xy).max() < np.deg2rad(8), name
        assert np.abs(rl.kappa).max() < 1.3, name
        # the line is smoother than the centerline in the corners and never closer than the margin
        kc = np.abs(curvature(c)); kr = np.abs(rl.kappa)
        assert np.percentile(kr, 99) < np.percentile(kc, 99), name
        assert _clearance(t, rl.xy).min() > 0.31 / 2 + 0.25 - 0.03, name
        # and quicker to drive (kinematic estimate) than the centerline at the same speed limits
        from f1sim.raceline import speed_profile
        vc = speed_profile(c); tc = float((np.linalg.norm(np.roll(c, -1, 0) - c, axis=1) / vc).sum())
        assert rl.lap_time < tc, (name, rl.lap_time, tc)


def _lap(xy, lim):
    from f1sim.raceline import speed_profile
    return float((np.linalg.norm(np.roll(xy, -1, 0) - xy, axis=1) / speed_profile(xy, *lim)).sum())


def test_batched_lap_time_is_the_speed_profile():
    # the min-time descent optimises `_lap_times`; it is only the teacher's lap time if it is the
    # same recursion as `speed_profile`, at the default limits and at the ones the car measured
    from f1sim.raceline import _lap_times
    xy = Raceline.build_cached(maps.load("real:map12x16")).xy
    for lim in [(10.0, 6.0, 6.0, 3.0), (10.0, 9.0, 7.0, 5.0)]:
        got = _lap_times(np.stack([xy, xy[::-1]]), *lim)
        assert abs(got[0] - _lap(xy, lim)) < 1e-9 and abs(got[1] - _lap(xy[::-1], lim)) < 1e-9, lim


def test_min_time_line_is_quicker_clear_and_steerable():
    from f1sim.raceline import min_time_raceline
    t = maps.load("real:map12x16")
    lim = (10.0, 9.0, 7.0, 5.0)
    start = Raceline.build_cached(t).xy
    kw = dict(v_max=lim[0], a_lat=lim[1], a_acc=lim[2], a_brake=lim[3])
    out = min_time_raceline(t, start, iters=3, inner_iters=15, knot_spacing=(2.0,), **kw)
    assert len(out) == len(start)
    assert _lap(out, lim) < 0.99 * _lap(start, lim)            # measured: 3 % from the coarse stage alone
    assert _clearance(t, out).min() > _clearance(t, start).min() - 0.03
    assert np.abs(curvature(out)).max() < 1.3 and _turn_angles(out).max() < np.deg2rad(8)
    # and with no budget to move, the line it was given comes back: never a slower one
    assert min_time_raceline(t, start, iters=0, **kw) is start


def test_default_raceline_cache_key_predates_the_objective(tmp_path):
    # Every cached line, obstacle placement and frozen suite was keyed before `objective` existed.
    # The default must still produce that key, or every catalogue track is silently rebuilt.
    import hashlib, inspect, os
    t = maps.load("real:map12x16")
    legacy = {k: v.default for k, v in inspect.signature(Raceline.build).parameters.items()
              if k not in ("track", "objective")}
    cl = np.asarray(t.centerline, dtype=np.float32).tobytes()
    h = hashlib.md5(np.packbits(t.occupancy).tobytes() + cl + repr(sorted(legacy.items())).encode() + b"rl5").hexdigest()[:12]
    Raceline.build_cached(t, cache_dir=str(tmp_path))
    Raceline.build_cached(t, cache_dir=str(tmp_path), objective="min_curvature")
    assert os.listdir(tmp_path) == [f"{t.name}_{h}.csv"]


def test_suspension_noise_knobs_do_not_move_the_cache_key():
    # `road_tilt_v` was added after the key was defined and `road_tilt`'s default went 0.017 -> 0 on
    # 2026-09-21. The line reads neither; either one reaching the key would rebuild all 738 cached
    # lines -- twenty to seventy minutes apiece -- on the next console session or training start.
    from f1sim.params import VehicleParams
    from f1sim.raceline import _LineKeyVehicle
    key = repr(_LineKeyVehicle(VehicleParams()))
    assert "road_tilt=0.017" in key and "road_tilt_v" not in key
    assert key == repr(_LineKeyVehicle(VehicleParams(road_tilt=0.017)))
    assert key == repr(_LineKeyVehicle(VehicleParams(road_tilt=0.0, road_tilt_v=5.0)))
