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
