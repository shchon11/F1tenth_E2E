"""Boot-camp environment pieces (2026-10-04): warped venues and opponent seats that may stay empty."""
import numpy as np
import pytest
import torch
from scipy import ndimage

from f1sim import Config, Simulator, Track
from f1sim import dynamics as dyn
from f1sim.opponent_slots import OpponentSlot


def ring(size=24.0, res=0.1, r_in=5.0, r_out=8.0):
    n = int(size / res)
    yy, xx = np.mgrid[0:n, 0:n] * res - size / 2
    rr = np.hypot(xx, yy)
    occ = (rr < r_in) | (rr > r_out)
    th = np.linspace(0, 2 * np.pi, 300, endpoint=False)
    cl = np.stack([6.5 * np.cos(th), 6.5 * np.sin(th)], 1)
    return Track.from_occupancy(occ, res, (-size / 2, -size / 2), cl, "ring")


def test_warp_is_deterministic_keeps_one_lane_and_its_width_floor():
    t = ring()
    a, b = t.with_warp(3), t.with_warp(3)
    assert np.array_equal(a.occupancy, b.occupancy) and a.resolution == b.resolution
    assert a.warp_edits == b.warp_edits
    for seed in range(6):
        w = t.with_warp(seed)
        assert 0.95 * t.resolution <= w.resolution <= 1.05 * t.resolution
        cl = w.centerline
        j = ((cl[:, 0] - w.origin[0]) / w.resolution).astype(int)
        i = ((cl[:, 1] - w.origin[1]) / w.resolution).astype(int)
        lab, _ = ndimage.label(~w.occupancy)
        assert len(set(lab[i, j].tolist())) == 1                      # the lap is still one lane
        drive = lab == lab[i[0], j[0]]
        half = ndimage.distance_transform_edt(drive) * w.resolution
        assert half[i, j].min() >= 0.55 - w.resolution                # never narrower than the floor
        assert not drive[0, :].any() and not drive[-1, :].any()      # the outside never opens


def test_slot_present_round_trips_and_is_validated():
    s = OpponentSlot.from_dict({"kind": "forzaeth", "present": 0.4})
    assert s.present == 0.4 and OpponentSlot.from_dict(s.to_dict()).present == 0.4
    assert "present" not in OpponentSlot.from_dict({"kind": "forzaeth"}).to_dict()
    with pytest.raises(ValueError):
        OpponentSlot.from_dict({"kind": "forzaeth", "present": 1.5})


def test_a_parked_car_stays_put_and_touches_nothing():
    n = 300
    occ = np.zeros((n, n), dtype=bool); occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    track = Track.from_occupancy(occ, 0.2, (-30.0, -30.0))
    cfg = Config(); cfg.rand.enabled = False
    sim = Simulator(track, cfg, num_envs=2, device="cpu", race_size=2)
    sim.reset(poses=torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]))       # overlapping: in contact
    sim.step(torch.tensor([[0.0, 2.0], [0.0, 2.0]]))
    assert bool(sim.car_collision.any())
    sim.park(torch.tensor([1]))
    before = sim.state[1].clone()
    for _ in range(20):
        r = sim.step(torch.tensor([[0.0, 2.0], [0.3, 5.0]]))
    assert torch.equal(sim.state[1], before)                          # frozen where it was parked
    assert float(before[0]) < -900                                     # far off the map
    assert not bool(r.car_collision.any()) and not bool(r.collision[1])
    sim.unpark(torch.tensor([1]))
    assert not bool(sim.absent.any()) and not sim.absent_any
