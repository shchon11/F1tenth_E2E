"""The 2026-09-28 realism changes do what they say: stiffness that does not follow mu, car-car
contact boxes the size the LiDAR draws, and wall contacts that push where the car touches."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy import ndimage

from f1sim import dynamics as dyn
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.params import Config
from f1sim.track import Track


def _open_floor():
    res, n = 0.1, 300
    occ = np.zeros((n, n), bool); occ[:2] = occ[-2:] = True; occ[:, :2] = occ[:, -2:] = True
    edt = ndimage.distance_transform_edt(~occ).astype(np.float32) * res
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    return Track(occupancy=occ, resolution=res, origin=(0.0, 0.0), edt=edt,
                 centerline=np.stack([15 + 10 * np.cos(th), 15 + 10 * np.sin(th)], 1), name="open",
                 duct=occ.copy(), tall=np.zeros_like(occ))


def _sim(B=2, race_size=1, **sim_kw):
    cfg = Config(); cfg.sim.compile = False; cfg.rand.enabled = False; cfg.sim.terminate_on_collision = False
    for k, v in sim_kw.items():
        setattr(cfg.sim, k, v)
    env = common.make_env([_open_floor()], B, "cpu", EnvConfig(action_mode="direct", race_size=race_size,
                                                                collision_mode="soft"), cfg=cfg, seed=1)
    env.reset(seed=1)
    return env.sim


def test_cornering_stiffness_does_not_follow_mu_unless_asked():
    sim = _sim(B=3)
    P = dict(sim.P)
    P["mu"] = torch.tensor([0.7, float(P["mu_ref"][0]), 1.1])
    bf, br = dyn.lateral_B(P)
    stiff = P["mu"] * bf
    assert torch.allclose(stiff, stiff[1].expand(3), rtol=1e-5)          # mu * B held at the mu_ref value
    assert abs(float(bf[1]) - float(P["B_f"][1])) < 1e-6                  # untouched at mu_ref
    P["mu_stiffness_exp"] = torch.ones(3)                                 # the formula's own scaling
    bf1, _ = dyn.lateral_B(P)
    assert torch.allclose(bf1, P["B_f"])


def test_car_contact_box_is_the_size_the_other_cars_lidar_draws():
    sim = _sim(B=2, race_size=2)
    # two cars nose to tail on the same line, 0.56 m between centres
    pose = torch.tensor([[15.0, 15.0, 0.0], [15.56, 15.0, 0.0]])
    sim.reset(torch.arange(2), pose, torch.zeros(2))
    sim.car_rear[:, 0] = 0.0                                              # no rear box: bodies only
    sim.car_dims[:, 0] = 0.58                                             # drawn 0.58 long: 0.336 half-length -> overlap
    assert bool(sim._car_contacts(sim.state).all())
    sim.car_dims[:, 0] = 0.45                                             # drawn 0.45 long: 0.261 half-length -> clear
    assert not bool(sim._car_contacts(sim.state).any())


def _hit(yaw, **kw):
    """One car driven into the bottom wall at 4 m/s, returns its yaw rate right after contact."""
    sim = _sim(B=1, **kw)
    sim.reset(torch.arange(1), torch.tensor([[15.0, 0.75, yaw]]), torch.tensor([4.0]))
    out = []
    for _ in range(8):
        sim.step(torch.tensor([[0.0, 4.0]]))
        out.append(float(sim.state[0, dyn.IR]))
    return out


def test_a_square_hit_does_not_turn_the_car_and_a_corner_hit_does():
    square = _hit(-math.pi / 2)
    assert max(abs(r) for r in square) < 0.05
    glancing = _hit(-math.pi / 2 + 0.6)                                   # nose pointing right of the normal
    assert max(abs(r) for r in glancing) > 0.5
    old = _hit(-math.pi / 2 + 0.6, contact_at_point=False)
    assert max(abs(r) for r in old) < max(abs(r) for r in glancing)
