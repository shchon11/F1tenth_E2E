"""The profile executor: a plan's own speed profile, sent to the VESC directly, and the contract that says so."""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from f1sim import mpc
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.params import Config
from f1sim.track import Track


def _open_floor():
    res, n = 0.1, 400
    occ = np.zeros((n, n), bool); occ[:2] = occ[-2:] = True; occ[:, :2] = occ[:, -2:] = True
    edt = ndimage.distance_transform_edt(~occ).astype(np.float32) * res
    th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    return Track(occupancy=occ, resolution=res, origin=(0.0, 0.0), edt=edt,
                 centerline=np.stack([20 + 15 * np.cos(th), 20 + 15 * np.sin(th)], 1), name="open",
                 duct=occ.copy(), tall=np.zeros_like(occ))


def _launch(speed_mode, speed_command, steps=20):
    """A car at 2 m/s asked for everything on a straight: mean acceleration over `steps` control steps."""
    cfg = Config(); cfg.sim.compile = False; cfg.rand.enabled = False
    env = common.make_env([_open_floor()], 2, "cpu", EnvConfig(action_mode="plan", race_size=1, collision_mode="soft",
                                                               speed_mode=speed_mode, speed_command=speed_command,
                                                               compile_tracker=False, speed_cap=9.0),
                          cfg=cfg, seed=1)
    env.reset(seed=1)
    env.sim.reset(torch.arange(2), torch.tensor([[5.0, 20.0, 0.0]] * 2), torch.full((2,), 2.0))
    a = torch.zeros(2, env.act_dim)
    a[:, -2:] = 1.0                                          # linear: v0 = v1 = max; envelope: a_hat max, v_end max
    v0 = float(env.sim.state[0, 3])
    for _ in range(steps):
        env.step(a)
    return (float(env.sim.state[0, 3]) - v0) / (steps * env.sim.control_dt)


def test_the_profile_command_drives_the_car_near_its_limit_where_the_tracker_does_not():
    through_tracker = _launch("envelope", "tracker")
    direct = _launch("envelope", "profile")
    assert direct > through_tracker + 0.5, (direct, through_tracker)
    assert direct > 3.5, direct                                   # a_max 5.6 less the power limit above 6 m/s


def test_the_contract_defaults_to_what_every_old_checkpoint_was_and_reads_either_place():
    assert common.plan_output_of({}) == {"speed_mode": "linear", "speed_command": "tracker"}
    assert common.plan_output_of({"speed_mode": "envelope", "speed_command": "profile"}) == \
        {"speed_mode": "envelope", "speed_command": "profile"}
    assert common.plan_output_of({"experiment": {"speed_mode": "knots", "speed_command": "profile"}}) == \
        {"speed_mode": "knots", "speed_command": "profile"}


def test_the_forward_pass_respects_the_cars_power_limit():
    spec = mpc.PlanSpec(speed_mode="envelope")
    a = torch.zeros(1, mpc.act_dim("envelope")); a[:, -2:] = 1.0      # straight, all the grip, top end speed
    _, Lp, prof = mpc.decode_profile(a, torch.tensor([6.0]), 10.0, torch.tensor([10.0]), spec)
    ds = float(Lp[0]) / (prof.shape[1] - 1)
    v = prof[0].numpy()
    gain = (v[1:] ** 2 - v[:-1] ** 2) / (2 * ds)                      # the acceleration each step implies
    assert gain.max() <= spec.a_drive_profile * spec.v_switch_profile / 6.0 + 1e-3
