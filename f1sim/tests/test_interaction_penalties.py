"""The rear-end and merge penalties charge what they say and nothing else."""
from __future__ import annotations

import functools

import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common

TRACK = "gen:competition:0"


@functools.lru_cache(maxsize=1)
def _track_and_raceline():
    from f1sim.raceline import Raceline
    tr = maps.load(TRACK)
    return tr, Raceline.build_cached(tr)


def _env(**kw):
    tr, rl = _track_and_raceline()
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(race_size=2, opponent="teacher", action_mode="plan", collision_mode="soft", max_steps=4000,
                     hist_len=0, **kw)
    env = common.make_env([tr], 2, "cpu", ecfg, cfg=cfg, seed=5, rls=[rl])
    env.reset(seed=5)
    return env


def _place(env, s_me, s_op, lat_me=0.0, lat_op=0.0, v_me=0.0, v_op=0.0, vy_me=0.0):
    s = torch.tensor([s_me, s_op])
    xy, yaw = env.sim.track.pose_at_s(s, env.sim.tid)
    nrm = torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1)
    xy = xy + nrm * torch.tensor([lat_me, lat_op])[:, None]
    env.sim.reset(torch.arange(2), torch.cat([xy, yaw[:, None]], 1), torch.tensor([v_me, v_op]))
    env.sim.state[0, 4] = vy_me


def test_rear_end_is_charged_once_at_onset_by_squared_relative_speed():
    env = _env(reward_rear_end=2.0)
    _place(env, 20.0, 20.5, v_me=4.0, v_op=1.0)            # a car 0.5 m ahead, bodies already touching
    hit = torch.tensor([True, True])
    pen_r, pen_m = env._interaction_penalties(env.sim.state, hit)
    dv2 = float((env.sim.state[0, 3] - env.sim.state[1, 3]) ** 2)
    assert abs(float(pen_r[0]) + 2.0 * dv2) < 1e-4 and float(pen_m.abs().sum()) == 0.0
    pen_r2, _ = env._interaction_penalties(env.sim.state, hit)            # still touching: not charged again
    assert float(pen_r2[0]) == 0.0


def test_rear_end_does_not_charge_the_car_that_is_ahead():
    env = _env(reward_rear_end=2.0)
    _place(env, 20.5, 20.0, v_me=1.0, v_op=4.0)            # I am the one in front
    pen_r, _ = env._interaction_penalties(env.sim.state, torch.tensor([True, True]))
    assert float(pen_r[0]) == 0.0


def test_merge_charges_only_my_own_motion_toward_a_car_level_or_just_behind():
    env = _env(reward_merge=10.0)
    # I am 0.5 m ahead and 0.6 m to its left, sliding right (toward it) at 0.8 m/s
    _place(env, 20.5, 20.0, lat_me=0.6, lat_op=0.0, v_me=3.0, v_op=3.0, vy_me=-0.8)
    _, pen_m = env._interaction_penalties(env.sim.state, torch.tensor([False, False]))
    assert float(pen_m[0]) < 0.0 and float(pen_m[1]) == 0.0              # the opponent row is never charged
    # sliding away from it: nothing
    _place(env, 20.5, 20.0, lat_me=0.6, lat_op=0.0, v_me=3.0, v_op=3.0, vy_me=+0.8)
    _, pen_m = env._interaction_penalties(env.sim.state, torch.tensor([False, False]))
    assert float(pen_m[0]) == 0.0
    # the same slide with the other car 3 m behind: nothing
    _place(env, 23.0, 20.0, lat_me=0.6, lat_op=0.0, v_me=3.0, v_op=3.0, vy_me=-0.8)
    _, pen_m = env._interaction_penalties(env.sim.state, torch.tensor([False, False]))
    assert float(pen_m[0]) == 0.0


def test_off_leaves_the_components_at_zero():
    env = _env()
    a = torch.zeros(env.B, env.act_dim)
    _, _, _, _, info = env.step(a)
    rc = info["reward_components"]
    assert float(rc["rear_end"].abs().sum()) == 0.0 and float(rc["merge"].abs().sum()) == 0.0
