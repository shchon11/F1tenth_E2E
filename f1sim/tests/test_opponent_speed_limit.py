"""The speed limit for a car ahead (`f1sim/opponent_speed_limit.py`, 2026-10-25): the rule on the policy's own
nearest-car reading, its feed from the policy runner, and the env holding the learner's speed command to it."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.opponent_speed_limit import DEFAULT, no_car, speed_limit


def reading(dx_lidar, dy, closing, v_lat, seen=True):
    """one nearest-car output row: [exists logit, dx / 5, dy / 2, closing / 5, v_lat / 2] (LiDAR frame)"""
    return torch.tensor([[4.0 if seen else -4.0, dx_lidar / 5.0, dy / 2.0, closing / 5.0, v_lat / 2.0]])


def test_a_car_in_our_lane_ahead_sets_the_speed_we_can_still_slow_to_its_speed_from():
    # LiDAR 0.297 m ahead of the reference point: dx = 2.2 + 0.297, body gap = dx - 0.5
    lim = speed_limit(reading(2.2, 0.1, 2.0, 0.0), torch.tensor([5.0]))
    gap = 2.2 + 0.297 - 0.5
    room = gap - 0.2 - 2.0 * 0.2
    assert float(lim) == pytest.approx(3.0 + math.sqrt(2 * 3.0 * room), rel=1e-5)


@pytest.mark.parametrize("why, r", [
    ("no car seen", reading(2.2, 0.1, 2.0, 0.0, seen=False)),
    ("beside us, not wholly ahead", reading(0.0, 0.5, 2.0, 0.0)),
    ("beyond reach", reading(7.0, 0.0, 2.0, 0.0)),
    ("beside our lane, drifting away", reading(2.2, 0.6, 2.0, 0.5)),
    ("beside our lane, too slow to arrive within the horizon", reading(2.2, 0.8, 2.0, -0.3)),
])
def test_no_limit(why, r):
    assert torch.isinf(speed_limit(r, torch.tensor([5.0]))).all(), why


def test_a_car_drifting_into_our_lane_within_the_horizon_counts_unless_we_are_past_it_by_then():
    # 0.6 m to the left, coming at 0.5 m/s: under 0.35 m in 0.5 s, still ahead then -> limited
    assert torch.isfinite(speed_limit(reading(2.2, 0.6, 2.0, -0.5), torch.tensor([5.0]))).all()
    # the same drift, but the car is only 0.4 m ahead of the LiDAR and we close at 3 m/s: a car length past it by then
    assert torch.isinf(speed_limit(reading(0.4, 0.6, 3.0, -0.25), torch.tensor([5.0]))).all()
    assert DEFAULT.horizon == 1.0


def test_the_feed_copies_the_policys_reading_and_refuses_a_policy_without_one():
    env = SimpleNamespace(opponent_reading=no_car(3), learner=torch.tensor([True, False, True]))
    actor = SimpleNamespace(near_car=object(), last_reading=None)
    feed = common.opponent_reading_feeder(SimpleNamespace(actor=actor), env)
    assert actor.keep_reading is True
    feed()                                                     # nothing read yet: unchanged
    assert torch.equal(env.opponent_reading, no_car(3))
    actor.last_reading = torch.arange(15, dtype=torch.float32).view(3, 5)
    feed()
    assert torch.equal(env.opponent_reading, actor.last_reading)
    actor.last_reading = torch.ones(2, 5)                      # an actor run on the learner rows only
    feed()
    assert torch.equal(env.opponent_reading[[0, 2]], torch.ones(2, 5))
    assert common.opponent_reading_feeder(SimpleNamespace(actor=actor), SimpleNamespace(opponent_reading=None)) is None
    with pytest.raises(ValueError, match="nearest-car output"):
        common.opponent_reading_feeder(SimpleNamespace(actor=SimpleNamespace(near_car=None)), env)


def test_the_env_holds_the_learners_speed_command_to_the_limit_and_leaves_other_rows_alone():
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"
    env = common.make_env([maps.load("real:korea_2025_iccas")], 2, "cpu",
                          EnvConfig(max_steps=100, hist_len=0, action_mode="plan", speed_mode="wp4",
                                    compile_tracker=False, opponent_speed_limit=True), cfg=cfg, seed=1)
    env.reset(seed=1)
    env.learner[1] = False
    assert env.opponent_reading is not None and env.opponent_reading.shape == (2, 5)
    a = torch.zeros(2, env.act_dim)
    a[:, 4:] = 0.9                                             # every waypoint speed near the top
    env.step(a)
    free = env.last_cmd_raw[:, 1].clone()                      # no car seen yet: nothing held
    assert not bool(env.opponent_limited.any())
    # a car 0.6 m ahead of the LiDAR in our lane, closing at 3 m/s: the limit is about its speed
    env.opponent_reading.copy_(torch.cat([reading(0.6, 0.0, 3.0, 0.0)] * 2))
    env.step(a)
    v_me = env.sim.state[:, 3]
    assert bool(env.opponent_limited[0]) and not bool(env.opponent_limited[1])
    assert float(env.last_cmd_raw[0, 1]) < float(free[0])
    lim = speed_limit(env.opponent_reading, v_me, float(cfg.lidar.mount_x), float(cfg.lidar.mount_y))
    assert float(env.last_cmd_raw[0, 1]) <= float(lim[0]) + 1e-5
