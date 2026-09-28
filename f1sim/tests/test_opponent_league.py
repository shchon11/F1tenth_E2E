"""A league slot: its car is driven by a checkpoint in a share of races and by its teacher kind in the rest."""
from __future__ import annotations

import functools
import os

import pytest
import torch

from f1sim import Config, maps
from f1sim.gym_env import OPP_DRIVER_POOL, OPP_DRIVER_TEACHER, EnvConfig
from f1sim.learn import common
from f1sim.opponent_slots import OpponentSlot, parse_slots

TRACK = "gen:competition:0"
CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "iccas_specialist_s915_u768.pt")
needs_ckpt = pytest.mark.skipif(not os.path.exists(CKPT), reason="archived checkpoint not present")


@functools.lru_cache(maxsize=1)
def _track_and_raceline():
    from f1sim.raceline import Raceline
    tr = maps.load(TRACK)
    return tr, Raceline.build_cached(tr)


def _env(slot: dict, envs=8, seed=3):
    tr, rl = _track_and_raceline()
    cfg = Config(); cfg.sim.compile = False; cfg.lidar.n_beams = 1081
    ecfg = EnvConfig(race_size=2, opponent="slots", opponent_slots=[slot], action_mode="plan",
                     collision_mode="soft", max_steps=400, hist_len=20, scan_stack=6)
    env = common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl])
    env.reset(seed=seed)
    return env


def test_a_league_is_only_for_a_teacher_slot_and_needs_its_checkpoints():
    with pytest.raises(ValueError, match="teacher-driven"):
        OpponentSlot.from_dict({"kind": "self", "ckpt_mix": ["a.pt"]})
    with pytest.raises(ValueError, match="probability"):
        OpponentSlot.from_dict({"kind": "raceline", "ckpt_mix": ["a.pt"], "ckpt_p": 1.5})
    with pytest.raises(ValueError, match="no checkpoints"):
        OpponentSlot.from_dict({"kind": "raceline", "ckpt_p": 0.3})
    sl = OpponentSlot.from_dict({"kind_mix": ["forzaeth", "lane_switch"], "ckpt_mix": ["a.pt", "b.pt"], "ckpt_p": 0.4})
    assert sl.ckpt_mix == ("a.pt", "b.pt") and sl.ckpt_p == 0.4
    assert parse_slots([sl.to_dict()])[0] == sl                                   # round trip


@needs_ckpt
def test_p_one_hands_every_opponent_to_the_checkpoint_and_it_drives():
    env = _env({"kind_mix": ["forzaeth", "lane_switch"], "ckpt_mix": [CKPT], "ckpt_p": 1.0, "speed_scale": [0.8, 0.8]})
    opp = env.slot > 0
    assert bool((env.opp_driver[opp] == OPP_DRIVER_POOL).all())
    assert not bool(env.teacher_driven.any()) and bool(env.pool_driven[opp].all())
    assert env.pool is not None
    x0 = env.sim.state[opp, :2].clone()
    for _ in range(60):
        env.step(torch.zeros(env.B, env.act_dim))
    assert float((env.sim.state[opp, :2] - x0).norm(dim=1).min()) > 1.0          # the checkpoint moved its car


@needs_ckpt
def test_a_half_league_mixes_drivers_across_races_and_learners_never_move():
    env = _env({"kind_mix": ["forzaeth", "lane_switch"], "ckpt_mix": [CKPT], "ckpt_p": 0.5,
                "events": ["brake"], "event_rate": 2.0}, envs=32)
    learners = env.learner.clone()
    seen_pool = seen_teacher = 0
    for _ in range(4):
        env._reset_envs(torch.arange(env.B))
        opp = env.slot > 0
        seen_pool += int(env.pool_driven[opp].sum()); seen_teacher += int(env.teacher_driven[opp].sum())
        assert bool(((env.opp_driver[opp] == OPP_DRIVER_POOL) | (env.opp_driver[opp] == OPP_DRIVER_TEACHER)).all())
        assert torch.equal(env.learner, learners)
        # the event gate is the teacher rows: a checkpoint car is never scripted
        env.step(torch.zeros(env.B, env.act_dim))
        assert torch.equal(env.events.gate, env.teacher_driven)
        assert not bool((env.events.gate & env.pool_driven).any())
    assert seen_pool > 0 and seen_teacher > 0
