"""Contact replay: the scenes shortly before the learner's car contacts come back as race starts."""
from __future__ import annotations

import functools

import pytest
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


def _env(p=1.0, envs=4, seed=5):
    tr, rl = _track_and_raceline()
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(race_size=2, opponent="teacher", action_mode="plan", collision_mode="soft", max_steps=4000,
                     hist_len=0, replay_contacts=p)
    return common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl])


def test_off_builds_nothing():
    env = _env(p=0.0)
    assert env._replay is None


def test_a_recorded_scene_is_where_a_full_race_reset_puts_both_cars():
    env = _env()
    env.reset(seed=5)
    rp = env._replay
    tid = float(env.sim.tid[0]); kind = 0.0                    # not slot mode: one kind
    xy, yaw = env.sim.track.pose_at_s(torch.tensor([10.0, 12.5]), env.sim.tid[:2])
    scene = torch.tensor([tid, kind, float(xy[0, 0]), float(xy[0, 1]), float(yaw[0]), 3.0,
                          float(xy[1, 0]), float(xy[1, 1]), float(yaw[1]), 2.5])
    rp.bank[0] = scene; rp.n = 1
    env._reset_envs(torch.arange(env.B))
    for race in range(env.B // 2):
        me, other = 2 * race, 2 * race + 1
        assert torch.allclose(env.sim.state[me, :3], scene[2:5], atol=1e-5)
        assert torch.allclose(env.sim.state[other, :3], scene[6:9], atol=1e-5)
        assert abs(float(env.sim.state[me, 3]) - 3.0) < 1e-5 and abs(float(env.sim.state[other, 3]) - 2.5) < 1e-5
    assert rp.replayed == env.B // 2


def test_a_contact_records_the_scene_from_before_it():
    """Drive the learner into its opponent; the bank gets the two poses `lag` steps before the onset."""
    env = _env()
    env.reset(seed=5)
    rp = env._replay
    B = env.B
    straight = torch.zeros(B, env.act_dim); straight[:, -2:] = 3.0 / env.ecfg.v_max_policy * 2 - 1
    trail = []
    for t in range(rp.lag + 5):                   # long enough that the ring holds a scene `lag` back
        env.step(straight)
        trail.append(env.sim.state[:, :2].clone())
    assert rp.recorded == 0
    # then each opponent 1.0 m ahead of its learner on the line, nearly still, learner at 5 m/s
    s0 = torch.tensor([20.0, 21.0] * (B // 2))
    xy, yaw = env.sim.track.pose_at_s(s0, env.sim.tid)
    env.sim.reset(torch.arange(B), torch.cat([xy, yaw[:, None]], 1), torch.tensor([5.0, 0.3] * (B // 2)))
    fast = straight.clone(); fast[:, -2:] = 5.0 / env.ecfg.v_max_policy * 2 - 1
    for t in range(40):
        env.step(fast)
        trail.append(env.sim.state[:, :2].clone())
        if rp.recorded:
            break
    assert rp.recorded > 0, "no contact was recorded"
    rec = rp.bank[0]
    onset = len(trail) - 1
    want = trail[onset - rp.lag]
    me = int(((env.sim.tid.float() == rec[0]) & env.learner).nonzero()[0])
    assert torch.allclose(rec[2:4], want[me], atol=1e-4), (rec[2:4], want[me])
