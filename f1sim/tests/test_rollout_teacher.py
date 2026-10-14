"""The shadow copy the rollout teacher decides in has to be the real race, not a stale picture of it."""
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


def _env(envs=4, seed=5):
    tr, rl = _track_and_raceline()
    cfg = Config(); cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    ecfg = EnvConfig(race_size=2, opponent="teacher", action_mode="plan", collision_mode="soft",
                     max_steps=4000, hist_len=0)
    env = common.make_env([tr], envs, "cpu", ecfg, cfg=cfg, seed=seed, rls=[rl])
    return env, common.make_teacher([rl], env)


def _raw(env, teacher):
    return teacher.plan_action(env.sim.state, env.sim.P, env.sim.tid, env.ecfg.v_max_policy,
                               env.tracker.spec, plan_speed=env._tracker_plan_speed())


def test_every_refresh_is_the_real_race_not_the_first_one():
    """Refresh, let the real race run on, refresh again: the shadow must follow the real env both
    times. It did not, at first: the refresh cached tensor references, the simulator rebinds some of
    them every step (`self.s = s`), and every decision after the first one was taken in a copy of the
    race as it had been at the first."""
    from f1sim.rollout_teacher import ShadowEnv
    env, teacher = _env()
    env.reset(seed=5)
    for _ in range(20):
        env.step(_raw(env, teacher))
    sh = ShadowEnv(env, 2, lidar_beams=None)
    for rounds in range(2):
        sh.refresh()
        assert torch.equal(sh.S.sim.state.view(2, env.B, -1)[1], env.sim.state)
        assert sh.S.sim.t == env.sim.t, "odometry is stamped against the clock; it has to be copied"
        for _ in range(8):
            env.step(_raw(env, teacher))


def test_the_shadow_drives_like_the_real_race():
    """Same policy, same start: the shadow and the real env stay within a centimetre for 10 steps
    (they draw different sensor noise, so not bit for bit)."""
    from f1sim.rollout_teacher import ShadowEnv
    env, teacher = _env()
    env.reset(seed=5)
    for _ in range(20):
        env.step(_raw(env, teacher))
    sh = ShadowEnv(env, 2, lidar_beams=None)
    tS = common.make_teacher([_track_and_raceline()[1]], sh.S)
    sh.refresh()
    for _ in range(10):
        sh.S.step(_raw(sh.S, tS))
        env.step(_raw(env, teacher))
    err = (sh.S.sim.state[:, :2].view(2, env.B, 2) - env.sim.state[None, :, :2]).norm(dim=2)
    assert float(err.max()) < 0.01, f"shadow drifted {float(err.max()):.4f} m in 10 steps"


def test_the_candidate_set_has_to_contain_the_raceline():
    from f1sim.rollout_teacher import RolloutTeacher
    env, teacher = _env(envs=2)
    with pytest.raises(ValueError):
        RolloutTeacher(teacher, env, offsets=(-0.3, 0.3))
