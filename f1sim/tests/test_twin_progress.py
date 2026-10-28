"""Twin worlds (`common.make_twin`, `learn.twin_sync`, `learn.twin_progress`; rl-audit 2026-10-28): a twin built by the
same call and synced from its master steps exactly like it, a branch with no perturbation measures exactly nothing, and
the loss pushes the mean action along the measured gradient."""
from __future__ import annotations

import functools

import torch

from f1sim import Config, maps
from f1sim.gym_env import EnvConfig
from f1sim.learn import common
from f1sim.learn.twin_progress import TwinProgress, _rng_state, _set_rng_state
from f1sim.learn.twin_sync import sync

TRACK = "gen:competition:0"


@functools.lru_cache(maxsize=1)
def _track():
    from f1sim.raceline import Raceline
    tr = maps.load(TRACK)
    return tr, Raceline.build_cached(tr)


def _env(envs=4, seed=3):
    tr, rl = _track()
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    e = dict(action_mode="plan", race_size=2, opponent="teacher", opp_speed_range=(0.7, 0.9), collision_mode="soft",
             compile_tracker=False, max_steps=400, speed_cap=9.0, obs_version=2, speed_mode="wp4", scan_stack=1,
             hist_len=20, hist_stride=2, procedural_obstacles=1.0, procedural_density=1.5, procedural_max_props=10,
             spawn_runway=3.0)
    env = common.make_env([tr], envs, "cpu", EnvConfig(**e), cfg=cfg, seed=seed, rls=[rl])
    env.reset(seed=seed)
    return env


def _actor(env):
    from f1sim.learn.model_v2 import ActorCriticV2
    spec = common.obs_spec(env)
    torch.manual_seed(0)
    return ActorCriticV2(n_beams=spec.n_beams, n_frames=spec.n_frames, ego_dim=spec.ego_dim, act_dim=spec.act_dim,
                         speed_mode=spec.speed_mode).actor


def _step_both(a, b, act_a, act_b):
    s = _rng_state()
    out_a = a.step(act_a)
    _set_rng_state(s)
    out_b = b.step(act_b)
    return out_a, out_b


def test_make_twin_is_the_same_world():
    env = _env()
    tw = common.make_twin(env, 6)
    assert tw.B == 6 and tw.ecfg is env.ecfg and tw.act_dim == env.act_dim
    assert torch.equal(common.make_twin(env, env.B).learner_ids, env.learner_ids)


def test_a_synced_twin_steps_exactly_like_its_master():
    env = _env()
    tw = common.make_twin(env, env.B)
    s = _rng_state(); tw.reset(seed=3); _set_rng_state(s)
    gen = torch.Generator().manual_seed(1)
    for _ in range(5):                                       # lockstep: lazily created state exists in both
        act = torch.rand(env.B, env.act_dim, generator=gen) * 2 - 1
        _step_both(env, tw, act, act)
    for _ in range(3):                                       # the worlds part ways ...
        _step_both(env, tw, torch.rand(env.B, env.act_dim, generator=gen) * 2 - 1,
                   torch.rand(env.B, env.act_dim, generator=gen) * 2 - 1)
    assert not torch.equal(env.sim.state, tw.sim.state)
    sync(env, tw)                                            # ... and the twin becomes the master again
    assert torch.equal(env.sim.state, tw.sim.state)
    for _ in range(6):
        act = torch.rand(env.B, env.act_dim, generator=gen) * 2 - 1
        (oa, ra, *_), (ob, rb, *_) = _step_both(env, tw, act, act)
        assert torch.equal(env.sim.state, tw.sim.state)
        assert torch.equal(ra, rb) and torch.equal(oa["scan"], ob["scan"])


def test_a_branch_without_a_perturbation_measures_exactly_nothing():
    env = _env()
    m, t = common.make_twin(env, 4), common.make_twin(env, 4)
    actor = _actor(env)
    tp = TwinProgress(m, t, actor, cond=lambda: torch.zeros(m.B, 1), live=[0, 1, 2, 4], deltas=[0, 0, 0, 0],
                      h=12, kp=4, warmup=6, seed=5)
    st = tp.branch()
    assert st["n"] >= 1 and st["dP_mean"] == 0.0 and torch.equal(tp.batch["g"], torch.zeros_like(tp.batch["g"]))
    tp2 = TwinProgress(m, t, actor, cond=lambda: torch.zeros(m.B, 1), live=[0, 1, 2, 4], deltas=[0.1, 0.1, 0.1, 0.1],
                       h=12, kp=4, warmup=2, seed=5)
    tp2.branch()
    assert torch.isfinite(tp2.batch["g"]).all() and tp2.batch["g"].abs().sum() > 0   # a perturbation shows


def test_the_loss_pushes_the_mean_along_the_measured_gradient():
    env = _env()
    m, t = common.make_twin(env, 4), common.make_twin(env, 4)
    actor = _actor(env)
    tp = TwinProgress(m, t, actor, cond=lambda: torch.zeros(m.B, 1), live=[0, 1, 2, 4], deltas=[0.1] * 4,
                      h=6, kp=2, warmup=2, seed=5)
    tp.branch()
    n = tp.batch["g"].shape[0]
    g = torch.zeros(n, 4); g[:, 3] = 1.0                     # "more v1 is better" everywhere
    tp.batch["g"] = g
    actor.zero_grad()
    tp.loss().backward()
    grad_b = actor.mu.bias.grad
    assert grad_b[4] < 0 and torch.allclose(grad_b[[0, 1, 2, 3, 5, 6, 7]], torch.zeros(7))   # descent raises v1 only
