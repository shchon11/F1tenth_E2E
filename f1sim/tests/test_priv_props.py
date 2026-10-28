"""`EnvConfig.priv_props` / `ppo --priv-props` (rl-audit 2026-10-28): the props ahead in the v2 critic's privileged vector,
and widening an --init critic that does not read them so that it starts exactly as it was."""
from __future__ import annotations

import functools

import torch

from f1sim import Config, maps
from f1sim.gym_env import PRIV_OPP_DIST_SCALE, PRIV_V2_DIM, PRIV_V2_PROP_DIM, EnvConfig
from f1sim.learn import common

TRACK = "gen:competition:0"


@functools.lru_cache(maxsize=1)
def _track():
    from f1sim.raceline import Raceline
    tr = maps.load(TRACK)
    return tr, Raceline.build_cached(tr)


def _env(envs=4, seed=3, **kw):
    tr, rl = _track()
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 36
    e = dict(action_mode="plan", race_size=2, opponent="teacher", opp_speed_range=(0.7, 0.9), collision_mode="soft",
             compile_tracker=False, max_steps=40, speed_cap=9.0, obs_version=2, speed_mode="wp4", scan_stack=1,
             hist_len=20, hist_stride=2, procedural_obstacles=1.0, procedural_density=1.5, procedural_max_props=10,
             spawn_runway=3.0)
    e.update(kw)
    env = common.make_env([tr], envs, "cpu", EnvConfig(**e), cfg=cfg, seed=seed, rls=[rl])
    env.reset(seed=seed)
    return env


def test_the_prop_block_is_appended_only_when_asked():
    assert _env().privileged_v2().shape[1] == PRIV_V2_DIM
    env = _env(priv_props=True)
    pv = env.privileged_v2()
    assert pv.shape == (env.B, PRIV_V2_DIM + PRIV_V2_PROP_DIM)
    assert torch.equal(pv[:, :PRIV_V2_DIM], _env().privileged_v2())     # the old columns are untouched


def test_planted_props_nearest_first_and_none_behind_or_far():
    env = _env(priv_props=True)
    po, st = env.procedural, env.sim.state
    po.p_sid.fill_(-1)                                                   # every slot dead ...
    x, y, yaw = (float(v) for v in st[0, :3])
    c, s = torch.cos(torch.tensor(yaw)).item(), torch.sin(torch.tensor(yaw)).item()

    def plant(slot, fwd, left):                                          # ... then props at body-frame spots
        po.p_poses[0, slot, 0] = x + fwd * c - left * s
        po.p_poses[0, slot, 1] = y + fwd * s + left * c
        po.p_sid[0, slot] = 0

    plant(0, 6.0, -0.4)
    plant(1, 3.0, 0.5)
    plant(2, -3.0, 0.0)                                                  # behind: not counted
    plant(3, 20.0, 0.0)                                                  # beyond the range: not counted
    blk = env.privileged_v2()[0, PRIV_V2_DIM:].reshape(3, 4)
    S = PRIV_OPP_DIST_SCALE
    want0 = torch.tensor([1.0, 3.0 / S, 0.5 / S, (3.0 ** 2 + 0.5 ** 2) ** 0.5 / S])
    want1 = torch.tensor([1.0, 6.0 / S, -0.4 / S, (6.0 ** 2 + 0.4 ** 2) ** 0.5 / S])
    assert torch.allclose(blk[0], want0, atol=1e-4) and torch.allclose(blk[1], want1, atol=1e-4)
    assert torch.equal(blk[2], torch.zeros(4))
    assert torch.equal(env.privileged_v2()[1:, PRIV_V2_DIM:], torch.zeros(env.B - 1, PRIV_V2_PROP_DIM))


def test_widening_an_init_critic_changes_none_of_its_outputs():
    from f1sim.learn.model_v2 import ActorCriticV2
    from f1sim.learn.ppo_v2 import widen_critic_priv
    torch.manual_seed(0)
    m = ActorCriticV2(n_beams=36, n_frames=1, act_dim=8, speed_mode="wp4", priv_dim=PRIV_V2_DIM)
    B = 5
    scan = torch.rand(B, 1, 36); ego = torch.randn(B, m.critic.ego_dim); cc = torch.rand(B, 1)
    priv = torch.randn(B, PRIV_V2_DIM)
    v0, k0 = m.critic(scan, ego, cc, priv)
    assert widen_critic_priv(m, PRIV_V2_DIM + PRIV_V2_PROP_DIM) == (PRIV_V2_DIM, PRIV_V2_DIM + PRIV_V2_PROP_DIM)
    assert m.meta["priv_dim"] == PRIV_V2_DIM + PRIV_V2_PROP_DIM
    v1, k1 = m.critic(scan, ego, cc, torch.cat([priv, torch.randn(B, PRIV_V2_PROP_DIM)], 1))
    assert torch.allclose(v0, v1) and torch.allclose(k0, k1)
    assert widen_critic_priv(m, PRIV_V2_DIM + PRIV_V2_PROP_DIM) is None   # already wide: nothing to do
    m2 = ActorCriticV2(**m.meta)                                         # a saved checkpoint rebuilds the wide critic
    assert m2.critic.priv_dim == PRIV_V2_DIM + PRIV_V2_PROP_DIM


def test_the_race_time_still_to_run_is_the_last_column():
    from f1sim.gym_env import PRIV_V2_TIME_DIM
    env = _env(priv_time=True, max_steps=40)
    pv = env.privileged_v2()
    assert pv.shape[1] == PRIV_V2_DIM + PRIV_V2_TIME_DIM
    assert torch.allclose(pv[:, -1], torch.ones(env.B))                 # a fresh race: all of it still to run
    for _ in range(3):
        env.step(torch.zeros(env.B, env.act_dim))
    assert torch.allclose(env.privileged_v2()[:, -1], torch.full((env.B,), 37 / 40))
    both = _env(priv_props=True, priv_time=True, max_steps=40).privileged_v2()
    assert both.shape[1] == PRIV_V2_DIM + PRIV_V2_PROP_DIM + PRIV_V2_TIME_DIM


def test_the_end_of_a_race_is_terminal_not_bootstrapped():
    from f1sim.learn.ppo_v2 import race_end_flags
    from f1sim.learn.returns import compute_gae
    term = torch.tensor([[0.0], [0.0]]); trunc = torch.tensor([[0.0], [1.0]])
    d, tr = race_end_flags(term.bool(), trunc.bool(), False)
    assert torch.equal(d, term.bool()) and torch.equal(tr, trunc.bool())  # off: as before
    d, tr = race_end_flags(term.bool(), trunc.bool(), True)
    assert d.tolist() == [[False], [True]] and not tr.any()
    rew = torch.tensor([[1.0], [1.0]]); val = torch.tensor([[5.0], [5.0], [5.0]]); final = torch.tensor([[0.0], [100.0]])
    boot = compute_gae(rew, val, term, trunc, final, 1.0, 1.0)            # bootstrapped: the last step sees 100
    race = compute_gae(rew, val, d.float(), tr.float(), final, 1.0, 1.0)  # the race is over: nothing after it
    assert float(boot[1, 0]) == 1.0 + 100.0 - 5.0 and float(race[1, 0]) == 1.0 - 5.0
