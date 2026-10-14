"""The league in a v2 learner's race: every seat drives in its own observation contract and plan format.

The boot camp's league seats are v1 checkpoints (six frames, the 366-wide proprio vector, linear
plans); a v2 learner's env emits contract v2 and plans WP4; the learner's own v2 snapshots join the
league as it grows. `learn.seat_obs` keeps each contract's histories for every row, fed by the env
where it updates its own, and the env reads each league entry's rows in the entry's own format
(`F1VecEnv._set_entry_formats`). What has to hold, and is checked here:

* a v1 league seat in a v2 env commands, bit for bit, what the same checkpoint commands in a v1 env
  from the same state -- its observation, its plan, its command and the whole simulator state, step
  for step through full race resets and a single car's respawn, with the v2 env's learner cap on;
* the reverse: a v2 entry in a v1 env (its own v2 builder) commands what it commands in a v2 env (the
  env's own observation);
* a mixed race -- a v2 learner, a v1 league seat, a scripted seat, a v2 league entry -- runs, each car
  planned in its own format, each one moving;
* the learner's snapshot takes a v1 seat mid-run: that seat's rows switch to wp4 and the env's own
  observation, the v1 builder goes when nothing reads it;
* what a seat cannot be shown honestly is refused with the reason.

Each env runs alone from its seeded reset: the simulator draws part of its sensor noise from the
global generator, so two envs stepped in turn would draw each other's noise.
"""
from __future__ import annotations

import functools
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1sim import Config, maps  # noqa: E402
from f1sim.gym_env import EnvConfig  # noqa: E402
from f1sim.learn import common  # noqa: E402
from f1sim.learn.obs import flatten_obs  # noqa: E402

TRACK = "gen:competition:0"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.join(ROOT, "checkpoints", "iccas_bootcamp_safest.pt")
needs_v1 = pytest.mark.skipif(not os.path.exists(V1), reason="archived v1 checkpoint not present")
#: steps per identity run: two full race resets (max_steps 25) and a respawn at step 11
STEPS = 60


@functools.lru_cache(maxsize=1)
def _track():
    from f1sim.raceline import Raceline
    tr = maps.load(TRACK)
    return tr, Raceline.build_cached(tr)


@pytest.fixture(scope="module")
def v2_ckpt(tmp_path_factory):
    from v2_fixtures import generated_v2_checkpoint
    return generated_v2_checkpoint(str(tmp_path_factory.mktemp("v2") / "gen_v2.pt"))


def _env(slots, *, v2: bool, envs=4, seed=7, cap=None, **kw):
    tr, rl = _track()
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 1081
    if v2:
        obs = dict(obs_version=2, speed_mode="wp4", scan_stack=1, hist_len=20, hist_stride=2)
    else:
        obs = dict(scan_stack=6, hist_len=20, hist_stride=2)
    obs.update(kw)
    e = EnvConfig(action_mode="plan", race_size=len(slots) + 1, opponent="slots", opponent_slots=slots,
                  collision_mode="soft", compile_tracker=False, max_steps=25, speed_cap=9.0,
                  **({"learner_cap_range": cap} if cap else {}), **obs)
    env = common.make_env([tr], envs, "cpu", e, cfg=cfg, seed=seed, rls=[rl])
    env.reset(seed=seed)
    return env


def _drive(env, entry_obs, steps=STEPS, respawn_at=11):
    """Drive the learner cars by external command (the same in every env, whatever its format), let
    the league drive its own; record, per step, the league rows' observation as the entry saw it,
    the entry's plan, the command its car got and the whole simulator state."""
    opp = ~env.learner
    rec = []
    for k in range(steps):
        for i in torch.nonzero(env.learner).flatten().tolist():
            env.set_external_command(i, 0.05 * ((k % 7) - 3), 2.0 + 0.05 * k)
        scan, pro = flatten_obs(entry_obs(env))
        if k == respawn_at:
            env._reset_envs(torch.nonzero(opp).flatten()[:1])        # one league car respawns alone
        env.step(torch.zeros(env.B, env.act_dim))
        rec.append({"scan": scan[opp].clone(), "obs": pro[opp].clone(), "plan": env.prev_action[opp].clone(),
                    "cmd": env.last_cmd[opp].clone(), "state": env.sim.state.clone()})
    return rec


def _identical(a, b):
    bad = [(k, x) for k in range(len(a)) for x in a[k] if not torch.equal(a[k][x], b[k][x])]
    return bad


@needs_v1
def test_a_v1_league_seat_in_a_v2_env_commands_what_it_commands_in_a_v1_env():
    slot = [{"kind_mix": ["forzaeth", "lane_switch"], "ckpt_mix": [V1], "ckpt_p": 1.0, "speed_scale": [0.8, 0.9]}]
    v1 = _env(slot, v2=False)
    assert v1.pool.entries[0].seat_key is None                    # a v1 env: the env's own observation
    a = _drive(v1, lambda e: e._last_obs)
    v2 = _env(slot, v2=True, cap=(3.0, 9.0), opponent_speed_mode="linear")
    ent = v2.pool.entries[0]
    assert ent.seat_key is not None and ent.speed_mode == "linear" and len(v2.pool.seat_builders) == 1
    b = _drive(v2, lambda e: e.pool.seats[ent.seat_key].obs)
    assert v2.tracker.row_format.tolist() == [1, 0, 1, 0]          # learner wp4, league seat linear
    assert _identical(a, b) == [], _identical(a, b)[:4]
    # the race really was raced: the league cars moved and the resets happened
    assert float((a[-1]["state"][~v1.learner, :2] - a[0]["state"][~v1.learner, :2]).norm(dim=1).min()) > 0.5
    assert int(v1.ep_step.max()) < STEPS
    # the v2 env's learner caps were drawn and never reached the league seat's observation
    assert bool(((v2.speed_cap[v2.learner] >= 3.0) & (v2.speed_cap[v2.learner] <= 9.0)).all())


def test_a_v2_entry_in_a_v1_env_commands_what_it_commands_in_a_v2_env(v2_ckpt):
    slot = [{"kind_mix": ["forzaeth", "lane_switch"], "ckpt_mix": [v2_ckpt], "ckpt_p": 1.0, "speed_scale": [0.8, 0.9]}]
    v2 = _env(slot, v2=True)
    assert v2.pool.entries[0].seat_key is None and v2.tracker.row_format is None   # its own contract and format
    a = _drive(v2, lambda e: e._last_obs)
    v1 = _env(slot, v2=False)
    ent = v1.pool.entries[0]
    assert ent.seat_key is not None and ent.speed_mode == "wp4"
    b = _drive(v1, lambda e: e.pool.seats[ent.seat_key].obs)
    assert v1.tracker.row_format.tolist() == [0, 1, 0, 1]          # a linear learner, the v2 entry in wp4
    assert _identical(a, b) == [], _identical(a, b)[:4]


@needs_v1
def test_a_mixed_race_runs_every_seat_in_its_own_contract_and_format(v2_ckpt):
    from f1sim.learn.memory import Hidden
    from f1sim.learn.model import load_checkpoint
    from f1sim.learn.watch import dial_for
    slots = [{"kind_mix": ["forzaeth", "lane_switch"], "ckpt_mix": [V1], "ckpt_p": 1.0, "speed_scale": [0.8, 0.9]},
             {"kind": "lane_switch", "speed_scale": [0.7, 0.8]},
             {"kind_mix": ["forzaeth"], "ckpt_mix": [v2_ckpt], "ckpt_p": 1.0, "speed_scale": [0.8, 0.9]}]
    env = _env(slots, v2=True, envs=8, seed=5, cap=(3.0, 9.0), opponent_speed_mode="linear")
    model, _ = load_checkpoint(v2_ckpt)
    dial = dial_for(model, "cpu")
    obs, h = env._last_obs, Hidden()
    driven = torch.zeros(env.B)
    formats = []
    for _ in range(40):
        act, _lp, h = model.act(obs["scan"], obs["ego"], deterministic=True, c=dial.c(env.B), h=h)
        obs, _r, term, trunc, info = env.step(act)
        h = h.reset(term | trunc)
        driven += info["progress"]                                 # metres along the lane, through the reset
        formats.append(env.tracker.row_format.view(-1, 4).clone())
    # learner wp4, v1 league seat linear, scripted seat linear, v2 league entry wp4 -- across a race reset
    want = torch.tensor([[1, 0, 0, 1]] * 2, dtype=torch.uint8)
    assert all(torch.equal(f, want) for f in formats)
    assert [e.speed_mode for e in env.pool.entries] == ["linear", "wp4"]
    assert env.pool.entries[0].seat_key is not None and env.pool.entries[1].seat_key is None
    assert bool(torch.isfinite(env.last_cmd).all())
    assert float(driven.min()) > 1.0, driven.view(-1, 4)            # every car, every seat kind, drove


@needs_v1
def test_the_learners_snapshot_takes_a_v1_seat_in_its_own_format_and_contract(v2_ckpt):
    slots = [{"kind_mix": ["forzaeth", "lane_switch"], "ckpt_mix": [V1], "ckpt_p": 1.0, "speed_scale": [0.8, 0.9]}]
    env = _env(slots, v2=True, opponent_speed_mode="linear")
    env.step(torch.zeros(env.B, env.act_dim))
    assert len(env.pool.seat_builders) == 1 and env.tracker.row_format.tolist() == [1, 0, 1, 0]
    env.pool.replace(0, v2_ckpt, env)
    ent = env.pool.entries[0]
    assert ent.speed_mode == "wp4" and ent.seat_key is None and env.pool.seat_builders == ()
    env.step(torch.zeros(env.B, env.act_dim))
    assert env.tracker.row_format.tolist() == [1, 1, 1, 1]         # the seat now plans wp4, read as wp4
    # a reset holds the speed in the seat's new layout
    env._reset_envs(torch.arange(env.B))
    sp = (env.prev_action[~env.learner] != 0).any(0)
    assert sp.tolist() == [False] * 4 + [True] * 4                 # wp4's four speed columns
    # and back: a v1 checkpoint replaces it again, its builder made from the env's latest step
    env.pool.replace(0, V1, env)
    seat = env.pool.seats[env.pool.entries[0].seat_key]
    assert seat.obs is not None and seat.obs["scan"].shape[1:] == (6, 1081)
    env.step(torch.zeros(env.B, env.act_dim))
    assert env.tracker.row_format.tolist() == [1, 0, 1, 0]


def test_what_a_seat_cannot_be_shown_is_refused(v2_ckpt, tmp_path):
    from f1sim.learn.opponent_pool import OpponentPool
    tr, rl = _track()
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 61
    e = EnvConfig(action_mode="plan", race_size=2, opponent="slots", collision_mode="soft", compile_tracker=False,
                  opponent_slots=[{"kind_mix": ["forzaeth"], "ckpt_mix": [v2_ckpt], "ckpt_p": 1.0}],
                  scan_stack=3, hist_len=0)
    env = common.make_env([tr], 4, "cpu", e, cfg=cfg, seed=0, rls=[rl], opponent_pool=False)
    with pytest.raises(ValueError, match="1081 beams"):
        OpponentPool.load([v2_ckpt], env)
    # a v1 checkpoint that names no observation spec cannot be shown "its own" observation in a v2 env
    from f1sim.learn.model import ActorCritic, save_checkpoint
    torch.manual_seed(0)
    bare = str(tmp_path / "bare.pt")
    save_checkpoint(bare, ActorCritic(n_stack=3, n_beams=61, proprio_dim=27, priv_dim=21, act_dim=8), {})
    cfg2 = Config(); cfg2.sim.compile = False; cfg2.sim.compile_mode = "none"; cfg2.lidar.n_beams = 61
    e2 = EnvConfig(action_mode="plan", race_size=2, opponent="slots", collision_mode="soft", compile_tracker=False,
                   opponent_slots=[{"kind_mix": ["forzaeth"], "ckpt_mix": [bare], "ckpt_p": 1.0}],
                   obs_version=2, speed_mode="wp4", scan_stack=1, hist_len=20, hist_stride=2)
    env2 = common.make_env([tr], 4, "cpu", e2, cfg=cfg2, seed=0, rls=[rl], opponent_pool=False)
    with pytest.raises(ValueError, match="records no observation spec"):
        OpponentPool.load([bare], env2)
