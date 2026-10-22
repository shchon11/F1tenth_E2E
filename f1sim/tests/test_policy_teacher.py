"""A trained v1 policy as a v2 student's DAgger teacher (`learn.policy_teacher`, `--wp4-label policy`, 2026-10-22).

The teacher drives off its own observation. In a v2 env, where the student drives, that observation is built by a
`TeacherSeatObsV1` that records the TEACHER's plans as the previous actions. The check: with the cars moved the same way
in both envs (external commands), the teacher in a v2 env sees -- and plans -- exactly what the same checkpoint sees and
plans in a v1 env whose own action history holds its plans. As `test_league_v2`, each env runs alone from its seeded
reset (the simulator draws sensor noise from the global generator)."""
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
from f1sim.learn.conditioning import mu_to_c  # noqa: E402
from f1sim.learn.obs import flatten_obs  # noqa: E402

TRACK = "gen:competition:0"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.join(ROOT, "checkpoints", "iccas_bootcamp_safest.pt")
needs_v1 = pytest.mark.skipif(not os.path.exists(V1), reason="archived v1 checkpoint not present")
STEPS = 60                       # two race resets at max_steps 25
SLOT = [{"kind_mix": ["forzaeth", "lane_switch"], "speed_scale": [0.8, 0.9]}]


@functools.lru_cache(maxsize=1)
def _track():
    from f1sim.raceline import Raceline
    tr = maps.load(TRACK)
    return tr, Raceline.build_cached(tr)


def _env(*, v2: bool, seed=7):
    tr, rl = _track()
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = 1081
    obs = (dict(obs_version=2, speed_mode="wp4", scan_stack=1, hist_len=20, hist_stride=2, opponent_speed_mode="linear")
           if v2 else dict(scan_stack=6, hist_len=20, hist_stride=2))
    e = EnvConfig(action_mode="plan", race_size=2, opponent="slots", opponent_slots=SLOT, collision_mode="soft",
                  compile_tracker=False, max_steps=25, speed_cap=9.0, **obs)
    return common.make_env([tr], 4, "cpu", e, cfg=cfg, seed=seed, rls=[rl])


def _commands(env, k):
    for i in torch.nonzero(env.learner).flatten().tolist():
        env.set_external_command(i, 0.05 * ((k % 7) - 3), 2.0 + 0.05 * k)


@needs_v1
def test_the_teacher_in_a_v2_env_sees_and_plans_what_it_does_in_a_v1_env():
    from f1sim.learn.policy_teacher import PolicyTeacher
    # v1 env: its own observation, its own action history -- fed the teacher's plans, as when the teacher drives
    v1 = _env(v2=False)
    t1 = PolicyTeacher(v1, V1, "cpu", dial=1.0489)    # used for its model / runtime / dial only
    v1.extra_seat_builders = ()                       # ... the v1 env's own observation is the reference
    v1.reset(seed=7)
    h = None
    ref = []
    for k in range(STEPS):
        _commands(v1, k)
        scan, pro = flatten_obs(v1._last_obs)
        c = (None if t1.dial is None else
             mu_to_c(torch.full((scan.shape[0],), t1.dial), t1.cspec).to(scan.dtype))
        a, _lp, h = t1.model.act(t1.runtime.observe(scan, pro), pro, deterministic=True, c=c, h=h)
        ref.append((scan[v1.learner].clone(), pro[v1.learner].clone(), a[v1.learner].clone()))
        _obs, _r, term, trunc, _i = v1.step(a)
        done = (term | trunc)
        if bool(done.any()):
            if t1.runtime.stateful:
                t1.runtime.reset(done)
            if h is not None:
                h = h.reset(done)
    # v2 env: the student's format; the teacher's own builder
    v2 = _env(v2=True)
    t2 = PolicyTeacher(v2, V1, "cpu", dial=1.0489)
    v2.reset(seed=7)
    got = []
    for k in range(STEPS):
        _commands(v2, k)
        scan, pro = flatten_obs(t2.seat.obs)
        a = t2.act()
        got.append((scan[v2.learner].clone(), pro[v2.learner].clone(), a[v2.learner].clone()))
        v2.step(torch.zeros(v2.B, v2.act_dim))
    bad = [(k, j) for k in range(STEPS) for j in range(3) if not torch.equal(ref[k][j], got[k][j])]
    assert bad == [], bad[:6]
    assert int(v2.ep_step.max()) < STEPS                # the episodes really reset
    assert t2.lin.speed_mode == "linear"


@needs_v1
def test_a_fresh_episode_starts_from_hold_in_the_linear_layout():
    from f1sim.learn.policy_teacher import PolicyTeacher
    from f1sim.mpc import _LAYOUTS
    env = _env(v2=True)
    t = PolicyTeacher(env, V1, "cpu", dial=1.0489)
    held = {}
    real = t.reset_rows

    def spy(ids, h):
        held["v"] = env.sim.odom.state[ids, 3].clone()          # the spawn speed the hold is built from
        held["h"] = h.clone()
        real(ids, h)

    t.reset_rows = spy
    env.reset(seed=7)
    sp, path = _LAYOUTS["linear"].speed, _LAYOUTS["linear"].path
    v_n = held["v"] / float(env.ecfg.v_max_policy) * 2 - 1
    assert torch.allclose(held["h"][:, sp], v_n[:, None].expand(-1, sp.stop - sp.start))   # v0 = v1 = spawn speed
    assert float(held["h"][:, path].abs().max()) == 0.0                                    # straight on
    assert torch.equal(t.seat.act_hist[:, -1], held["h"])     # the oldest slot still holds it after the settling step


@pytest.mark.slow
@needs_v1
def test_a_v2_dagger_run_learns_from_the_policy_teacher(tmp_path, monkeypatch):
    import importlib
    import json
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_MODE", "disabled")
        monkeypatch.setattr(sys, "argv", ["f1sim.learn.dagger", "--arch", "v2", "--name", "pt", "--envs", "4",
                                          "--race-size", "2", "--opponent", "teacher", "--spawn-gap", "0.8", "1.2",
                                          "--tracks", TRACK, "--iters", "1", "--steps", "40", "--epochs", "1",
                                          "--batch", "64", "--device", "cpu", "--eager", "--wandb", "disabled",
                                          "--wp4-label", "policy", "--label-policy", V1, "--label-policy-dial", "1.0489",
                                          "--rand-range", "vehicle.mu=1.0,1.0", "--rand-range", "vehicle.mu_f_scale=0.92,0.92",
                                          "--eval-steps", "20",
                                          "--teacher-eval-steps", "0", "--speed-cap", "8", "--aux-b8", "0",
                                          "--log-every", "1"])
        importlib.import_module("f1sim.learn.dagger").main()
        rec = [json.loads(x) for x in open(tmp_path / "pt" / "progress.jsonl")][-1]
        assert rec["kind"] == "dagger" and rec["samples"] > 0 and rec["loss"] == rec["loss"]      # finite
    finally:
        torch.set_num_threads(threads)
