"""The copycat study's options in the v2 DAgger trainer, and the two smaller fixes its smoke found
(docs/research/copycat-2026-10-14.md).

* the command line: `--v2-ego-actions` (arm A: the contract without the issued actions) and
  `--action-dropout` (arm B) default to the contract (without the issued actions since 2026-10-15,
  `learn.obs.EGO_ACTIONS_DEFAULT`) / no dropout, refuse each other and nonsense, are refused by a v1
  run, and a run continuing a checkpoint (`--init`, a PPO leg) takes its contract;
  evaluate's `--speed-mode` takes wp4;
* the labels: written only for the cars the action drives (`label_rows`, `F1VecEnv.action_driven`) --
  the guard is asked for those rows and gives each the label the whole-batch call gives it -- while
  every other car's command is the env's own;
* end to end (slow, CPU): a DAgger student on the arm-A contract is a 42-wide network that `evaluate`
  drives and PPO continues on its own contract; an arm-B run records what it dropped.
"""
from __future__ import annotations

import functools
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1sim import Config, maps  # noqa: E402
from f1sim.gym_env import EnvConfig  # noqa: E402
from f1sim.learn import common, dagger_v2, ppo_v2  # noqa: E402

TRACK = "gen:competition:0"


def _parser(module):
    from f1sim.viewer.console.training_schema_export import capture_parser
    return capture_parser(module)


# ---------------------------------------------------------------- the command line
def test_the_copycat_options_default_to_the_contract_and_refuse_each_other():
    ap = _parser("f1sim.learn.dagger")
    argv = ["--arch", "v2"]
    a = ap.parse_args(argv)
    dagger_v2.prepare_args(ap, a, argv)
    # the contract's default since 2026-10-15: the ego carries no issued action
    assert a.v2_ego_actions == "off" and a.action_dropout == 0.0
    assert dagger_v2.env_kwargs(a)["ego_actions"] is False
    argv = ["--arch", "v2", "--v2-ego-actions", "off"]
    a = ap.parse_args(argv)
    dagger_v2.prepare_args(ap, a, argv)
    assert a.v2_ego_actions == "off" and dagger_v2.env_kwargs(a)["ego_actions"] is False
    argv = ["--arch", "v2", "--v2-ego-actions", "on"]               # the full contract, by name
    a = ap.parse_args(argv)
    dagger_v2.prepare_args(ap, a, argv)
    assert a.v2_ego_actions == "on" and dagger_v2.env_kwargs(a)["ego_actions"] is True
    argv = ["--arch", "v2", "--action-dropout", "0.5", "--v2-ego-actions", "on"]
    a = ap.parse_args(argv)
    dagger_v2.prepare_args(ap, a, argv)
    assert a.action_dropout == 0.5 and a.v2_ego_actions == "on" and dagger_v2.env_kwargs(a)["ego_actions"] is True
    # arm B zeroes the issued-action columns: refused where there are none -- the default included --
    # and refused as nonsense on the contract that has them
    for bad in (["--action-dropout", "0.5"], ["--action-dropout", "0.5", "--v2-ego-actions", "off"],
                ["--action-dropout", "1.5", "--v2-ego-actions", "on"],
                ["--action-dropout", "-0.1", "--v2-ego-actions", "on"]):
        argv = ["--arch", "v2", *bad]
        with pytest.raises(SystemExit):
            dagger_v2.prepare_args(ap, ap.parse_args(argv), argv)


def test_updates_per_iteration_is_a_positive_count_or_unset():
    ap = _parser("f1sim.learn.dagger")
    for argv, want in ((["--arch", "v2"], None), (["--arch", "v2", "--updates", "3000"], 3000)):
        a = ap.parse_args(argv)
        dagger_v2.prepare_args(ap, a, argv)
        assert a.updates == want
    for bad in ("0", "-5"):
        argv = ["--arch", "v2", "--updates", bad]
        with pytest.raises(SystemExit):
            dagger_v2.prepare_args(ap, ap.parse_args(argv), argv)


@pytest.mark.parametrize("flag,val", [("--v2-ego-actions", "off"), ("--action-dropout", "0.5"), ("--updates", "100")])
def test_a_v1_run_refuses_the_copycat_flags(monkeypatch, flag, val):
    from f1sim.learn import dagger
    monkeypatch.setattr(sys, "argv", ["f1sim.learn.dagger", flag, val])
    with pytest.raises(SystemExit) as e:
        dagger.main()
    assert "--arch v2" in str(e.value)


def test_a_run_that_continues_a_checkpoint_takes_its_contract(tmp_path):
    from v2_fixtures import generated_v2_checkpoint
    a_ck = generated_v2_checkpoint(str(tmp_path / "a.pt"), ego_actions=False)
    full_ck = generated_v2_checkpoint(str(tmp_path / "full.pt"), ego_actions=True)
    assert dagger_v2.ego_actions_of(a_ck) is False and dagger_v2.ego_actions_of(full_ck) is True
    ap = _parser("f1sim.learn.dagger")
    # either contract: since 2026-10-15 it is the full one that differs from the default
    for ck, want in ((a_ck, "off"), (full_ck, "on")):
        argv = ["--arch", "v2", "--init", ck]
        a = ap.parse_args(argv)
        dagger_v2.prepare_args(ap, a, argv)
        assert a.v2_ego_actions == want
    # a PPO leg has no flag of its own: the env reads the contract --init's network was trained on
    pp = _parser("f1sim.learn.ppo")
    for ck, want in ((a_ck, False), (full_ck, True)):
        argv = ["--arch", "v2", "--init", ck]
        b = pp.parse_args(argv)
        ppo_v2.prepare_args(pp, b, argv)
        assert ppo_v2.env_kwargs(b)["ego_actions"] is want
    # and a fresh network the contract's default, without the issued actions
    argv = ["--arch", "v2"]
    b = pp.parse_args(argv)
    ppo_v2.prepare_args(pp, b, argv)
    assert ppo_v2.env_kwargs(b)["ego_actions"] is False


def test_evaluate_takes_the_wp4_speed_mode():
    ap = _parser("f1sim.learn.evaluate")
    a = ap.parse_args(["--teacher", "--action-mode", "plan", "--speed-mode", "wp4"])
    assert a.speed_mode == "wp4"
    from f1sim.mpc import SPEED_MODES
    assert {"wp4", "linear", "tspeed"} <= set(SPEED_MODES)
    with pytest.raises(SystemExit):
        ap.parse_args(["--speed-mode", "nonsense"])


# ---------------------------------------------------------------- the labels
@functools.lru_cache(maxsize=1)
def _track():
    from f1sim.raceline import Raceline
    tr = maps.load(TRACK)
    return tr, Raceline.build_cached(tr)


def _env(envs=8, seed=3, beams=61, **kw):
    tr, rl = _track()
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = beams
    e = dict(action_mode="plan", race_size=2, opponent="teacher", opp_speed_range=(0.7, 0.9), collision_mode="soft",
             compile_tracker=False, max_steps=400, speed_cap=9.0, obs_version=2, speed_mode="wp4", scan_stack=1,
             hist_len=20, hist_stride=2, spawn_gap=(0.8, 1.6))
    e.update(kw)
    env = common.make_env([tr], envs, "cpu", EnvConfig(**e), cfg=cfg, seed=seed, rls=[rl])
    env.reset(seed=seed)
    return env


def test_the_action_drives_the_learners_and_the_label_rows_are_theirs():
    env = _env()
    mine = env.action_driven()
    assert torch.equal(mine, env.slot == 0)
    assert torch.equal(dagger_v2.label_rows(env), torch.arange(0, env.B, env.M))
    # what `step` does with the action: a command on another row is overwritten by the teacher's
    a0 = torch.zeros(env.B, env.act_dim)
    a1 = a0.clone(); a1[~mine] = 0.7
    out0, out1 = env._opponent_actions(a0), env._opponent_actions(a1)
    assert torch.equal(out0[~mine], out1[~mine]) and torch.equal(out0[mine], a0[mine])
    # self-play: every car is driven by the action, and every row is labelled
    sp = _env(envs=4, opponent="policy")
    assert bool(sp.action_driven().all()) and dagger_v2.label_rows(sp) is None


def test_the_guard_labels_the_learners_only_and_each_as_the_whole_batch_call_does(monkeypatch):
    from f1sim.interactive_teacher import InteractiveTeacher
    env = _env()
    teacher = common.make_teacher([_track()[1]], env)
    labeler = dagger_v2.Wp4Labels(env, teacher, "env")
    sizes = []
    real = InteractiveTeacher.plan_action

    def spy(self, state, *a, **k):
        sizes.append(int(state.shape[0]))
        return real(self, state, *a, **k)
    monkeypatch.setattr(InteractiveTeacher, "plan_action", spy)
    zero = torch.zeros(env.B, env.act_dim)
    compared = 0
    for t in range(30):
        mu = 0.8 + 0.02 * torch.arange(env.B, dtype=torch.float32)          # the dial reaches the rows
        rows = dagger_v2.label_rows(env)
        whole, v_whole = labeler(mu=mu)
        part, v_part = labeler(mu=mu, rows=rows)
        assert sizes[-2:] == [env.B, rows.numel()], "the subset call planned more than the learners"
        assert part.shape == whole.shape and v_part.shape == v_whole.shape
        other = torch.ones(env.B, dtype=torch.bool); other[rows] = False
        assert part[other].abs().sum() == 0 and not bool(v_part[other].any())
        # the reference teacher's Gauss-Newton fit reduces in another order on a smaller batch: the
        # tolerances `test_opponent_variety` measured for `plan_rows` (1e-3 path, 2e-3 speed)
        torch.testing.assert_close(part[rows][:, :4], whole[rows][:, :4], rtol=0, atol=1e-3)
        torch.testing.assert_close(part[rows][:, 4:], whole[rows][:, 4:], rtol=0, atol=2e-3)
        assert torch.equal(v_part[rows], v_whole[rows])
        compared += 1
        a = zero.clone(); a[rows] = part[rows]
        env.step(a)
    assert compared == 30


def test_a_collection_labels_the_learner_rows_only(monkeypatch):
    from f1sim.learn import conditioning as cm
    from f1sim.learn.model_v2 import ActorCriticV2
    env = _env(envs=8)
    teacher = common.make_teacher([_track()[1]], env)
    asked = []
    real = env.teacher_label

    def spy(t, *a, rows=None, **k):
        if t is teacher:                           # not the opponents' commands, which go through it too
            asked.append(None if rows is None else rows.clone())
        return real(t, *a, rows=rows, **k)
    monkeypatch.setattr(env, "teacher_label", spy)
    torch.manual_seed(0)
    model = ActorCriticV2(n_beams=61, obs=dict(vars(common.obs_spec(env))))
    buf = dagger_v2.collect(env, model, dagger_v2.Wp4Labels(env, teacher, "env"), 12, 0.5, torch.device("cpu"),
                            dagger_v2.SequenceBuffer(1), noise=0.0, dial=cm.DialDraw(env, 0.3, 0.3),
                            cspec=cm.CondSpec.from_meta(model.meta["cond"]), b8=False, near=False).finalize()
    learners = torch.arange(0, env.B, env.M)
    assert len(asked) == 12 and all(r is not None and torch.equal(r, learners) for r in asked)
    assert buf.data["label"].shape == (12, learners.numel(), 8) and buf.valid_count > 0
    assert float(buf.data["label"].abs().sum()) > 0


# ---------------------------------------------------------------- end to end (CPU)
def _run_main(module, argv, runs, monkeypatch):
    import importlib
    monkeypatch.setattr(common, "RUNS_DIR", str(runs))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(sys, "argv", [module, *argv])
    importlib.import_module(module).main()


@pytest.mark.slow
def test_an_arm_a_student_is_trained_evaluated_and_continued_on_its_own_contract(tmp_path, monkeypatch):
    from f1sim.learn.evaluate import evaluate
    from f1sim.learn.model import load_checkpoint
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        _run_main("f1sim.learn.dagger", ["--arch", "v2", "--name", "ca", "--envs", "4", "--race-size", "2",
                                         "--opponent", "teacher", "--tracks", TRACK, "--iters", "2", "--steps", "40",
                                         "--epochs", "1", "--batch", "64", "--device", "cpu", "--eager",
                                         "--wandb", "disabled", "--wp4-label", "env", "--eval-steps", "20",
                                         "--teacher-eval-steps", "20", "--speed-cap", "8", "--aux-near-car", "0",
                                         "--log-every", "1", "--v2-ego-actions", "off"],
                  tmp_path, monkeypatch)
        st = str(tmp_path / "ca" / "student_latest.pt")
        model, extra = load_checkpoint(st)
        spec = model.obs_spec
        assert spec.ego_actions is False and spec.ego_dim == 42 and model.actor.ego[0].in_features == 42
        assert extra["spec"]["ego_actions"] is False and extra["action_dropout"] == 0.0
        recs = [json.loads(x) for x in open(tmp_path / "ca" / "progress.jsonl")]
        assert [r["iter"] for r in recs] == [0, 1] and recs[0]["teacher_coll_per_km"] == recs[0]["teacher_coll_per_km"]
        res = evaluate(st, [TRACK], envs=2, steps=120, speed_cap=6.0, device="cpu", seed=5, cfg=_eager_config())
        assert res["metadata"]["env_config"]["ego_actions"] is False
        assert res["progress_rate_mps"] == res["progress_rate_mps"]
        _run_main("f1sim.learn.ppo", ["--arch", "v2", "--name", "cp", "--init", st, "--envs", "4", "--race-size",
                                      "2", "--opponent", "policy", "--tracks", TRACK, "--horizon", "16",
                                      "--minibatch", "32", "--grad-accum", "1", "--total", "64", "--epochs", "1",
                                      "--device", "cpu", "--sim-backend", "eager", "--wandb", "disabled",
                                      "--log-every", "1", "--save-every", "1", "--critic-warmup", "1",
                                      "--aux-near-car", "0", "--collision-mode", "soft"],
                  tmp_path, monkeypatch)
        m2, ex2 = load_checkpoint(str(tmp_path / "cp" / "ppo_final.pt"))
        assert m2.obs_spec == spec and ex2["arch"] == "v2"
    finally:
        torch.set_num_threads(threads)


@pytest.mark.slow
def test_an_arm_b_run_drops_half_its_sequences_action_columns_and_says_so(tmp_path, monkeypatch):
    from f1sim.learn.model import load_checkpoint
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        _run_main("f1sim.learn.dagger", ["--arch", "v2", "--name", "cb", "--envs", "4", "--race-size", "2",
                                         "--opponent", "teacher", "--tracks", TRACK, "--iters", "1", "--steps", "40",
                                         "--epochs", "4", "--batch", "64", "--device", "cpu", "--eager",
                                         "--wandb", "disabled", "--wp4-label", "env", "--eval-steps", "20",
                                         "--teacher-eval-steps", "0", "--speed-cap", "8", "--aux-near-car", "0",
                                         "--log-every", "1", "--action-dropout", "0.5", "--v2-ego-actions", "on"],
                  tmp_path, monkeypatch)
        rec = [json.loads(x) for x in open(tmp_path / "cb" / "progress.jsonl")][-1]
        assert 0.0 < rec["train/action_dropped_frac"] < 1.0
        model, extra = load_checkpoint(str(tmp_path / "cb" / "student_latest.pt"))
        assert extra["action_dropout"] == 0.5 and model.obs_spec.ego_dim == 210
    finally:
        torch.set_num_threads(threads)


def _eager_config():
    cfg = Config()
    cfg.sim.compile = False
    return cfg
