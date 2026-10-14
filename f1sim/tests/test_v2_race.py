"""The race a v2 run trains in (UNIT 5b): the opponents' plan format, the env's own speed cap, and the
boot camp's league in the v2 trainers.

* what 2026-10-14 removed stays removed: the learner's unobserved random cap (`EnvConfig.
  learner_cap_range` / `learner_cap_prob`, `--learner-cap-range` / `--learner-cap-prob`), the critic's
  speed-cap column (`privileged_v2` is 33 wide) and the BC clearance mask (`--bc-clearance`,
  `train_v2.plan_clearance`, `prop_math.prism_separation`) -- neither the env nor a command line
  takes them;
* the cap: the env's own (DAgger's fixed `--speed-cap`), not observed; a step says where it clipped
  the plan, and a collection imitates every valid label and reports its capped share;
* the command lines: v2 runs default to linear opponents in a race; v1 refuses the flag; evaluate's
  "auto" picks linear opponents for a v2 checkpoint only;
* end to end (slow): v2 PPO races a v1 league seat that its own snapshot takes over, and v2 DAgger
  logs the capped share of its collection.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import shutil
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1sim import Config, maps, mpc  # noqa: E402
from f1sim.gym_env import PRIV_V2_DIM, PRIV_V2_KEYS, EnvConfig  # noqa: E402
from f1sim.learn import common  # noqa: E402

TRACK = "gen:competition:0"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.join(ROOT, "checkpoints", "iccas_bootcamp_safest.pt")
#: The flags 2026-10-14 removed: no v2 default stood on a measurement.
REMOVED_FLAGS = {"f1sim.learn.dagger": (["--learner-cap-range", "3", "9"], ["--learner-cap-prob", "0.3"],
                                        ["--bc-clearance", "0.3"]),
                 "f1sim.learn.ppo": (["--learner-cap-range", "3", "9"], ["--learner-cap-prob", "0.3"])}


@functools.lru_cache(maxsize=1)
def _track():
    from f1sim.raceline import Raceline
    tr = maps.load(TRACK)
    return tr, Raceline.build_cached(tr)


def _env(envs=8, seed=3, beams=36, **kw):
    tr, rl = _track()
    cfg = Config(); cfg.sim.compile = False; cfg.sim.compile_mode = "none"; cfg.lidar.n_beams = beams
    e = dict(action_mode="plan", race_size=2, opponent="teacher", opp_speed_range=(0.7, 0.9), collision_mode="soft",
             compile_tracker=False, max_steps=40, speed_cap=9.0, obs_version=2, speed_mode="wp4", scan_stack=1,
             hist_len=20, hist_stride=2)
    e.update(kw)
    env = common.make_env([tr], envs, "cpu", EnvConfig(**e), cfg=cfg, seed=seed, rls=[rl])
    env.reset(seed=seed)
    return env


# ===================================================================== removed on 2026-10-14
def test_the_random_cap_its_critic_column_and_the_bc_mask_are_gone():
    from f1sim import prop_math
    from f1sim.learn import dagger_v2, ppo_v2, train_v2
    fields = {f.name for f in dataclasses.fields(EnvConfig)}
    assert not fields & {"learner_cap_range", "learner_cap_prob"}
    with pytest.raises(TypeError):
        EnvConfig(learner_cap_range=(3.0, 9.0))
    assert "speed_cap" not in PRIV_V2_KEYS and PRIV_V2_DIM == 8 + 3 * 7 + 4
    assert "bcok" not in dagger_v2.SequenceBuffer.KEYS
    for mod, names in ((dagger_v2, ("LEARNER_CAP_RANGE", "LEARNER_CAP_PROB", "cap_summary")),
                       (train_v2, ("BC_CLEARANCE_M", "plan_clearance", "rect_separation")),
                       (prop_math, ("prism_separation",))):
        assert not [n for n in names if hasattr(mod, n)], mod.__name__
    for defaults in (dagger_v2.V2_ONLY_DEFAULTS, ppo_v2.V2_ONLY_DEFAULTS):
        assert not set(defaults) & {"learner_cap_range", "learner_cap_prob", "bc_clearance"}


def test_the_critic_reads_33_columns_and_the_actor_no_cap():
    env = _env(envs=4)
    pv = env.privileged_v2()
    assert pv.shape == (env.B, PRIV_V2_DIM) == (4, 33)
    assert not hasattr(env, "learner_cap") and not hasattr(env, "learner_cap_suspended")
    assert set(env._last_obs) == {"scan", "ego"}
    env.set_speed_cap(5.0)                                               # the curriculum moves the env's cap
    assert bool((env.speed_cap == 5.0).all())
    env._reset_envs(torch.arange(env.B))                                 # and a reset draws none of its own
    assert bool((env.speed_cap[env.learner] == 5.0).all())


# ===================================================================== the env's own cap
def test_the_step_says_which_learner_steps_the_cap_clipped():
    env = _env(speed_cap=3.5)
    a = torch.zeros(env.B, env.act_dim)
    a[:, mpc.plan_layout("wp4").speed] = 0.6                          # 8 m/s asked for everywhere
    _o, _r, _t, _tr, info = env.step(a)
    assert bool(info["speed_capped"][env.learner].all())
    a[:, mpc.plan_layout("wp4").speed] = -0.4                         # 3 m/s: under the cap
    _o, _r, _t, _tr, info = env.step(a)
    assert not bool(info["speed_capped"][env.learner].any())


def test_a_collection_imitates_every_valid_label_and_reports_its_capped_share():
    from f1sim.learn import conditioning as cm
    from f1sim.learn import dagger_v2
    from f1sim.learn.model_v2 import ActorCriticV2
    env = _env(envs=8, beams=61, speed_cap=3.0, opponent_speed_mode="linear", spawn_gap=(0.8, 1.2))
    teacher = common.make_teacher([_track()[1]], env)
    torch.manual_seed(0)
    model = ActorCriticV2(n_beams=61, obs=dict(vars(common.obs_spec(env))))
    buf = dagger_v2.collect(env, model, dagger_v2.Wp4Labels(env, teacher, "env"), 30, 1.0, torch.device("cpu"),
                            dagger_v2.SequenceBuffer(1), noise=0.0, dial=cm.DialDraw(env, 0.3, 0.3),
                            cspec=cm.CondSpec.from_meta(model.meta["cond"]), b8=False, near=False,
                            beam=False).finalize()
    assert set(buf.stats) == {"speed_capped_frac"}
    assert 0.0 < buf.stats["speed_capped_frac"] <= 1.0                 # the teacher plans above a 3 m/s cap
    assert "bcok" not in buf.data and buf.valid_count > 0
    # every valid row carries the imitation loss: the weight is the validity and nothing else
    T = buf.T
    s = buf.sample(2, T, torch.device("cpu"), generator=torch.Generator().manual_seed(0))
    assert "bcok" not in s
    kw = dict(burn=0, plan_spec=env.tracker.spec, v_max=float(env.ecfg.v_max_policy), coefs={}, beams=False)
    full = dagger_v2.sequence_losses(model, s, **kw)
    none = dagger_v2.sequence_losses(model, {**s, "valid": torch.zeros_like(s["valid"])}, **kw)
    assert float(full["bc"].detach()) > 0.0 and float(none["bc"].detach()) == 0.0


# ===================================================================== held-out auxiliary metrics
def test_the_heldout_metrics_read_the_heads_in_metres():
    from f1sim.learn.train_v2 import aux_heldout_metrics
    T, m = 4, 2
    near = torch.zeros(T, m, 5)
    near[..., 0] = 1.0; near[..., 1] = 2.0 / 5.0; near[..., 2] = 0.5 / 2.0; near[..., 3] = 1.0 / 5.0
    near[0, 0, 0] = 0.0; near[0, 0, 1:] = 0.0                                # no car seen there
    pred = near.clone(); pred[..., 0] = torch.where(near[..., 0] > 0, 4.0, -4.0)
    pred[..., 2] += 0.1 / 2.0                                                # 0.1 m off sideways
    beam_lab = torch.tensor([1, 1, 0, 0, -1], dtype=torch.int8).expand(T, m, 5)
    beam_lg = torch.tensor([3.0, -3.0, -3.0, 3.0, 3.0]).expand(T, m, 5)
    lab = torch.zeros(T, m, 3, 3); lab[..., 0, :] = torch.tensor([1.0 / 5.0, -0.4 / 2.0, 0.0])
    cand = torch.zeros(T, m, 3, dtype=torch.bool); cand[..., 0] = True
    b8 = torch.zeros(T, m, 3, 4); b8[..., 2, 0] = 5.0; b8[..., 2, 1:3] = torch.tensor([1.0 / 5.0, -0.3 / 2.0])
    b8[..., :2, 0] = -5.0                                                     # the car is in slot 2
    out = {"near_car": pred, "beam_car": beam_lg, "b8": b8}
    s = {"near": near, "beam": beam_lab, "b8_lab": lab, "b8_cand": cand, "b8_age": torch.zeros(T, m, 3, dtype=torch.long)}
    got = aux_heldout_metrics(out, s, torch.full((T, m), 10, dtype=torch.long))
    assert got["near_presence_auc"] == 1.0 and got["near_rows_3m"] == T * m - 1
    assert got["near_dy_p50_3m"] == pytest.approx(0.1, abs=1e-5) and got["near_dx_p50_3m"] == pytest.approx(0.0, abs=1e-6)
    assert got["beam_precision"] == pytest.approx(0.5) and got["beam_recall"] == pytest.approx(0.5)
    assert got["b8_dy_p50_3m"] == pytest.approx(0.1, abs=1e-5) and got["b8_dx_p50_3m"] == pytest.approx(0.0, abs=1e-6)
    assert aux_heldout_metrics({}, {}) == {}


# ===================================================================== the command lines
def _parser(module):
    from f1sim.viewer.console.training_schema_export import capture_parser
    return capture_parser(module)


@pytest.mark.parametrize("module,prep", [("f1sim.learn.dagger", "dagger_v2"), ("f1sim.learn.ppo", "ppo_v2")])
def test_a_v2_run_races_linear_opponents_unless_told(module, prep):
    import importlib
    p = importlib.import_module(f"f1sim.learn.{prep}")
    ap = _parser(module)

    def run(*extra):
        argv = ["--arch", "v2", *extra]
        a = ap.parse_args(argv)
        applied = p.prepare_args(ap, a, argv)
        return a, applied
    a, applied = run("--race-size", "4", "--opponent", "teacher")
    assert a.opponent_speed_mode == "linear" and applied["opponent_speed_mode"] == "linear"
    # the perception gate's decision: one frame, every perception head on at the gate's coefficients
    assert (a.v2_frames, a.aux_b8, a.aux_near_car, a.aux_beam_car) == (1, 0.5, 0.5, 5.0)
    kw = p.env_kwargs(a)
    assert kw["opp_visibility"] is True and not set(kw) & {"learner_cap_range", "learner_cap_prob"}
    a, _ = run("--race-size", "4", "--opponent", "teacher", "--opponent-speed-mode", "wp4")
    assert a.opponent_speed_mode == "wp4"
    a, applied = run()                                                    # alone: no other car to plan for
    assert a.opponent_speed_mode is None and "opponent_speed_mode" not in applied
    a, _ = run("--race-size", "2", "--opponent", "policy")             # self-play: every car the learner
    assert a.opponent_speed_mode is None
    if prep == "ppo_v2":
        a, _ = run("--league-slots", "x.pt")                             # no longer refused (UNIT 4's refusal)
        assert a.league_slots == "x.pt"


@pytest.mark.parametrize("module", sorted(REMOVED_FLAGS))
def test_the_removed_flags_are_unknown_to_both_architectures(module):
    ap = _parser(module)
    for flag in REMOVED_FLAGS[module]:
        for argv in (flag, ["--arch", "v2", *flag]):
            with pytest.raises(SystemExit):
                ap.parse_args(argv)
    from f1sim.viewer.console.training_schema_export import SCHEMA_PATH
    schema = SCHEMA_PATH.read_text()
    assert not [d for d in ("learner_cap_range", "learner_cap_prob", "bc_clearance") if f'"dest": "{d}"' in schema]


@pytest.mark.parametrize("module,flag,val", [("f1sim.learn.ppo", "--opponent-speed-mode", ["linear"]),
                                             ("f1sim.learn.dagger", "--opponent-speed-mode", ["wp4"])])
def test_a_v1_run_refuses_the_v2_race_flags(monkeypatch, module, flag, val):
    import importlib
    mod = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [module, flag, *val])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "--arch v2" in str(e.value)


def test_evaluate_races_a_v2_checkpoint_against_linear_cars_and_leaves_v1_as_it_was():
    from f1sim.learn.evaluate import _opponent_format
    v2, v1 = {"version": 2, "n_frames": 1}, {"scan_stack": 6}
    assert _opponent_format("auto", v2, "plan", 4, "slots") == "linear"
    assert _opponent_format("auto", v2, "plan", 2, "teacher") == "linear"
    assert _opponent_format("auto", v2, "plan", 1, "policy") is None
    assert _opponent_format("auto", v2, "plan", 2, "policy") is None   # self-play: no other driver
    assert _opponent_format("auto", v1, "plan", 4, "slots") is None
    assert _opponent_format("wp4", v2, "plan", 4, "slots") == "wp4"
    assert _opponent_format(None, v2, "plan", 4, "slots") is None


# ===================================================================== end to end (CPU)
def _run_main(module, argv, runs, monkeypatch):
    import importlib
    monkeypatch.setattr(common, "RUNS_DIR", str(runs))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(sys, "argv", [module, *argv])
    importlib.import_module(module).main()


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(V1), reason="archived v1 checkpoint not present")
def test_v2_ppo_races_the_v1_league_and_its_snapshot_takes_the_seat(tmp_path, monkeypatch):
    from f1sim.learn.model import load_checkpoint
    from f1sim.learn.model_v2 import ActorCriticV2
    from v2_fixtures import generated_v2_checkpoint
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        seat = str(tmp_path / "league_seat.pt")
        shutil.copy(V1, seat)
        init = generated_v2_checkpoint(str(tmp_path / "student.pt"))
        slots = json.dumps([{"kind_mix": ["forzaeth", "lane_switch"], "ckpt_mix": [seat], "ckpt_p": 1.0,
                             "speed_scale": [0.8, 0.9]}])
        _run_main("f1sim.learn.ppo", ["--arch", "v2", "--name", "pl", "--init", init, "--envs", "4", "--race-size",
                                      "2", "--opp-slots", slots, "--tracks", TRACK, "--horizon", "16",
                                      "--minibatch", "32", "--grad-accum", "1", "--total", "64", "--epochs", "1",
                                      "--device", "cpu", "--sim-backend", "eager", "--wandb", "disabled",
                                      "--log-every", "1", "--save-every", "1", "--critic-warmup", "0",
                                      "--league-slots", seat, "--league-every", "1", "--collision-mode", "soft",
                                      "--car-contact-penalty", "1", "--cap0", "9", "--cap1", "9"],
                  tmp_path, monkeypatch)
        recs = [json.loads(x) for x in open(tmp_path / "pl" / "progress.jsonl")]
        assert [r["update"] for r in recs] == [1, 2]
        assert all(0.0 <= r["rollout/speed_capped_frac"] <= 1.0 and "rollout/learner_cap_mean" not in r
                   for r in recs)
        snap = tmp_path / "pl" / "league_0.pt"
        model, extra = load_checkpoint(str(snap))
        assert isinstance(model, ActorCriticV2) and "opt" not in extra and extra["speed_mode"] == "wp4"
        assert all(k in recs[0] for k in ("heldout/near_presence_auc", "heldout/beam_precision", "heldout/b8_dy_p50_3m"))
        _m, ex = load_checkpoint(str(tmp_path / "pl" / "ppo_final.pt"))
        assert ex["experiment"]["checkpoint_rule"] == "last"
        assert ex["experiment"]["race"] == {"opponent_speed_mode": "linear", "league_seats": [0], "league_every": 1}
        assert _m.meta["priv_dim"] == PRIV_V2_DIM
    finally:
        torch.set_num_threads(threads)


@pytest.mark.slow
def test_v2_dagger_logs_the_capped_share_of_its_collection(tmp_path, monkeypatch):
    from f1sim.learn.model import load_checkpoint
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        _run_main("f1sim.learn.dagger", ["--arch", "v2", "--name", "dm", "--envs", "4", "--race-size", "2",
                                         "--opponent", "teacher", "--spawn-gap", "0.8", "1.2", "--tracks", TRACK,
                                         "--iters", "1", "--steps", "40", "--epochs", "1", "--batch", "64",
                                         "--device", "cpu", "--eager", "--wandb", "disabled", "--wp4-label", "env",
                                         "--eval-steps", "20", "--teacher-eval-steps", "0", "--speed-cap", "8",
                                         "--aux-b8", "0", "--log-every", "1"], tmp_path, monkeypatch)
        rec = [json.loads(x) for x in open(tmp_path / "dm" / "progress.jsonl")][-1]
        assert 0.0 <= rec["collect/speed_capped_frac"] <= 1.0
        assert not [k for k in rec if k.startswith("collect/bc_")]
        _m, extra = load_checkpoint(str(tmp_path / "dm" / "student_latest.pt"))
        assert extra["race"] == {"opponent_speed_mode": "linear"} and extra["cap"] == 8.0
        assert extra["collect"] == {"speed_capped_frac": rec["collect/speed_capped_frac"]}
        assert _m.meta["priv_dim"] == PRIV_V2_DIM
        # the perception heads by default, scored held out every iteration; the last weights kept
        assert extra["checkpoint_rule"] == "last" and rec["heldout_scores_iter"] == -1
        assert all(k in rec for k in ("heldout/near_presence_auc", "heldout/beam_recall", "heldout/near_dy_p50_3m"))
        assert set(extra["heldout"]) == {k[len("heldout/"):] for k in rec if k.startswith("heldout/")}
    finally:
        torch.set_num_threads(threads)
