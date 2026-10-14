"""The v2 trainers (`learn.dagger_v2`, `learn.ppo_v2`) and the v2 path through `learn.evaluate`.

* the command line: a v2 run gets the redesign's defaults where the flags say nothing and keeps
  what they say; it refuses v1-only options and a KL leash; a v1 run refuses the v2-only flags;
* the DAgger buffer rebuilds the frame stack the env had (an episode's first frame repeated
  backwards), and its sequences carry the recurrence mask;
* the burn-in steps carry no loss -- their labels can be anything -- and B8 reaches the trunk;
* end to end on the CPU (slow): a tiny DAgger run writes a v2 student, a tiny PPO run continues it
  for two updates (the stored-state replay reproduces the rollout's log probabilities: approx KL 0
  while the actor is frozen, and the drift monitor reads 0 against itself), and `evaluate` loads
  the PPO checkpoint and drives it.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.learn import dagger_v2, ppo_v2  # noqa: E402
from f1sim.learn.obs import ObsSpecV2  # noqa: E402

TRACK = "gen:competition:0"


def _parser(module):
    from f1sim.viewer.console.training_schema_export import capture_parser
    return capture_parser(module)


# ---------------------------------------------------------------- the command line
def test_a_v2_ppo_run_gets_the_redesigns_defaults_and_keeps_what_it_was_told():
    ap = _parser("f1sim.learn.ppo")
    argv = ["--arch", "v2", "--lr", "5e-5", "--horizon", "64", "--minibatch", "256", "--grad-accum", "2"]
    a = ap.parse_args(argv)
    applied = ppo_v2.prepare_args(ap, a, argv)
    assert (a.clip_split, a.adam_eps, a.vf_clip, a.kl_coef) == (True, 1e-8, 0.0, 0.0)
    assert (a.lr, a.lr_end, a.horizon, a.minibatch, a.grad_accum) == (5e-5, 1e-5, 64, 256, 2)
    assert "lr" not in applied and applied["lr_end"] == 1e-5 and applied["clip_split"] is True
    assert (a.critic_lr, a.contact_bce, a.aux_b8, a.value_norm_beta) == (3e-4, 1.0, 0.5, 0.99)
    assert (a.action_mode, a.speed_mode, a.cond, a.scan_stack, a.hist_len) == ("plan", "wp4", "dial", 1, 20)
    for bad in (["--kl-coef", "0.05"], ["--memory", "gru"], ["--controller", "fixed_low"], ["--init-log-std", "-2"],
                ["--minibatch", "100"]):
        argv = ["--arch", "v2", *bad]
        with pytest.raises(SystemExit):
            ppo_v2.prepare_args(ap, ap.parse_args(argv), argv)
    argv = ["--arch", "v2", "--kl-coef", "0"]                     # restating the decision is fine
    ppo_v2.prepare_args(ap, ap.parse_args(argv), argv)


def test_a_v2_dagger_run_gets_burn_in_40_and_chunk_80_unless_told():
    ap = _parser("f1sim.learn.dagger")
    argv = ["--arch", "v2"]
    a = ap.parse_args(argv)
    applied = dagger_v2.prepare_args(ap, a, argv)
    assert (a.seq_burn, a.chunk_length) == (40, 80) and applied == {"chunk_length": 80, "seq_burn": 40}
    assert (a.mirror_prob, a.aux_b8, a.wp4_label, a.v2_frames) == (0.5, 0.5, "env", 1)
    argv = ["--arch", "v2", "--chunk-length", "40", "--v2-frames", "2"]
    a = ap.parse_args(argv)
    applied = dagger_v2.prepare_args(ap, a, argv)
    assert (a.seq_burn, a.chunk_length, a.scan_stack) == (40, 40, 2) and "chunk_length" not in applied
    for bad in (["--memory", "gru"], ["--speed-mode", "linear"], ["--hist-len", "20"], ["--solo-fraction", "0.3"],
                ["--wp4-label", "linear-refit", "--teacher", "interactive"]):
        argv = ["--arch", "v2", *bad]
        with pytest.raises(SystemExit):
            dagger_v2.prepare_args(ap, ap.parse_args(argv), argv)


@pytest.mark.parametrize("module,flag", [("f1sim.learn.ppo", "--critic-lr"), ("f1sim.learn.dagger", "--mirror-prob")])
def test_a_v1_run_refuses_the_v2_flags(monkeypatch, module, flag):
    import importlib
    mod = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [module, flag, "0.3"])
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert "--arch v2" in str(e.value)


# ---------------------------------------------------------------- the DAgger buffer
def test_the_buffer_rebuilds_the_frame_stack_the_env_had():
    buf = dagger_v2.SequenceBuffer(n_frames=3)
    T, L, N = 6, 2, 4
    for t in range(T):
        buf.add(scan=torch.full((L, N), float(t)), ego=torch.zeros(L, 210), label=torch.zeros(L, 8),
                valid=torch.ones(L, dtype=torch.bool), newep=torch.tensor([t == 0, t in (0, 3)]),
                c=torch.zeros(L, 1))
    buf.finalize()
    stack = buf.frames_at(torch.tensor([4, 4, 1]), torch.tensor([0, 1, 1]))
    # env 0 at t 4: frames 4, 3, 2; env 1 began an episode at 3: frames 4, 3, 3; t 1: 1, 0, 0
    assert stack[:, :, 0].tolist() == [[4, 3, 2], [4, 3, 3], [1, 0, 0]]
    s = buf.sample(3, 5, "cpu")
    assert s["scan"].shape == (5, 3, 3, N) and s["ego"].shape == (5, 3, 210) and s["keep"].shape == (5, 3)
    assert s["valid"].dtype == torch.bool and "newep" not in s


def test_burn_in_steps_carry_no_loss_and_b8_reaches_the_trunk():
    from f1sim.learn.model_v2 import ActorCriticV2
    from f1sim.learn.train_v2 import actor_trunk_parameters, loss_grad_norms
    from f1sim.mpc import PlanSpec
    torch.manual_seed(0)
    m = ActorCriticV2(n_beams=61, aux={"b8": True})
    T, n, burn = 8, 2, 3
    g = torch.Generator().manual_seed(1)
    s = {"scan": torch.rand(T, n, 1, 61, generator=g), "ego": 0.3 * torch.rand(T, n, 210, generator=g),
         "c": torch.zeros(T, n, 1), "keep": torch.ones(T, n), "valid": torch.ones(T, n, dtype=torch.bool),
         "label": torch.rand(T, n, 8, generator=g) * 2 - 1, "b8_lab": 0.2 * torch.randn(T, n, 3, 3, generator=g),
         "b8_cand": torch.ones(T, n, 3, dtype=torch.bool), "b8_age": torch.zeros(T, n, 3, dtype=torch.long)}
    kw = dict(burn=burn, plan_spec=PlanSpec(speed_mode="wp4"), v_max=10.0, coefs={"b8": 0.5}, beams=False)
    a = dagger_v2.sequence_losses(m, s, **kw)
    s2 = dict(s); s2["label"] = s["label"].clone(); s2["label"][:burn] = 0.9   # burn-in labels: anything
    b = dagger_v2.sequence_losses(m, s2, **kw)
    assert torch.equal(a["bc"], b["bc"]) and float(a["b8"].detach()) > 0
    gn = loss_grad_norms({"bc": a["bc"], "b8": 0.5 * a["b8"]}, actor_trunk_parameters(m.actor))
    assert gn["bc"] > 0 and gn["b8"] > 0
    assert float(a["b8_knowable"]) == 1.0


# ---------------------------------------------------------------- end to end (CPU)
@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("v2runs")
    return d


def _run_main(module, argv, runs, monkeypatch):
    import importlib
    from f1sim.learn import common
    monkeypatch.setattr(common, "RUNS_DIR", str(runs))
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(sys, "argv", [module, *argv])
    importlib.import_module(module).main()


@pytest.mark.slow
def test_dagger_ppo_evaluate_end_to_end_on_the_cpu(runs, monkeypatch):
    from f1sim.learn.evaluate import evaluate
    from f1sim.learn.model import load_checkpoint
    from f1sim.learn.model_v2 import ActorCriticV2
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        _run_main("f1sim.learn.dagger", ["--arch", "v2", "--name", "dg", "--envs", "4", "--race-size", "2",
                                         "--opponent", "policy", "--tracks", TRACK, "--iters", "1", "--steps", "48",
                                         "--epochs", "2", "--batch", "64", "--device", "cpu", "--eager",
                                         "--wandb", "disabled", "--wp4-label", "linear-refit", "--eval-steps", "30",
                                         "--teacher-eval-steps", "0", "--aux-near-car", "0.5", "--aux-beam-car", "0.5",
                                         "--speed-cap", "6", "--log-every", "1"], runs, monkeypatch)
        st = str(runs / "dg" / "student_latest.pt")
        model, extra = load_checkpoint(st)
        assert isinstance(model, ActorCriticV2) and extra["phase"] == "dagger" and extra["speed_mode"] == "wp4"
        assert extra["sequence"] == {"burn_in": 40, "chunk": 80} and extra["wp4_label"] == "linear-refit"
        assert model.meta["aux"] == {"b8": True, "beam_car": True, "near_car": True}
        rec = [json.loads(x) for x in open(runs / "dg" / "progress.jsonl")][-1]
        assert rec["arch"] == "v2" and rec["train/grad_bc"] > 0 and rec["train/grad_b8"] > 0
        assert all(k in rec for k in ("train/bc_lat", "train/bc_speed", "train/b8", "train/grad_near", "train/grad_beam"))
        _run_main("f1sim.learn.ppo", ["--arch", "v2", "--name", "pp", "--init", st, "--envs", "4", "--race-size",
                                      "2", "--opponent", "policy", "--tracks", TRACK, "--horizon", "48",
                                      "--minibatch", "96", "--grad-accum", "1", "--total", "384", "--epochs", "2",
                                      "--device", "cpu", "--sim-backend", "eager", "--wandb", "disabled",
                                      "--log-every", "1", "--save-every", "1", "--critic-warmup", "1",
                                      "--aux-near-car", "0.5", "--aux-beam-car", "0.5", "--car-contact-penalty", "1",
                                      "--collision-mode", "soft"], runs, monkeypatch)
        recs = [json.loads(x) for x in open(runs / "pp" / "progress.jsonl")]
        assert [r["update"] for r in recs] == [1, 2]
        first = recs[0]
        # update 1 freezes the actor (critic warm-up): replaying the stored chunk from the stored
        # state must give back the rollout's own log probabilities, and the drift monitor reads 0
        assert abs(first["loss/approx_kl"]) < 1e-6 and abs(first["loss/kl_ref"]) < 1e-6
        assert first["policy/actor_move"] == 0.0 and recs[1]["policy/actor_move"] > 0
        assert recs[1]["value_norm/std"] > 0 and 0 < first["contact/label_valid_frac"] < 1
        for k in ("grad/pg", "grad/b8", "grad/near", "grad/beam", "grad/critic_vf", "grad/critic_contact",
                  "rollout/speed_capped_frac", "loss/contact_bce", "loss/b8_knowable"):
            assert k in first, k
        ck = str(runs / "pp" / "ppo_final.pt")
        m2, ex2 = load_checkpoint(ck)
        assert isinstance(m2, ActorCriticV2) and ex2["value_norm"]["debias"] > 0 and ex2["arch"] == "v2"
        res = evaluate(ck, [TRACK], envs=2, steps=200, speed_cap=6.0, device="cpu", seed=5,
                       cfg=_eager_config())
        md = res["metadata"]
        assert md["speed_mode"] == "wp4" and md["env_config"]["obs_version"] == 2 and md["steps"] == 200
        assert res["progress_rate_mps"] == res["progress_rate_mps"]          # a number, not NaN
    finally:
        torch.set_num_threads(threads)


def _eager_config():
    from f1sim import Config
    cfg = Config()
    cfg.sim.compile = False
    return cfg
