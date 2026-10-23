"""A v2 PPO run holds the dial when told and records what its plans were followed by (2026-10-22).

Before, `ppo_v2.train` drew the dial per episode whatever `--dial-constant` said, and its checkpoints carried no
`plan_follower`: a v2 policy trained with MAP following its plans was then evaluated (its contract) on the iLQR.
`ppo.main` does both for v1."""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1sim.learn import common  # noqa: E402
from f1sim.mpc import contract_of  # noqa: E402

TRACK = "gen:competition:0"


@pytest.mark.slow
def test_a_v2_run_holds_its_dial_and_records_its_follower(tmp_path, monkeypatch):
    import importlib

    from f1sim.learn import conditioning as cond_mod
    from f1sim.learn.model import load_checkpoint
    from v2_fixtures import generated_v2_checkpoint

    def refuse(*a, **k):
        raise AssertionError("--dial-constant was given: the dial must not be drawn")

    monkeypatch.setattr(cond_mod, "DialDraw", refuse)
    held = []
    real_const = cond_mod.ConstantDial

    class Spy(real_const):
        def __init__(self, env, value):
            held.append(float(value))
            super().__init__(env, value)

    monkeypatch.setattr(cond_mod, "ConstantDial", Spy)
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        init = generated_v2_checkpoint(str(tmp_path / "student.pt"))
        monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_MODE", "disabled")
        mp = '{"q_l1":0.4}'
        monkeypatch.setattr(sys, "argv", ["f1sim.learn.ppo", "--arch", "v2", "--name", "dc", "--init", init, "--envs", "4",
                                          "--tracks", TRACK, "--horizon", "16", "--minibatch", "32", "--grad-accum", "1",
                                          "--total", "64", "--epochs", "1", "--device", "cpu", "--sim-backend", "eager",
                                          "--wandb", "disabled", "--log-every", "1", "--save-every", "1",
                                          "--critic-warmup", "0", "--collision-mode", "soft", "--cap0", "9", "--cap1", "9",
                                          "--dial-constant", "1.0489", "--rand-range", "vehicle.mu=1.0,1.0",
                                          "--plan-follower", "map", "--plan-follower-map", mp])
        importlib.import_module("f1sim.learn.ppo").main()
        assert held == [pytest.approx(1.0489)]
        _m, ex = load_checkpoint(str(tmp_path / "dc" / "ppo_final.pt"))
        c = contract_of(ex)
        assert c["speed_mode"] == "wp4" and c["plan_follower"] == "map"
        assert c["plan_follower_map"]["q_l1"] == pytest.approx(0.4)
        assert ex["experiment"]["dial_constant"] == pytest.approx(1.0489)
    finally:
        torch.set_num_threads(threads)


def test_a_constraint_can_start_at_its_own_multiplier():
    """`--lagrange car=1.0@300`: the car constraint starts at 300 while the wall one keeps --lagrange-init (2026-10-23)."""
    from f1sim.learn.ppo import parse_lagrange
    assert parse_lagrange("wall=1.0,car=1.0@300", 80.0) == {"wall": (1.0, 80.0), "car": (1.0, 300.0)}
    assert parse_lagrange(" wall=0.5 , car=2 ", 8.0) == {"wall": (0.5, 8.0), "car": (2.0, 8.0)}
    assert parse_lagrange("", 8.0) == {}
