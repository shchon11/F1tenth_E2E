"""A v2 PPO run holds the dial when told and records what its plans were followed by (2026-10-22).

Before, `ppo_v2.train` drew the dial per episode whatever `--dial-constant` said, and its checkpoints carried no
`plan_follower`: a v2 policy trained with MAP following its plans was then evaluated (its contract) on the iLQR.
`ppo.main` does both for v1."""
from __future__ import annotations

import json
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


def test_a_short_rollout_still_labels_the_critics_contact_head(tmp_path, monkeypatch):
    """--horizon 16 < CONTACT_HORIZON 40: every window used to run past the buffer, label_valid_frac 0, the critic's
    contact head never trained (every --horizon 32 run since 2026-10-22). The rollouts are kept and the oldest labelled
    from the ones after it, ceil(39 / 16) = 3 updates late."""
    import importlib

    from v2_fixtures import generated_v2_checkpoint
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        init = generated_v2_checkpoint(str(tmp_path / "student.pt"))
        monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_MODE", "disabled")
        monkeypatch.setattr(sys, "argv", ["f1sim.learn.ppo", "--arch", "v2", "--name", "cl", "--init", init, "--envs", "4",
                                          "--tracks", TRACK, "--horizon", "16", "--minibatch", "32", "--grad-accum", "1",
                                          "--total", str(6 * 16 * 4), "--epochs", "1", "--device", "cpu",
                                          "--sim-backend", "eager", "--wandb", "disabled", "--log-every", "1",
                                          "--save-every", "100", "--critic-warmup", "0", "--collision-mode", "soft",
                                          "--cap0", "9", "--cap1", "9", "--dial-constant", "1.0489"])
        importlib.import_module("f1sim.learn.ppo").main()
        rows = [json.loads(line) for line in open(tmp_path / "cl" / "progress.jsonl")]
        assert [r["update"] for r in rows] == [1, 2, 3, 4, 5, 6]
        assert all(r["contact/label_delay_updates"] == 3 for r in rows)
        assert all(r["contact/label_valid_frac"] == 0 for r in rows[:3])      # nothing to label from yet
        assert all(r["contact/label_valid_frac"] > 0.5 for r in rows[3:])     # the oldest rollout, labelled
    finally:
        torch.set_num_threads(threads)


def test_the_shield_hinge_pulls_the_first_speed_under_the_safe_one(tmp_path, monkeypatch):
    """--shield-aux (2026-10-24): in a race, the steps with a car in our path carry (mu_v1 - v_safe)_+^2; the run
    logs where it applied and by how much the mean first speed exceeded the safe one."""
    import importlib

    from f1sim.learn.ppo_v2 import safe_speed
    from v2_fixtures import generated_v2_checkpoint
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        init = generated_v2_checkpoint(str(tmp_path / "student.pt"))
        monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_MODE", "disabled")
        monkeypatch.setattr(sys, "argv", ["f1sim.learn.ppo", "--arch", "v2", "--name", "sh", "--init", init, "--envs", "8",
                                          "--tracks", TRACK, "--horizon", "16", "--minibatch", "32", "--grad-accum", "1",
                                          "--total", str(3 * 16 * 4), "--epochs", "1", "--device", "cpu",
                                          "--sim-backend", "eager", "--wandb", "disabled", "--log-every", "1",
                                          "--save-every", "100", "--critic-warmup", "0", "--collision-mode", "soft",
                                          "--cap0", "9", "--cap1", "9", "--dial-constant", "1.0489", "--race-size", "2",
                                          "--shield-aux", "1.0", "--shield-path-width", "3.0"])
        importlib.import_module("f1sim.learn.ppo").main()
        rows = [json.loads(line) for line in open(tmp_path / "sh" / "progress.jsonl")]
        assert all("loss/shield" in r and "loss/shield_active_frac" in r for r in rows)
        assert max(r["loss/shield_active_frac"] for r in rows) > 0          # a car was in the (wide) path somewhere
    finally:
        torch.set_num_threads(threads)


def test_the_pass_slow_ceiling_allows_a_closing_speed_and_holds_beside_the_car():
    """--shield-dv / --shield-alongside (2026-10-24): the in-lane ceiling matches the opponent's speed for a car wholly
    ahead only; pass-slow allows `dv` over it, reaches 0.8 m sideways, and holds while the opponent's centre is ahead of
    ours -- beside us too, where the in-lane one reads no car (its ceiling there held us beside the opponent)."""
    from types import SimpleNamespace

    from f1sim import dynamics as dyn
    from f1sim.learn.ppo_v2 import safe_speed

    def env_with(dx, dy, v_me, v_opp):
        st = torch.zeros(2, dyn.STATE_DIM)
        st[1, dyn.IX], st[1, dyn.IY] = dx, dy
        st[0, dyn.IVX], st[1, dyn.IVX] = v_me, v_opp
        return SimpleNamespace(sim=SimpleNamespace(state=st, other_idx=torch.tensor([[1], [0]])),
                               ecfg=SimpleNamespace(car_len=0.5))

    beside = env_with(0.3, 0.6, 5.0, 3.0)                        # centre 0.3 m ahead, overlapping, 0.6 m to the left
    assert torch.isinf(safe_speed(beside, 3.0, 0.35, 0.2, 0.2)[0])                    # in-lane: not in our path
    assert torch.isinf(safe_speed(beside, 3.0, 0.8, 0.2, 0.2, dv=1.0)[0])             # wholly-ahead test still applies
    assert safe_speed(beside, 3.0, 0.8, 0.2, 0.2, dv=1.0, alongside=True)[0] == pytest.approx(4.0)
    ahead = env_with(2.5, 0.0, 5.0, 3.0)                         # body gap 2.0 m, in lane
    room = 2.0 - 0.2 - (5.0 - 3.0) * 0.2
    assert safe_speed(ahead, 3.0, 0.35, 0.2, 0.2)[0] == pytest.approx(3.0 + (6.0 * room) ** 0.5)
    room = 2.0 - 0.2 - (5.0 - 3.0 - 1.0) * 0.2
    assert safe_speed(ahead, 3.0, 0.8, 0.2, 0.2, dv=1.0, alongside=True)[0] == pytest.approx(4.0 + (6.0 * room) ** 0.5)
    behind = env_with(-0.3, 0.6, 5.0, 3.0)                       # its centre behind ours: not ours to slow for
    assert torch.isinf(safe_speed(behind, 3.0, 0.8, 0.2, 0.2, dv=1.0, alongside=True)[0])


def test_the_shield_cost_prices_the_risky_command_on_its_own_step(tmp_path, monkeypatch):
    """--shield-cost (2026-10-24): with no hinge, a step whose sampled first speed is over the ceiling of a car in our
    path costs PER_MPS x the excess on that step's reward; the run logs how often and how much, and still monitors the
    mean's excess (loss/shield_excess_mps) with the hinge itself weighted 0."""
    import importlib
    from v2_fixtures import generated_v2_checkpoint
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        init = generated_v2_checkpoint(str(tmp_path / "student.pt"))
        monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_MODE", "disabled")
        monkeypatch.setattr(sys, "argv", ["f1sim.learn.ppo", "--arch", "v2", "--name", "sc", "--init", init, "--envs", "8",
                                          "--tracks", TRACK, "--horizon", "16", "--minibatch", "32", "--grad-accum", "1",
                                          "--total", str(3 * 16 * 4), "--epochs", "1", "--device", "cpu",
                                          "--sim-backend", "eager", "--wandb", "disabled", "--log-every", "1",
                                          "--save-every", "100", "--critic-warmup", "0", "--collision-mode", "soft",
                                          "--cap0", "9", "--cap1", "9", "--dial-constant", "1.0489", "--race-size", "2",
                                          "--shield-cost", "0.5", "--shield-path-width", "3.0", "--shield-brake", "0.5",
                                          "--shield-horizon", "1.0"])
        importlib.import_module("f1sim.learn.ppo").main()
        rows = [json.loads(line) for line in open(tmp_path / "sc" / "progress.jsonl")]
        assert all("shield_cost/per_step" in r and "loss/shield_excess_mps" in r for r in rows)
        assert max(r["shield_cost/active_frac"] for r in rows) > 0           # a car was in the (wide) path somewhere
        assert max(r["shield_cost/per_step"] for r in rows) > 0              # and some command went over its ceiling
        assert all(r["loss/shield"] == r["loss/shield"] for r in rows)        # the hinge's value is still measured
    finally:
        torch.set_num_threads(threads)


def test_the_predictive_ceiling_counts_a_car_drifting_into_our_lane_within_the_horizon():
    """--shield-horizon (2026-10-24): a car wholly ahead but beside our lane counts when its lateral offset, carried on at
    its current lateral rate, comes under the path width within the horizon -- not when it drifts away, not after the
    horizon, and not when we would be a car length past it by then. Frenet coordinates (a straight track here)."""
    from types import SimpleNamespace

    from f1sim import dynamics as dyn
    from f1sim.learn.ppo_v2 import safe_speed

    track = SimpleNamespace(project=lambda xy, tid, prev_idx=None: (xy[:, 0], xy[:, 1], torch.zeros(xy.shape[0], dtype=torch.long)),
                            cl_tangent=torch.tensor([[[1.0, 0.0]]]), length=torch.tensor([100.0]))

    def env_with(dx, dy, vy_opp, v_me=5.0, v_opp=3.0):
        st = torch.zeros(2, dyn.STATE_DIM)
        st[0, dyn.IX], st[1, dyn.IX], st[1, dyn.IY] = 10.0, 10.0 + dx, dy
        st[0, dyn.IVX], st[1, dyn.IVX], st[1, dyn.IVY] = v_me, v_opp, vy_opp
        sim = SimpleNamespace(state=st, other_idx=torch.tensor([[1], [0]]), track=track,
                              tid=torch.zeros(2, dtype=torch.long), cl_idx=torch.zeros(2, dtype=torch.long))
        return SimpleNamespace(sim=sim, ecfg=SimpleNamespace(car_len=0.5))

    drifting_in = env_with(2.5, 0.6, -0.5)                          # 0.6 m to the left, coming at 0.5 m/s: under 0.35 in 0.5 s
    want = 3.0 + (6.0 * (2.0 - 0.2 - 2.0 * 0.2)) ** 0.5
    assert safe_speed(drifting_in, 3.0, 0.35, 0.2, 0.2, horizon=1.0)[0] == pytest.approx(want)
    assert torch.isinf(safe_speed(drifting_in, 3.0, 0.35, 0.2, 0.2, horizon=0.25)[0])   # not within the horizon
    assert torch.isinf(safe_speed(drifting_in, 3.0, 0.35, 0.2, 0.2)[0])                  # the in-lane test does not see it
    assert torch.isinf(safe_speed(env_with(2.5, 0.6, +0.5), 3.0, 0.35, 0.2, 0.2, horizon=1.0)[0])   # drifting away
    late = env_with(0.7, 0.9, -0.5)                                 # in our lane after 1.1 s, when we are 1.5 m past it
    assert torch.isinf(safe_speed(late, 3.0, 0.35, 0.2, 0.2, horizon=1.2)[0])
    in_lane = env_with(2.5, 0.1, 0.0)                               # already in our lane: the in-lane ceiling, in Frenet
    assert safe_speed(in_lane, 3.0, 0.35, 0.2, 0.2, horizon=1.0)[0] == pytest.approx(want)


def test_dead_plan_columns_are_driven_at_the_mean_and_the_warm_up_holds_the_actor(tmp_path, monkeypatch):
    """--pg-dead-dims (2026-10-27): the columns the follower never executes are driven at the actor's mean (no noise)
    and carry no gradient -- their log std never moves while a live one does. And the critic warm-up holds the actor
    exactly: the restored Adam momentum used to step it while its loss was zeroed."""
    import importlib

    from f1sim import gym_env
    from f1sim.learn import model_v2
    from v2_fixtures import generated_v2_checkpoint
    dead = [3, 5, 6, 7]
    seen = {"act": [], "mu": []}
    real_step, real_dist = gym_env.F1VecEnv.step, model_v2.ActorV2.step_dist

    def step(self, action):
        seen["act"].append(action[self.learner_ids].detach().clone())
        return real_step(self, action)

    def step_dist(self, scan, ego, c, h=None):
        d, h_next = real_dist(self, scan, ego, c, h)
        seen["mu"].append(d.mean.detach().clone())
        return d, h_next

    monkeypatch.setattr(gym_env.F1VecEnv, "step", step)
    monkeypatch.setattr(model_v2.ActorV2, "step_dist", step_dist)
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        init = generated_v2_checkpoint(str(tmp_path / "student.pt"))
        monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_MODE", "disabled")
        monkeypatch.setattr(sys, "argv", ["f1sim.learn.ppo", "--arch", "v2", "--name", "dd", "--init", init, "--envs", "4",
                                          "--tracks", TRACK, "--horizon", "16", "--minibatch", "32", "--grad-accum", "1",
                                          "--total", str(3 * 16 * 4), "--epochs", "2", "--device", "cpu", "--lr", "1e-3",
                                          "--lr-end", "1e-3", "--sim-backend", "eager", "--wandb", "disabled",
                                          "--log-every", "1", "--save-every", "100", "--critic-warmup", "1",
                                          "--collision-mode", "soft", "--cap0", "9", "--cap1", "9",
                                          "--dial-constant", "1.0489", "--pg-dead-dims", ",".join(map(str, dead))])
        start = torch.load(init, map_location="cpu", weights_only=True)
        importlib.import_module("f1sim.learn.ppo").main()
        # rollout steps: the action handed to the env equals the mean on the dead columns, not on the live ones
        pairs = [(a, m) for a, m in zip(seen["act"], seen["mu"]) if a.shape == m.shape]
        assert pairs, "no rollout step was recorded"
        live = [j for j in range(8) if j not in dead]
        assert all(torch.equal(a[:, dead], m[:, dead].clamp(-1, 1)) for a, m in pairs)
        assert any(not torch.equal(a[:, live], m[:, live].clamp(-1, 1)) for a, m in pairs)
        rows = [json.loads(line) for line in open(tmp_path / "dd" / "progress.jsonl")]
        assert rows[0]["policy/actor_move"] == 0.0 and rows[-1]["policy/actor_move"] > 0.0
        end = torch.load(tmp_path / "dd" / "ppo_final.pt", map_location="cpu", weights_only=True)
        ls0, ls1 = start["state_dict"]["actor.log_std"], end["state_dict"]["actor.log_std"]
        assert torch.equal(ls0[dead], ls1[dead])
        assert not torch.equal(ls0[live], ls1[live])
    finally:
        torch.set_num_threads(threads)


def test_a_parked_self_play_car_is_kept_out_of_the_losses(tmp_path, monkeypatch):
    """A self-play seat (`kind: self`) is a learner row; with `present` < 1 its car is parked off the map in the races that
    leave the seat empty. Those steps are not driving and get weight 0 in every loss (2026-10-27); the learner's own seat
    keeps weight 1."""
    import importlib

    from f1sim.learn import ppo_v2
    from v2_fixtures import generated_v2_checkpoint
    seen = []
    real_gae = ppo_v2.compute_gae

    def gae(*args, **kw):
        f = sys._getframe(1).f_locals
        seen.append((f["buf_mask"].clone(), f["env"].slot[f["lid"]].clone()))
        return real_gae(*args, **kw)

    monkeypatch.setattr(ppo_v2, "compute_gae", gae)
    threads = torch.get_num_threads()
    torch.set_num_threads(min(2, threads))
    try:
        init = generated_v2_checkpoint(str(tmp_path / "student.pt"))
        monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_MODE", "disabled")
        monkeypatch.setattr(sys, "argv", ["f1sim.learn.ppo", "--arch", "v2", "--name", "sp", "--init", init, "--envs", "4",
                                          "--tracks", TRACK, "--horizon", "16", "--minibatch", "32", "--grad-accum", "1",
                                          "--total", str(2 * 16 * 4), "--epochs", "1", "--device", "cpu",
                                          "--sim-backend", "eager", "--wandb", "disabled", "--log-every", "1",
                                          "--save-every", "100", "--critic-warmup", "0", "--collision-mode", "soft",
                                          "--cap0", "9", "--cap1", "9", "--dial-constant", "1.0489", "--race-size", "2",
                                          "--opp-slots", '[{"kind": "self", "present": 0.0}]'])
        importlib.import_module("f1sim.learn.ppo").main()
        assert seen, "no update ran"
        for mask, slot in seen:
            assert mask.shape[1] == 4, "both seats of both races are learner rows"
            assert torch.all(mask[:, slot == 1] == 0), "the parked self-play car's steps reached the losses"
            assert torch.all(mask[:, slot == 0] == 1), "the learner's own seat was masked"
    finally:
        torch.set_num_threads(threads)
