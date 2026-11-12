"""`Actor2(branches=2)` (11-13): two plans and a choice. The first-step turn of plan 0 is never right of plan 1's, the
plan driven is the one with the larger logit, a checkpoint keeps both, a one-plan actor is what it was, and the input
that says which plan was chosen last is written by `plan_v2` and cleared by a restart."""
import os
import types

import pytest
import torch

from f1sim import dynamics as dyn
from f1sim import map_controller as M
from f1sim import plan_steps as PS
from f1sim.learn.plan_policy import Actor2, N_SCANS, PlanObs2, STATE_DIM2, load_actor2, plan_v2

BEAMS, B = 90, 6


def _actor(branches=2):
    torch.manual_seed(0)
    a = Actor2(BEAMS, memory=16, mem_front=True, branches=branches)
    with torch.no_grad():
        a.pi.weight.normal_(0.0, 0.3)
        if branches == 2:
            a.choice.weight.normal_(0.0, 0.5)
    return a


def test_plan_0_turns_left_of_plan_1_and_the_larger_logit_drives():
    a = _actor()
    scans, state = torch.rand(B, N_SCANS, BEAMS), torch.randn(B, STATE_DIM2 + 1)
    mean, log_std, v_hat = a(scans, state, torch.zeros(B, 16))
    assert a.means_out.shape == (B, 2, PS.ACT_DIM) and a.logits_out.shape == (B, 2)
    assert mean.shape == (B, PS.ACT_DIM) and log_std.shape == (B, PS.ACT_DIM) and v_hat.shape == (B,)
    assert bool((a.means_out[:, 0, 0] > a.means_out[:, 1, 0]).all())            # a larger turn output steers left
    assert torch.equal(a.branch_out, a.logits_out.argmax(1))
    assert torch.equal(mean, a.means_out[torch.arange(B), a.branch_out])
    f = torch.randn(64, 256)                                                     # the trunk's output, many kinds of it
    mean, _, _ = a.heads(f)
    assert len(set(a.branch_out.tolist())) == 2                                  # either plan can be the one driven
    assert torch.equal(mean, a.means_out[torch.arange(64), a.logits_out.argmax(1)])
    assert bool((a.means_out[:, 0, 0] >= a.means_out[:, 1, 0]).all())            # the gap can close, never turn over
    # the two plans share nothing but the first turn's middle: the rest of each is its own output
    assert not torch.allclose(a.means_out[:, 0, 1:], a.means_out[:, 1, 1:])


def test_at_the_start_the_plans_differ_by_the_set_gap_and_are_as_likely():
    torch.manual_seed(1)
    a = Actor2(BEAMS, memory=16, mem_front=True, branches=2)
    with torch.no_grad():
        a.pi.weight.zero_()
    a(torch.rand(B, N_SCANS, BEAMS), torch.randn(B, STATE_DIM2 + 1), torch.zeros(B, 16))
    gap = a.means_out[:, 0, 0] - a.means_out[:, 1, 0]
    assert torch.allclose(gap, torch.full_like(gap, Actor2.GAP0), atol=1e-5)
    assert torch.allclose(a.logits_out, torch.zeros_like(a.logits_out))


def test_a_checkpoint_keeps_both_plans_and_a_one_plan_actor_is_what_it_was():
    a = _actor()
    ck = {"actor": a.state_dict(), "beams": BEAMS, "args": {"std0": 0.6, "memory": 16, "mem_front": True, "branches": 2}}
    b = load_actor2(ck, "cpu", check_mount=False)
    scans, state = torch.rand(B, N_SCANS, BEAMS), torch.randn(B, STATE_DIM2 + 1)
    a.eval()
    with torch.no_grad():
        m1 = a(scans, state, torch.zeros(B, 16))[0]; m2 = b(scans, state, torch.zeros(B, 16))[0]
    assert torch.equal(m1, m2) and torch.equal(a.means_out, b.means_out)
    one = _actor(branches=1)
    assert not hasattr(one, "choice") and one.pi.out_features == PS.ACT_DIM
    mean, _, _ = one(scans, state[:, :STATE_DIM2], torch.zeros(B, 16))
    assert mean.shape == (B, PS.ACT_DIM)
    ck1 = {"actor": one.state_dict(), "beams": BEAMS, "args": {"std0": 0.6, "memory": 16, "mem_front": True}}
    assert load_actor2(ck1, "cpu", check_mount=False).branches == 1


def test_a_policy_trained_before_the_two_plans_still_loads():
    path = "/home/shchon11/f1sim_runs/plan25y/plan_u150.pt"
    if not os.path.exists(path):
        pytest.skip("no local checkpoint")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = load_actor2(ck, "cpu", check_mount=False)
    assert a.branches == 1
    mean, _, _ = a(torch.rand(2, N_SCANS, int(ck["beams"])), torch.randn(2, STATE_DIM2), torch.zeros(2, a.memory))
    assert mean.shape == (2, PS.ACT_DIM)


def _env():
    lidar = types.SimpleNamespace(angles=torch.linspace(-2.3, 2.3, BEAMS))
    cfg = types.SimpleNamespace(lidar=types.SimpleNamespace(range_max=10.0))
    st = torch.zeros(B, dyn.STATE_DIM)
    sim = types.SimpleNamespace(lidar=lidar, cfg=cfg, state=st)
    res = types.SimpleNamespace(scan=torch.rand(B, BEAMS) * 10.0, imu=torch.randn(B, 4, 6))
    return types.SimpleNamespace(sim=sim, learner_ids=torch.arange(B), device=torch.device("cpu"), last_result=res)


def test_plan_v2_writes_which_plan_was_chosen_and_a_restart_clears_it():
    lut, nominal = M.config_lut()
    lim = PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu")
    a = _actor().eval()
    env = _env()
    obs = PlanObs2(env, lim)
    assert obs.prev_branch is None and obs.state().shape[1] == STATE_DIM2          # a one-plan policy's inputs: as before
    with torch.no_grad():
        u, mean, _, _, p = plan_v2(a, obs, lim)
    assert obs.state().shape[1] == STATE_DIM2 + 1
    assert torch.equal(obs.prev_branch, 1.0 - 2.0 * a.branch_out.float())          # plan 0 -> +1, plan 1 -> -1
    assert torch.equal(obs.state()[:, -1], obs.prev_branch)
    assert torch.equal(u, mean) and p["x"].shape[0] == B
    obs.reset_rows(torch.tensor([1, 4]))
    assert obs.prev_branch[[1, 4]].abs().sum() == 0 and bool((obs.prev_branch[[0, 2, 3, 5]].abs() == 1).all())
    assert obs.mem_h[[1, 4]].abs().sum() == 0 and obs.mem_h[[0, 2]].abs().sum() > 0
