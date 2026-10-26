"""A plan held on the track between decisions (`EnvConfig.decision_every`, 2026-10-27): MAP keeps following the
waypoints a row was given at its last decision, from where the car now stands in the frame they were issued in, and
the PPO advantage at a decision is the decision-level GAE (lambda 1 inside a block)."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from f1sim.learn.returns import compute_gae                                     # noqa: E402
from test_map_wp4 import follower, plan                                         # noqa: E402

V = torch.tensor([5.0, 5.0])
CAP = torch.full((2,), 9.0)
K_CURVED = torch.tensor([[0.3, 0.4, 0.4, 0.2], [0.3, 0.4, 0.4, 0.2]])
V_KNOTS = torch.tensor([[5.0, 5.5, 6.0, 6.0], [5.0, 5.5, 6.0, 6.0]])


def _pose(x, y, yaw):
    return torch.tensor([[x, y, yaw], [x, y, yaw]], dtype=torch.float32)


def test_a_held_row_ignores_the_new_action():
    a0 = plan(K_CURVED, V_KNOTS, V)
    a1 = plan(-K_CURVED, V_KNOTS + 2.0, V)
    ref = follower(2)
    c0 = ref(a0, V, CAP)                                    # the plan driven as issued
    tr = follower(2)
    tr.pose_now = _pose(0.0, 0.0, 0.0)
    assert torch.allclose(tr(a0, V, CAP), c0)               # a decision: the same as without the hold
    tr.hold_rows = torch.tensor([True, False])
    out = tr(a1, V, CAP)                                    # same pose: row 0 still drives a0, row 1 drives a1
    fresh = follower(2)
    fresh(a0, V, CAP)                                       # MAP's own state (steering rate) as in `tr`
    c1 = fresh(a1, V, CAP)
    assert torch.allclose(out[0], c0[0], atol=1e-5)
    assert torch.allclose(out[1], c1[1], atol=1e-5)
    assert not torch.allclose(c0[0], c1[0], atol=1e-3)      # the test can tell the two plans apart


def test_the_car_stands_in_the_frame_the_plan_was_issued_in():
    """The same displacement in the issue frame gives the same command wherever and however the plan was issued."""
    a0 = plan(K_CURVED, V_KNOTS, V)
    outs = []
    for (x0, y0, yaw0) in ((0.0, 0.0, 0.0), (5.0, -3.0, math.pi / 2), (-2.0, 7.0, -2.5)):
        tr = follower(2)
        tr.pose_now = _pose(x0, y0, yaw0)
        tr(a0, V, CAP)
        # 1.2 m along the issue heading and 0.1 m to its left, turned 0.15 rad
        dx, dy = 1.2, 0.1
        x = x0 + dx * math.cos(yaw0) - dy * math.sin(yaw0)
        y = y0 + dx * math.sin(yaw0) + dy * math.cos(yaw0)
        tr.pose_now = _pose(x, y, yaw0 + 0.15)
        tr.hold_rows = torch.tensor([True, True])
        outs.append(tr(plan(-K_CURVED, V_KNOTS, V), V, CAP))
    for o in outs[1:]:
        assert torch.allclose(o, outs[0], atol=1e-4)
    still = follower(2)
    still.pose_now = _pose(0.0, 0.0, 0.0)
    still(a0, V, CAP)
    still.hold_rows = torch.tensor([True, True])
    assert not torch.allclose(still(a0, V, CAP), outs[0], atol=1e-3)   # moving along the plan matters


def test_lambda_one_inside_a_block_is_the_decision_level_gae():
    torch.manual_seed(0)
    T, B, k, gamma, lam = 12, 3, 4, 0.99, 0.9
    rew, val = torch.randn(T, B), torch.randn(T + 1, B)
    done = torch.zeros(T, B); trunc = torch.zeros(T, B); fval = torch.zeros(T, B)
    dec = (torch.arange(T) % k == 0)[:, None].expand(T, B)
    dec_next = torch.cat([dec[1:], torch.ones(1, B, dtype=torch.bool)], 0)
    lam_t = torch.where(dec_next, torch.full((T, B), lam), torch.ones(T, B))
    adv = compute_gae(rew, val, done, trunc, fval, gamma, lam_t)
    # by hand, on the decisions: delta_d = sum_l gamma^l r + gamma^k V(next decision) - V(decision)
    starts = list(range(0, T, k))
    want = torch.zeros(len(starts), B)
    nxt = torch.zeros(B)
    for i in reversed(range(len(starts))):
        t0 = starts[i]
        ret = sum(gamma ** l * rew[t0 + l] for l in range(k))
        delta = ret + gamma ** k * val[t0 + k] - val[t0]
        nxt = delta + (gamma ** k) * lam * nxt
        want[i] = nxt
    assert torch.allclose(adv[starts], want, atol=1e-5)
    assert torch.allclose(compute_gae(rew, val, done, trunc, fval, gamma, lam),
                          compute_gae(rew, val, done, trunc, fval, gamma, torch.full((T, B), lam)), atol=1e-6)
