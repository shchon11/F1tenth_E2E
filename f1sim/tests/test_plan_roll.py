"""--plan-roll (11-22): the committed action moved one step on and rolled from the committed plan's own state after its first
step re-creates the committed plan's next 0.9 s exactly, and the new last step follows it"""
import math

import torch

from f1sim import plan_steps as PS
from f1sim.learn.plan_policy import roll_shift, plan_world, PT_IDX, ROLL_IDX
from f1sim import map_controller as M


def _lim():
    lut, nominal = M.config_lut()
    return PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu")


def test_roll_shift_keeps_committed_path():
    torch.manual_seed(0)
    lim = _lim()
    B = 64
    u0 = torch.randn(B, PS.ACT_DIM) * 0.8
    v0 = torch.rand(B) * 6.0
    d0 = (torch.rand(B) - 0.5) * 0.3
    p0 = PS.roll(torch.tanh(u0), v0, d0, lim)
    st0 = torch.zeros(B, 6)                                     # pose (0, 0, 0): the plan's own frame
    x0, y0, psi0 = plan_world(p0, st0)
    # the car after the first step: exactly the plan's pose there
    k = PS.SUB
    st1 = torch.zeros(B, 6); st1[:, 0] = x0[:, k]; st1[:, 1] = y0[:, k]; st1[:, 2] = psi0[:, k]
    u_new = torch.randn(B, PS.ACT_DIM)
    u1 = roll_shift(u0, u_new)
    p1 = PS.roll(torch.tanh(u1), p0["v"][:, k], p0["delta"][:, k], lim)
    x1, y1, _ = plan_world(p1, st1)
    # the new plan's points 1..9 are the old plan's points 2..10
    i1 = PT_IDX[:-1]; i0 = PT_IDX[1:]
    err = torch.hypot(x1[:, i1] - x0[:, i0], y1[:, i1] - y0[:, i0])
    assert float(err.max()) < 1e-4, float(err.max())
    # and the new last step is the policy's: its turn and speed outputs are the ones PPO scores
    assert torch.equal(u1[:, PS.N_STEP - 1], u_new[:, PS.N_STEP - 1]) and torch.equal(u1[:, 2 * PS.N_STEP - 1], u_new[:, 2 * PS.N_STEP - 1])
    assert ROLL_IDX == (PS.N_STEP - 1, 2 * PS.N_STEP - 1)


def test_roll_shift_c_keeps_the_committed_steps():
    """--roll-grow: with c steps kept the new plan's first c points are the old plan's points 2..c+1 (c = 9: `roll_shift`,
    c = 0: the new action itself); the step committed now is the policy's step c+1 and the masks pick it"""
    from f1sim.learn.plan_policy import roll_shift_c, commit_masks, tanh_logp, tanh_logp_m
    torch.manual_seed(1)
    lim = _lim()
    B = 32
    u0 = torch.randn(B, PS.ACT_DIM) * 0.8
    v0 = torch.rand(B) * 6.0
    d0 = (torch.rand(B) - 0.5) * 0.3
    p0 = PS.roll(torch.tanh(u0), v0, d0, lim)
    x0, y0, psi0 = plan_world(p0, torch.zeros(B, 6))
    k = PS.SUB
    st1 = torch.zeros(B, 6); st1[:, 0] = x0[:, k]; st1[:, 1] = y0[:, k]; st1[:, 2] = psi0[:, k]
    u_new = torch.randn(B, PS.ACT_DIM)
    N = PS.N_STEP
    assert torch.equal(roll_shift_c(u0, u_new, N - 1), roll_shift(u0, u_new))
    assert torch.equal(roll_shift_c(u0, u_new, 0), u_new)
    for c in range(1, N):
        u1 = roll_shift_c(u0, u_new, c)
        p1 = PS.roll(torch.tanh(u1), p0["v"][:, k], p0["delta"][:, k], lim)
        x1, y1, _ = plan_world(p1, st1)
        i1 = PT_IDX[:c]; i0 = PT_IDX[1:c + 1]
        err = torch.hypot(x1[:, i1] - x0[:, i0], y1[:, i1] - y0[:, i0])
        assert float(err.max()) < 1e-4, (c, float(err.max()))
        assert torch.equal(u1[:, c:N], u_new[:, c:N]) and torch.equal(u1[:, N + c:], u_new[:, N + c:])
        m_all, m_new = commit_masks(c)
        assert m_all.sum() == 2 * (c + 1) and m_new.sum() == 2 and bool(m_new[c]) and bool(m_new[N + c])
        mean = torch.randn(B, PS.ACT_DIM); ls = torch.full((B, PS.ACT_DIM), -0.7)
        ii = [c, N + c]
        assert torch.allclose(tanh_logp_m(u1, mean, ls, m_new), tanh_logp(u1[:, ii], mean[:, ii], ls[:, ii]), atol=1e-5)
