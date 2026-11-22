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
