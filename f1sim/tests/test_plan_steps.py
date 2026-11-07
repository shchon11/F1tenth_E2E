"""plan_steps: every output is a plan the car can drive, and different outputs are different plans."""
import math

import pytest
import torch

from f1sim import plan_steps as PS
from f1sim import map_controller as M


@pytest.fixture(scope="module")
def lim():
    lut, nominal = M.config_lut()
    return PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu")


def _starts(n, lim, g):
    v0 = torch.rand(n, generator=g) * (PS.V_FWD + PS.V_REV) - PS.V_REV
    # a start inside the grip at its own speed (what the car can be in after any driven plan)
    dg = PS.kappa_to_steer(lim.kappa_at(v0), lim).clamp(max=lim.s_max)
    d0 = (torch.rand(n, generator=g) * 2 - 1) * dg
    return v0, d0


def test_limits_hold_for_any_output(lim):
    g = torch.Generator().manual_seed(0)
    n = 200_000
    a = torch.rand(n, PS.ACT_DIM, generator=g) * 2 - 1
    a[: n // 4] = torch.sign(a[: n // 4])                  # the corners of the output box too
    v0, d0 = _starts(n, lim, g)
    d, v, forced = PS.step_targets(a, v0, d0, lim)
    eps = 1e-4
    assert float(forced.max()) < eps
    assert float(d.abs().max()) <= lim.s_max + eps
    dd = torch.diff(torch.cat([d0[:, None], d], 1), dim=1).abs()
    assert float(dd.max()) <= lim.sv_max * PS.DT + eps
    assert float(v.max()) <= PS.V_FWD + eps and float(v.min()) >= -PS.V_REV - eps
    vv = torch.cat([v0[:, None], v], 1)
    dv = torch.diff(vv, dim=1)
    assert float(dv.abs().max()) <= max(lim.a_acc, lim.a_brake) * PS.DT + eps
    # the tyres: curvature x speed^2 within the table's steady-state lateral acceleration at that speed
    kap = PS.steer_to_kappa(d, lim).abs()
    assert bool((kap <= lim.kappa_at(v) * (1 + 1e-3) + 1e-4).all())


def test_outputs_are_distinct_plans(lim):
    g = torch.Generator().manual_seed(1)
    n = 4096
    v0, d0 = _starts(n, lim, g)
    a = torch.rand(n, PS.ACT_DIM, generator=g) * 1.8 - 0.9
    for k in (0, 4, 9):
        for col in (k, PS.N_STEP + k):
            lo, hi = a.clone(), a.clone()
            lo[:, col] -= 0.05; hi[:, col] += 0.05
            dl, vl, _ = PS.step_targets(lo, v0, d0, lim)
            dh, vh, _ = PS.step_targets(hi, v0, d0, lim)
            moved = ((dh[:, k] - dl[:, k]).abs() + (vh[:, k] - vl[:, k]).abs()) > 1e-6
            assert float(moved.float().mean()) > 0.99, (k, col, float(moved.float().mean()))


def test_zero_output_keeps_going(lim):
    v0 = torch.tensor([5.0]); d0 = torch.tensor([0.0])
    p = PS.roll(torch.zeros(1, PS.ACT_DIM), v0, d0, lim)
    assert abs(float(p["y"][0, -1])) < 1e-4
    assert abs(float(p["pv"][0, -1]) - 5.0) < 0.5               # a = 0 is the middle of [brake, accelerate]


def test_constant_turn_is_a_circle(lim):
    # steady speed 2 m/s, steering held at a value the tyres hold: the points lie on the circle of radius 1 / kappa
    v0 = torch.tensor([2.0]); d0 = torch.tensor([0.2])
    d, v, _ = PS.step_targets(torch.zeros(1, PS.ACT_DIM), v0, d0, lim)
    p = PS.roll(torch.zeros(1, PS.ACT_DIM), v0, d0, lim)
    kap = float(PS.steer_to_kappa(torch.tensor(0.2), lim))
    if float((d - 0.2).abs().max()) < 1e-3:
        r = torch.hypot(p["x"][0], p["y"][0] - 1.0 / kap)
        assert float((r - 1.0 / kap).abs().max()) < 0.01


def test_reverse_plan(lim):
    # from standing, full brake / reverse on every step: the car backs up, never faster than V_REV
    a = torch.zeros(1, PS.ACT_DIM); a[:, PS.N_STEP:] = -1.0
    p = PS.roll(a, torch.tensor([0.0]), torch.tensor([0.0]), lim)
    assert float(p["x"][0, -1]) < -0.5
    assert float(p["v"].min()) >= -PS.V_REV - 1e-4
