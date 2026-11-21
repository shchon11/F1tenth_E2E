"""(11-21) the pace levels (`Actor2(pace_caps=...)`, plan_rl --pace-caps): a level picked every `pace_hold` decisions caps
every plan drawn under it (`plan_steps.roll` cap)."""
import math

import pytest
import torch

from f1sim import plan_steps as PS
from f1sim import map_controller as M
from f1sim.learn.plan_policy import Actor2, STATE_DIM2, N_SCANS, load_actor2


@pytest.fixture(scope="module")
def lim():
    lut, nominal = M.config_lut()
    return PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu")


def test_actor_pace_head_and_floor():
    a = Actor2(1080, pace_caps="6.5,5.5,4.5", pace_hold=10, pace_floor=0.05)
    assert a.pace_levels == 4
    assert torch.isinf(a.pace_v[0]) and torch.allclose(a.pace_v[1:], torch.tensor([6.5, 5.5, 4.5]))
    sd = STATE_DIM2 + 5                                            # the level in force (one-hot of 4) and the hold left
    mean, log_std, v_hat = a(torch.rand(3, N_SCANS, 1080) * 5.0, torch.zeros(3, sd))
    assert a.pace_logits_out.shape == (3, 4)
    p = a.pace_p(a.pace_logits_out)
    assert torch.allclose(p.sum(1), torch.ones(3)) and float(p.min()) >= 0.05 - 1e-6
    assert int(a.pace_logits_out.argmax(1)[0]) == 0                # no cap the likeliest at the start
    p2 = a.pace_p(torch.tensor([[50.0, 0.0, 0.0, 0.0]]))           # a sure no-cap still leaves every cap its floor
    assert float(p2[0, 1:].min()) >= 0.05 - 1e-6


def test_no_pace_unchanged():
    a = Actor2(1080)
    assert a.pace_levels == 0 and not hasattr(a, "pace_v")
    a(torch.rand(2, N_SCANS, 1080), torch.zeros(2, STATE_DIM2))
    assert a.pace_logits_out is None


def test_roll_cap_brakes_to_the_cap(lim):
    act = torch.ones(2, PS.ACT_DIM)                                 # full speed asked at every step
    v0 = torch.tensor([8.0, 8.0]); d0 = torch.zeros(2)
    free = PS.roll(act, v0, d0, lim)
    capped = PS.roll(act, v0, d0, lim, torch.tensor([math.inf, 5.5]))
    assert torch.allclose(free["pv"][0], capped["pv"][0])          # no cap: the same plan
    pv = capped["pv"][1]
    assert float(pv[0]) < 8.0                                       # braking at once
    assert float(pv[-1]) <= 5.5 + 1e-4                              # and at the cap once reachable
    assert float((pv[:-1] - pv[1:]).max()) <= lim.a_brake * PS.DT + 1e-4      # never harder than the brakes


def test_load_actor2_keeps_pace():
    a = Actor2(1080, pace_caps="6.5,5.5", pace_hold=7, pace_floor=0.04)
    ck = {"beams": 1080, "actor": a.state_dict(), "args": {"pace_caps": "6.5,5.5", "pace_hold": 7, "pace_floor": 0.04}}
    b = load_actor2(ck, "cpu", check_mount=False)
    assert b.pace_levels == 3 and b.pace_hold == 7 and abs(b.pace_floor - 0.04) < 1e-9
    assert torch.allclose(b.pace_v[1:], torch.tensor([6.5, 5.5]))
