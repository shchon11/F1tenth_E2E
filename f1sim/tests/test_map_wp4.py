"""MAP follows a v2 (wp4) plan along the iLQR's own reference geometry (`mpc.wp4_reference`)."""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim import mpc                                                            # noqa: E402
from f1sim.map_controller import GRIP_LEVELS, PLAN_WP_SPACING, POLICY_MAP, MAPParams, MAPPlanTracker, SteerLUT  # noqa: E402

WB, SMAX, VMAX = 0.3302, 0.4189, 8.0


def follower(B=2):
    steers = torch.linspace(0.0, SMAX, 12)
    vels = torch.linspace(0.5, 10.0, 8)
    g = torch.tensor(GRIP_LEVELS)[:, None, None]
    lut = SteerLUT(steers, vels, torch.tan(steers)[None, :, None] * (vels[None, None, :] ** 2) / WB * g.clamp(max=1.0))
    inner = mpc.PlanTracker(B, "cpu", WB, SMAX, VMAX, spec=mpc.PlanSpec(speed_mode="wp4"), compile_solver=False)
    tr = MAPPlanTracker(inner, lut, MAPParams.from_dict(POLICY_MAP), {"mu": 1.0489, "mu_f_scale": 0.92})
    tr.map_rows = torch.ones(B, dtype=torch.bool)
    tr.P = {"mu": torch.full((B,), 1.0489), "mu_f_scale": torch.full((B,), 0.92)}
    return tr


def plan(kappa, v_knots, v_meas):
    sp = mpc.PlanSpec(speed_mode="wp4")
    return mpc.encode_wp4(kappa, v_knots, VMAX, sp, v_meas)


def test_the_waypoints_are_the_ilqr_reference_path_and_profile():
    tr = follower(2)
    v = torch.tensor([3.0, 4.0])
    k = torch.tensor([[0.2, 0.3, 0.3, 0.1], [-0.1, -0.2, 0.0, 0.0]])
    a = plan(k, torch.tensor([[3.5, 4.0, 4.0, 3.0], [4.5, 5.0, 5.5, 5.5]]), v)
    cap = torch.full((2,), 9.0)
    path, pv, pk = tr.waypoints(a, v, cap)
    sp = tr.spec
    kk, L, vk = mpc.decode_wp4(a, v, VMAX, cap, sp)
    x, y, psi, s, kap = mpc.wp4_path(kk, L, sp)
    prof = mpc.wp4_profile(vk[:, 0], vk, L, sp)
    for b in range(2):
        for frac in (0.25, 0.5, 1.0, 2.0):                   # inside the plan and on its held extension
            s_q = frac * float(L[b])
            i = int(round(s_q / PLAN_WP_SPACING))
            j = int(torch.argmin((s[b] - i * PLAN_WP_SPACING).abs()))
            assert float(path[b, i, 0]) == pytest.approx(float(x[b, j]), abs=0.02)
            assert float(path[b, i, 1]) == pytest.approx(float(y[b, j]), abs=0.02)
            assert float(pv[b, i]) == pytest.approx(float(prof[b, j]), abs=0.05)
    assert float(pv[0, 0]) == pytest.approx(float(vk[0, 0]), abs=1e-4)   # at the start: the plan's first speed


def test_a_standing_car_is_sent_off_at_the_plans_first_speed():
    """MAP commands the speed of the point it stands on. The iLQR's profile starts at the measured speed and
    its walk accelerates; for MAP that start kept a standing car at 0 (the console, 2026-10-21)."""
    tr = follower(1)
    v = torch.zeros(1)
    a = plan(torch.zeros(1, 4), torch.tensor([[2.5, 3.0, 3.5, 3.5]]), v)
    out = tr(a, v, torch.full((1,), 9.0), yaw_rate=torch.zeros(1))
    assert float(out[0, 1]) > 2.0


def test_a_wp4_plan_is_driven_toward_its_side_at_its_speed():
    tr = follower(2)
    v = torch.tensor([3.0, 3.0])
    k = torch.tensor([[0.3] * 4, [-0.3] * 4])
    a = plan(k, torch.full((2, 4), 3.0), v)
    out = tr(a, v, torch.full((2,), 9.0), yaw_rate=torch.zeros(2))
    assert torch.isfinite(out).all()
    assert float(out[0, 0]) > 0.02 and float(out[1, 0]) < -0.02          # left plan steers left, right plan right
    assert float(out[0, 1]) == pytest.approx(3.0, abs=0.6)
