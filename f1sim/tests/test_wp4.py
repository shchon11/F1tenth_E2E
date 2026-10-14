"""The "wp4" plan: four waypoints x (curvature, speed) over the tracker's own 0.6 s window.

docs/research/output-format-2026-10-14.md section 3 is the definition and section 7 test 1 the gate
this file is: the encoding round-trips, the decoder is a projection onto plans the car can brake
through, the tracker reference is `mpc.reference`'s own walk, and the simulator and the car's
controller build the same tracker and command the same thing. The graph half of that last claim
(the real `controller_node`) is `test_graph_parity.py`.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from f1sim import mpc

SPEC = mpc.PlanSpec(speed_mode="wp4")
V_MAX = 10.0


def _plans():
    """Representative waypoint plans: (curvature (B, 4) [1/m], measured speed (B,), waypoint speeds
    (B, 4) [m/s]), every profile one the backward pass leaves alone -- a cruise, a launch from a
    standstill, a braking zone at 5 m/s^2's edge, a stop at the end of the window -- on a straight,
    arcs, an S-bend and a tightening corner."""
    k = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.3, 0.3, 0.3, 0.3], [-1.0, -1.0, -1.0, -1.0],
                      [0.8, 0.2, -0.5, -0.8], [0.1, 0.4, 0.9, 1.4], [-0.2, -0.6, -1.2, -1.55]])
    vm = torch.tensor([5.0, 2.0, 8.0, 3.0, 0.0, 6.5])
    v = torch.tensor([[5.0, 5.0, 5.0, 5.0], [3.0, 4.0, 5.0, 6.0], [7.0, 6.2, 5.3, 4.2],
                      [2.5, 1.8, 1.0, 0.0], [0.5, 1.0, 1.5, 2.0], [5.8, 5.0, 4.2, 3.5]])
    return k, vm, v


def test_the_plan_is_as_long_as_the_window_the_tracker_executes():
    assert SPEC.wp_horizon_s == pytest.approx(SPEC.N * SPEC.dt)        # 12 x 50 ms
    L = mpc.plan_length(torch.tensor([0.0, 1.0, 5.0, 9.0, 12.0, -4.0]), SPEC)
    assert torch.allclose(L, torch.tensor([1.0, 1.0, 3.0, 5.4, 6.0, 2.4]))
    assert mpc.act_dim("wp4") == 8 and mpc.plan_layout("wp4").path == slice(0, 4)


def test_encode_then_decode_returns_the_plan():
    k, vm, v = _plans()
    a = mpc.encode_wp4(k, v, V_MAX, SPEC)
    k2, L, v2 = mpc.decode_wp4(a, vm, V_MAX, torch.full_like(vm, 9.0), SPEC)
    assert (k2 - k).abs().max() < 1e-5
    assert (v2 - v).abs().max() < 1e-5
    assert torch.allclose(L, mpc.plan_length(vm, SPEC))
    # and the other way round, in the normalized action
    assert (mpc.encode_wp4(k2, v2, V_MAX, SPEC) - a).abs().max() < 1e-5


def test_the_decoder_is_a_projection_onto_plans_the_car_can_brake_through():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(512, 8, generator=g) * 2.2 - 1.1                   # past the box, too
    vm = torch.rand(512, generator=g) * 10.0
    cap = torch.rand(512, generator=g) * 8.0 + 1.0
    k, L, v = mpc.decode_wp4(a, vm, V_MAX, cap, SPEC)
    assert k.abs().max() <= SPEC.kappa_max + 1e-6
    assert (v <= cap[:, None] + 1e-6).all() and (v >= 0).all()
    ds = (L / mpc.WP_K)[:, None]
    assert (v[:, :-1] ** 2 <= v[:, 1:] ** 2 + 2 * SPEC.a_brake * ds + 1e-4).all()
    # decoding what it decoded changes nothing: the second pass finds nothing to brake for
    k2, _, v2 = mpc.decode_wp4(mpc.encode_wp4(k, v, V_MAX, SPEC), vm, V_MAX, cap, SPEC)
    assert (k2 - k).abs().max() < 1e-5 and (v2 - v).abs().max() < 1e-5


def test_a_stop_at_the_end_of_the_window_is_a_braking_curve_from_now():
    """v_4 = 0 at 9 m/s: every waypoint before it comes down to what 5 m/s^2 can still shed -- the
    end of the plan is a promise the plan keeps from its first waypoint on."""
    vm = torch.tensor([9.0])
    a = mpc.encode_wp4(torch.zeros(1, 4), torch.tensor([[9.0, 9.0, 9.0, 0.0]]), V_MAX, SPEC)
    _, L, v = mpc.decode_wp4(a, vm, V_MAX, torch.tensor([9.0]), SPEC)
    ds = float(L[0]) / 4
    want = [math.sqrt(2 * SPEC.a_brake * ds * n) for n in (3, 2, 1, 0)]
    assert np.allclose(v[0].numpy(), want, atol=1e-5)


def test_the_path_holds_the_first_curvature_then_runs_linear_and_extends_on_the_last():
    k = torch.tensor([[0.8, 0.2, -0.5, -0.8]])
    L = torch.tensor([3.0])
    x, y, psi, s, kap = mpc.wp4_path(k, L, SPEC)
    n_wp = SPEC.wp_grid // mpc.WP_K                                   # samples per waypoint interval
    assert torch.allclose(kap[0, :n_wp + 1], torch.full((n_wp + 1,), 0.8))          # [0, s_1]
    for j in range(4):
        assert abs(float(kap[0, (j + 1) * n_wp]) - float(k[0, j])) < 1e-6           # at s_k
    assert torch.allclose(kap[0, 4 * n_wp:], torch.full_like(kap[0, 4 * n_wp:], -0.8))  # past s_4
    assert float(s[0, -1]) == pytest.approx(SPEC.wp_extend * 3.0)
    # heading is exact at the waypoints: the curvature is piecewise linear between samples
    d = 3.0 / 4
    want = [0.8 * d, 0.8 * d + 0.5 * d, 0.8 * d + 0.5 * d - 0.15 * d, 0.8 * d + 0.5 * d - 0.15 * d - 0.65 * d]
    got = [float(psi[0, (j + 1) * n_wp]) for j in range(4)]
    assert np.allclose(got, want, atol=1e-5)


def test_a_constant_curvature_plan_is_a_circle():
    kap = 1.0
    x, y, psi, s, _ = mpc.wp4_path(torch.full((1, 4), kap), torch.tensor([3.0]), SPEC, extend=1.0)
    s = s[0].numpy()
    assert np.allclose(psi[0].numpy(), kap * s, atol=1e-5)
    # the trapezoid's position error over 3 m at 1/16 m steps is ~1e-3 m
    assert np.allclose(x[0].numpy(), np.sin(kap * s) / kap, atol=2e-3)
    assert np.allclose(y[0].numpy(), (1 - np.cos(kap * s)) / kap, atol=2e-3)


def test_the_fit_returns_a_path_the_format_can_hold():
    g = torch.Generator().manual_seed(1)
    B = 64
    v = torch.rand(B, generator=g) * 8
    k_true = (torch.rand(B, 4, generator=g) * 2 - 1) * 1.2
    L = mpc.plan_length(v, SPEC)
    x, y, _, _, _ = mpc.wp4_path(k_true, L, SPEC, extend=1.0)
    si = torch.arange(1, mpc.WP_FIT_POINTS + 1) * (SPEC.wp_grid // mpc.WP_FIT_POINTS)
    k = mpc.fit_wp4(k_true + 0.3 * torch.randn(B, 4, generator=g), x[:, si], y[:, si], L, SPEC)
    assert (k - k_true).abs().max() < 5e-5                         # measured 8e-6 1/m


def test_the_refit_keeps_the_teacher_s_path_over_the_window():
    """The label's path is the linear plan's own first 0.6 s, refitted (design 5)."""
    sl = mpc.PlanSpec()
    g = torch.Generator().manual_seed(2)
    B = 256
    v = torch.rand(B, generator=g) * 9
    Lp, L = mpc.plan_length(v, sl), mpc.plan_length(v, SPEC)
    assert (L <= Lp).all()                                        # the window is inside the source
    k6 = torch.full((B, 6), 0.7)
    assert (mpc.wp4_refit(k6, Lp, v, SPEC) - 0.7).abs().max() < 1e-5        # an arc stays that arc
    k6 = (torch.rand(B, 6, generator=g) * 2 - 1) * 0.6           # rougher than any teacher's plan
    kw = mpc.wp4_refit(k6, Lp, v, SPEC)
    x, y, _, s, _ = mpc.wp4_path(kw, L, SPEC, extend=1.0)
    xs, ys, _, ss = mpc.path_points(k6, Lp, n=241)
    dev = [np.hypot(x[b].numpy() - np.interp(s[b].numpy(), ss[b].numpy(), xs[b].numpy()),
                    y[b].numpy() - np.interp(s[b].numpy(), ss[b].numpy(), ys[b].numpy())).max()
           for b in range(B)]
    # measured: p50 4 mm, p90 2.5 cm, worst 6.2 cm -- the held first interval and the coarser knots
    assert np.median(dev) < 0.01 and max(dev) < 0.08


def _ref(k, vm, v, cap=9.0):
    a = mpc.encode_wp4(k, v, V_MAX, SPEC)
    return mpc.wp4_reference(a, vm, V_MAX, torch.full_like(vm, cap), SPEC)


def test_the_reference_walks_the_profile():
    t = torch.arange(SPEC.N + 1) * SPEC.dt
    ref = _ref(torch.zeros(1, 4), torch.tensor([5.0]), torch.full((1, 4), 5.0))   # cruise, straight
    assert torch.allclose(ref[0, :, 3], torch.full((SPEC.N + 1,), 5.0), atol=1e-5)
    assert torch.allclose(ref[0, :, 0], 5.0 * t, atol=1e-4) and ref[0, :, 1].abs().max() < 1e-6
    # a standstill asked to stay: the target is 0, the reference creeps at the linear plan's 0.3 m/s
    ref = _ref(torch.zeros(1, 4), torch.tensor([0.0]), torch.zeros(1, 4))
    assert ref[0, :, 3].abs().max() < 1e-6
    assert torch.allclose(ref[0, :, 0], SPEC.wp_walk_floor * t, atol=1e-5)
    # braking for a stop at the end: the target speed falls monotonically from the measured one
    ref = _ref(torch.zeros(1, 4), torch.tensor([6.0]), torch.tensor([[5.0, 4.0, 2.5, 0.0]]))
    vt = ref[0, :, 3]
    assert float(vt[0]) == pytest.approx(6.0, abs=1e-4) and (vt[1:] <= vt[:-1] + 1e-6).all()


def test_a_walk_past_the_plan_keeps_turning():
    """From 1.7 m/s asked for everything the walk runs 2.1 plan lengths: past s_4 it reads kappa_4,
    so a corner the plan was entering stays a corner instead of becoming a straight."""
    ref = _ref(torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.tensor([1.7]), torch.full((1, 4), 10.0))
    s_end = float(ref[0, -1, 0])                                     # x along the arc start
    assert s_end > 1.5                                              # well past the 1.02 m plan
    dpsi = ref[0, 1:, 2] - ref[0, :-1, 2]
    assert float(dpsi[-1]) > 0.05                                    # still turning at the end


def _tracker(spec):
    return mpc.PlanTracker(4, "cpu", 0.3302, 0.4189, V_MAX, spec=spec, compile_solver=False)


def test_the_tracker_drives_a_wp4_plan():
    tr = _tracker(SPEC)
    k = torch.tensor([[0.0] * 4, [0.6] * 4, [-0.6] * 4, [0.0] * 4])
    v = torch.tensor([[5.0] * 4, [5.0] * 4, [5.0] * 4, [4.0, 3.0, 1.5, 0.0]])
    vm = torch.full((4,), 5.0)
    a = mpc.encode_wp4(k, v, V_MAX, SPEC)
    for _ in range(5):
        cmd = tr(a, vm, torch.full((4,), 9.0), torch.zeros(4))
    assert abs(float(cmd[0, 0])) < 0.02 and abs(float(cmd[0, 1]) - 5.0) < 0.3
    assert float(cmd[1, 0]) > 0.05 and float(cmd[2, 0]) < -0.05     # the curvature steers, both ways
    assert float(cmd[3, 1]) < 4.6                                   # a stop ahead brakes now
    assert tr.last_ref.shape == (4, SPEC.N + 1, 4)


def test_the_sim_and_the_car_build_the_same_tracker_and_command_the_same():
    """The env's tracker (built from EnvConfig) and the one the controller node builds from a wp4
    checkpoint's contract: the same spec, and bit-identical commands over a stateful sequence."""
    from test_profile_executor import _open_floor
    from f1sim.gym_env import EnvConfig
    from f1sim.learn import common
    from f1sim.params import Config
    cfg = Config(); cfg.sim.compile = False; cfg.rand.enabled = False
    env = common.make_env([_open_floor()], 4, "cpu", EnvConfig(action_mode="plan", speed_mode="wp4",
                                                               compile_tracker=False), cfg=cfg, seed=1)
    car_spec = mpc.contract_spec(mpc.contract_of({"speed_mode": "wp4"}))
    assert env.tracker.spec == car_spec and env.act_dim == 8
    sim, car = env.tracker, _tracker(car_spec)
    sim.reset(torch.arange(4))
    g = torch.Generator().manual_seed(3)
    for _ in range(25):
        a = torch.rand(4, 8, generator=g) * 2 - 1
        vm = torch.rand(4, generator=g) * 8
        cap, yaw = torch.full((4,), 9.0), torch.randn(4, generator=g) * 0.5
        delay = torch.rand(4, generator=g) * 0.06
        c_sim = sim(a, vm, cap, yaw, delay=delay)
        c_car = car(a, vm, cap, yaw, delay=delay)
        assert torch.equal(c_sim, c_car) and torch.equal(sim.last_ref, car.last_ref)


def test_a_wp4_plan_is_never_read_as_a_linear_one():
    """Every reader that would take the numbers by the linear layout refuses instead."""
    a = torch.zeros(2, 8); vm = torch.ones(2); cap = torch.full((2,), 9.0)
    with pytest.raises(ValueError):
        mpc.decode(a, vm, V_MAX, cap, SPEC)                        # the arms' and viewers' decoder
    with pytest.raises(ValueError):
        mpc.decode_wp4(a, vm, V_MAX, cap, mpc.PlanSpec())
    with pytest.raises(ValueError, match="wp4"):
        mpc.solve(a, vm, cap, torch.zeros(2), torch.full((2,), 0.035), torch.zeros(2, 2), torch.zeros(2, 12, 2),
                  SPEC, 0.33, 0.42, V_MAX, v_limit=torch.ones(2, 25))       # an arm's speed limit
    # The raceline teacher used to refuse a wp4 spec here; it now encodes one itself
    # (`RacelineTeacher._plan_wp4`, tests/test_wp4_teachers.py).


def test_the_wp4_solve_traces_as_one_graph():
    """No host read and no data-dependent Python anywhere in the wp4 branch: dynamo traces `solve`
    whole (`fullgraph=True`), which is what a CUDA-graph capture of it needs."""
    g = torch.Generator().manual_seed(4)
    B = 16
    args = (torch.rand(B, 8, generator=g) * 2 - 1, torch.rand(B, generator=g) * 8, torch.full((B,), 9.0),
            torch.randn(B, generator=g), torch.full((B,), 0.035), torch.zeros(B, 2), torch.zeros(B, 12, 2),
            SPEC, 0.33, 0.42, V_MAX)
    f = torch.compile(mpc.solve, fullgraph=True, backend="eager")
    u1, z1, r1 = f(*args)
    u0, z0, r0 = mpc.solve(*args)
    assert torch.equal(u1, u0) and torch.equal(r1, r0)
