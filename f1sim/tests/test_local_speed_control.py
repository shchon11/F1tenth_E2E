"""Automatic local speed planning: real initial state, stage physics and final VESC request."""
from dataclasses import replace

import pytest
import torch

from f1sim import mpc
from f1sim.learn import grip_control as gc

SP = mpc.PlanSpec(N=8)
GS = gc.GripSpec(mode="estimated", profile_version="local-v2", drive_split_r=.5)
WB, SMAX, VMAX = .3302, .4189, 10.


def profile(k=None, mu=(.7,), initial=2., desired=8., length=12.):
    friction = torch.tensor(mu)
    batch = len(mu)
    if k is None:
        k = torch.zeros(batch, mpc.N_KNOTS)
    full = lambda value: torch.full((batch,), float(value))
    return gc.local_speed_profile(k, full(length), full(desired), full(desired), full(desired),
                                  friction, full(initial), GS, SP, VMAX)


def tracker(mu=(.7,), **changes):
    t = mpc.PlanTracker(len(mu), "cpu", WB, SMAX, VMAX, SP, compile_solver=False)
    g = gc.GripMPC(t, replace(GS, **changes), len(mu), "cpu", WB, SMAX, VMAX).install()
    g.update(torch.tensor(mu))
    return t, g


def action(batch=1, speed=8., curvature=0.):
    return mpc.encode(torch.full((batch, mpc.N_KNOTS), curvature), torch.full((batch,), speed),
                      torch.full((batch,), speed), VMAX, SP)


def test_future_corner_does_not_globally_reduce_straight_budget():
    k = torch.zeros(1, mpc.N_KNOTS)
    k[:, -2:] = 1.0
    _, straight = profile()
    _, corner = profile(k)
    torch.testing.assert_close(corner['a_acc'][:, 0], straight['a_acc'][:, 0])
    assert corner['a_acc'][0, 0] > corner['a_acc'][0, -1] + .1
    assert corner['v_feasible'][0, 0] > corner['v_feasible'][0, -1] + 1
    assert corner['v_reachable'][0, 1] > corner['v_reachable'][0, 0]


def test_tactical_low_target_survives_and_profiles_are_distinct():
    reach, d = profile(initial=1., desired=2.)
    assert torch.equal(d['v_desired'], torch.full_like(reach, 2.))
    assert (d['v_feasible'] <= 2.).all()
    assert (reach <= 2. + 1e-6).all()
    assert reach[0, 0] == 1.
    t, _ = tracker()
    t(action(speed=2.), torch.tensor([1.]), torch.tensor([10.]))
    assert t.last_ref[:, :, 3].max() <= 2. + 1e-6


def test_higher_mu_raises_feasible_corner_and_reachable_speed():
    k = torch.full((2, mpc.N_KNOTS), .8)
    reach, d = profile(k, mu=(.4, 1.1), initial=1.)
    assert (d['v_feasible'][1] >= d['v_feasible'][0]).all()
    assert d['v_feasible'][1, -1] > d['v_feasible'][0, -1] + .5
    assert reach[1, -1] > reach[0, -1]


def test_feasible_profile_edges_obey_both_axle_and_motor_constraints():
    k = torch.tensor([[0., 0., .3, .8, .4, .1], [0., .4, 0., -.7, -.5, 0.]])
    _, d = profile(k, mu=(.6, 1.1))
    v = d['v_feasible']
    ax = (v[:, 1:].square() - v[:, :-1].square()) / (2 * 12. / (GS.n_path - 1))
    for endpoint in (slice(None, -1), slice(1, None)):
        speed, kap = v[:, endpoint], d['kappa'][:, endpoint]
        lo, hi, bad = gc.local_accel_bounds(speed, kap, torch.tensor([.6, 1.1])[:, None], GS)
        assert not bad.any()
        assert (ax >= lo - 2e-4).all()
        assert (ax <= hi + 2e-4).all()


def test_overspeed_preserves_initial_state_and_brakes_without_soft_command_clip():
    k = torch.full((1, mpc.N_KNOTS), .8)
    reach, d = profile(k, initial=9., desired=3.)
    assert reach[0, 0] == 9.
    assert d['initial_overspeed'][0]
    assert d['path_infeasible'][0, 0]
    assert reach[0, 1] < reach[0, 0]
    t, g = tracker()
    command = t(action(speed=3., curvature=.8), torch.tensor([9.]), torch.tensor([3.]), torch.zeros(1))
    assert t.last_pred[0, 0, 3] == 9.
    assert t.last_ref[0, 0, 3] == 9.
    assert t.u_prev[0, 1] < 0
    assert 3. < command[0, 1] < 9.
    assert not g.last_diagnostics['command_infeasible'].any()


def test_actual_rollout_obeys_selected_steer_axle_and_slew_constraints():
    t, g = tracker(mu=(.4, .8, 1.2))
    t.u_seq.fill_(100.)  # intentionally impossible warm start
    t(action(3, speed=9., curvature=.8), torch.tensor([2., 5., 9.]), torch.full((3,), 10.), torch.zeros(3))
    d = g.last_diagnostics
    assert d['stage_axle_violation'].max() < 1e-5
    assert d['stage_accel_violation'].max() < 1e-5
    assert d['stage_slew_violation'].max() < 1e-6
    assert torch.isfinite(t.u_seq).all()


def test_impossible_initial_steering_recovers_at_slew_and_is_flagged():
    state = torch.tensor([[0., 0., 0., 10., .4, 0.]])
    u, _, _, bad = gc.project_local_control(state, torch.tensor([[0., -5.]]), torch.tensor([.4]), GS, SP, WB, SMAX, VMAX)
    assert bad[0]
    assert u[0, 0] == pytest.approx(.4 - GS.sv_max * SP.dt)


@pytest.mark.parametrize('speed,target,wheel', [(1., 8., 1.1), (9., 10., 9.2), (9., 2., 9.1)])
def test_final_command_inverts_nominal_motor_loop_and_respects_limits(speed, target, wheel):
    t, g = tracker()
    g.update_feedback(torch.tensor([wheel]))
    command = t(action(speed=target), torch.tensor([speed]), torch.tensor([target]))
    d = g.last_diagnostics
    actual_request = (command[:, 1] - wheel) / GS.motor_tau
    torch.testing.assert_close(actual_request, d['command_motor_request'])
    torch.testing.assert_close(actual_request - gc.resistance(torch.tensor([speed]), GS), d['command_net_accel'], atol=2e-6, rtol=1e-6)
    assert actual_request >= -GS.a_brake - 1e-5
    assert actual_request <= GS.a_max * min(1., GS.v_switch / wheel) + 1e-5
    assert not d['command_infeasible'].any()
    assert 0 <= command[0, 1] <= GS.actuator_v_max


def test_unavoidable_physical_target_saturation_is_exposed():
    t, g = tracker()
    g.update_feedback(torch.tensor([14.]))
    cmd = t(action(speed=2.), torch.tensor([14.]), torch.tensor([2.]))
    assert cmd[0, 1] == GS.actuator_v_max
    assert g.last_diagnostics['command_infeasible'][0]
    assert g.last_diagnostics['command_motor_violation'][0] > 0


def test_stage_bounds_clamp_warm_start_even_with_zero_iterations():
    sp = replace(SP, iters=0)
    z0 = torch.zeros(1, 6)
    z0[:, 3] = 2.
    ref = torch.zeros(1, sp.N + 1, 4)
    bounds = torch.zeros(1, sp.N, 2, 2)
    bounds[:, :, 0] = torch.tensor([-.1, -1.])
    bounds[:, :, 1, 0] = .1
    bounds[:, :, 1, 1] = torch.linspace(.1, 1., sp.N)
    u, _ = mpc.ilqr(z0, ref, torch.full((1, sp.N, 2), 100.), sp, WB, SMAX, VMAX, bounds=bounds)
    torch.testing.assert_close(u, bounds[:, :, 1])


def test_callback_clamps_initial_warm_start_before_linearization():
    sp = replace(SP, iters=0)
    state = torch.zeros(1, 6)
    state[:, 3] = 3.
    def project(z, u, i):
        return gc.project_local_control(z, u, torch.tensor([.7]), GS, sp, WB, SMAX, VMAX)[0]
    u, z = mpc.ilqr(state, torch.zeros(1, sp.N + 1, 4), torch.full((1, sp.N, 2), 100.),
                    sp, WB, SMAX, VMAX, projector=project)
    d = gc.stage_diagnostics(u, z, torch.tensor([.7]), GS, sp, WB)
    assert d['stage_axle_violation'].max() < 1e-5
    assert d['stage_slew_violation'].max() < 1e-6


def test_historical_conversion_and_hook_lifecycle_are_unchanged():
    t = mpc.PlanTracker(1, 'cpu', WB, SMAX, VMAX, SP, compile_solver=False)
    a = action(speed=2.)
    v, cap = torch.tensor([9.]), torch.tensor([2.])
    historical = t(a, v, cap)
    assert historical[0, 1] == 2.
    g = gc.GripMPC(t, GS, 1, 'cpu', WB, SMAX, VMAX).install()
    assert t._command_hook is not None
    g.release()
    assert t._command_hook is None
    t.reset(torch.tensor([0]))
    assert torch.equal(t(a, v, cap), historical)


def test_explicit_reference_profile_retains_initial_overspeed():
    k = torch.zeros(1, mpc.N_KNOTS)
    profile = torch.linspace(9., 3., 25)[None]
    budgets = torch.full((1, 25, 2), 2.)
    ref = mpc.reference(k, torch.tensor([12.]), torch.tensor([3.]), torch.tensor([3.]), SP,
                        torch.tensor([9.]), v_profile=profile, a_walk=budgets)
    assert ref[0, 0, 3] == 9.
    assert torch.isfinite(ref).all()


def test_reachable_profile_covers_power_limit_at_faster_edge_endpoint():
    reach, d = profile(mu=(1.1,), initial=7., desired=10., length=15.)
    ax = (reach[:, 1:].square() - reach[:, :-1].square()) / (2 * 15. / (GS.n_path - 1))
    for endpoint in (slice(None, -1), slice(1, None)):
        lo, hi, bad = gc.local_accel_bounds(reach[:, endpoint], d['reachable_kappa_bound'], torch.tensor([[1.1]]), GS)
        assert not bad.any()
        assert (ax >= lo - 1e-4).all()
        assert (ax <= hi + 1e-4).all()


def test_final_target_can_exceed_policy_scale_within_physical_actuator_limit():
    t, g = tracker()
    g.update_feedback(torch.tensor([10.5]))
    command = t(action(speed=10.), torch.tensor([10.5]), torch.tensor([10.]))
    assert 10. < command[0, 1] <= GS.actuator_v_max
    assert not g.last_diagnostics['command_infeasible'].any()


def test_pointwise_fusion_failure_is_explicit_and_not_retried_per_session(monkeypatch):
    t, g = tracker()
    attempts = []
    def unavailable():
        attempts.append(1)
        raise RuntimeError('compiler unavailable in test')
    monkeypatch.setattr(gc, '_POINTWISE_FAILURES', {})
    monkeypatch.setattr(gc, '_pointwise_kernels', unavailable)
    with pytest.warns(RuntimeWarning, match='compiler unavailable'):
        g._prepare_pointwise([])
    assert g._kernels is None
    assert g.acceleration_status['backend'] == 'eager'
    assert 'compiler unavailable' in g.acceleration_status['reason']
    with pytest.warns(RuntimeWarning, match='compiler unavailable'):
        g._prepare_pointwise([])
    assert attempts == [1]
    assert g.acceleration_status['cache_hit']
    # Failure happened before CUDA capture; the original CPU math is still usable.
    assert torch.isfinite(t(action(), torch.tensor([3.]), torch.tensor([10.]))).all()


@pytest.mark.parametrize('initial', [0., .1])
def test_zero_tactical_target_never_accelerates_from_stop_or_crawl(initial):
    t, g = tracker()
    g.update_feedback(torch.tensor([initial]))
    command = t(action(speed=0.), torch.tensor([initial]), torch.tensor([10.]))
    assert g.last_diagnostics['command_net_accel'][0] <= 1e-6
    assert t.last_ref[0, -1, 3] <= initial + 1e-6
    if initial == 0.:
        assert torch.equal(command, torch.zeros_like(command))
        assert torch.equal(t.last_ref, torch.zeros_like(t.last_ref))
    else:
        assert t.last_ref[0, -1, 0] < initial * SP.N * SP.dt


def test_positive_desired_speed_launches_from_exact_rest_with_bounded_acceleration():
    t, g = tracker()
    command = t(action(speed=8.), torch.zeros(1), torch.full((1,), 10.))
    assert command[0, 1] > 0
    assert g.last_diagnostics['command_net_accel'][0] > 0
    assert t.last_ref[0, -1, 0] > 0
    assert t.last_ref[0, 0, 3] == 0
    assert g.last_diagnostics['stage_accel_violation'].max() < 1e-5
    assert g.last_diagnostics['stage_axle_violation'].max() < 1e-5


@pytest.mark.parametrize('source', ['speed', 'wheel', 'yaw', 'delay', 'cap', 'action', 'mu', 'warm'])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_input_is_held_flagged_and_does_not_contaminate_other_cars(source, invalid):
    t, g = tracker(mu=(.7, .7))
    speed, cap, yaw, delay = torch.full((2,), 2.), torch.full((2,), 10.), torch.zeros(2), torch.full((2,), SP.delay)
    a = action(2)
    g.update_feedback(speed)
    t(a, speed, cap, yaw, delay)
    # Identical pre-fault states make the untouched second car an independent control.
    control, control_g = tracker()
    control.u_prev.copy_(t.u_prev[1:])
    control.u_seq.copy_(t.u_seq[1:])
    control_g.update_feedback(torch.tensor([2.]))
    if source == 'speed': speed[0] = invalid
    elif source == 'wheel': g.update_feedback(torch.tensor([invalid, 2.]))
    elif source == 'yaw': yaw[0] = invalid
    elif source == 'delay': delay[0] = invalid
    elif source == 'cap': cap[0] = invalid
    elif source == 'action': a[0, 0] = invalid
    elif source == 'mu': g.update(torch.tensor([invalid, .7]))
    elif source == 'warm': t.u_seq[0, 0, 0] = invalid
    result = t(a, speed, cap, yaw, delay)
    expected = control(action(), torch.tensor([2.]), torch.tensor([10.]), torch.zeros(1), torch.tensor([SP.delay]))
    assert torch.isfinite(result).all()
    assert torch.isfinite(t.last_ref).all()
    assert torch.isfinite(t.last_pred).all()
    assert g.last_diagnostics['input_fault'].tolist() == [True, False]
    assert g.last_diagnostics['command_infeasible'][0]
    torch.testing.assert_close(result[1:], expected, atol=1e-6, rtol=1e-6)
    assert g.last_diagnostics['v_desired'][0].max() == 0


def test_first_frame_fault_is_finite_stop_and_reset_clears_held_measurements_per_car():
    t, g = tracker(mu=(.7, .7))
    g.update_feedback(torch.tensor([float('nan'), 2.]))
    out = t(action(2), torch.tensor([float('nan'), 2.]), torch.full((2,), 10.))
    assert torch.equal(out[0], torch.zeros(2))
    assert g.last_diagnostics['command_infeasible'][0]
    g.update_feedback(torch.tensor([3., 2.]))
    t(action(2), torch.tensor([3., 2.]), torch.full((2,), 10.))
    t.reset(torch.tensor([0]))
    assert g._last_speed.tolist() == [0., 2.]
    assert g.wheel_feedback.tolist() == [0., 2.]
    g.update_feedback(torch.tensor([float('nan'), 2.]))
    out = t(action(2), torch.tensor([float('nan'), 2.]), torch.full((2,), 10.))
    assert torch.equal(out[0], torch.zeros(2))
    assert g.last_diagnostics['input_fault'].tolist() == [True, False]


@pytest.mark.parametrize('field', [name for name, value in vars(GS).items() if isinstance(value, (float, int))])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -float('inf')])
def test_all_grip_spec_numeric_fields_require_finite_values(field, invalid):
    with pytest.raises(ValueError, match='finite'):
        replace(GS, **{field: invalid}).validate()
