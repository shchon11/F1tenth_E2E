"""Friction-informed controller: default parity, envelope shape, bounds, and the live mu input.

The experiment's whole claim rests on `fixed` and `oracle` being the same algorithm with a different
number, and on `legacy` being untouched. These pin both.
"""
from __future__ import annotations

import math

import pytest
import torch

from f1sim import mpc
from f1sim.learn import grip_control as gc

SPEC = mpc.PlanSpec()
WB, SMAX, VMAX, B = 0.3302, 0.4189, 10.0, 4


def _args(B=B, curvature=0.0, v=4.0, seed=0):
    torch.manual_seed(seed)
    a = torch.zeros(B, mpc.ACT_DIM)
    a[:, :mpc.N_KNOTS] = curvature / SPEC.kappa_max        # decode multiplies by kappa_max
    a[:, mpc.N_KNOTS] = 2 * (v / VMAX) - 1                 # v_start
    a[:, mpc.N_KNOTS + 1] = 2 * (v / VMAX) - 1             # v_end
    return [a, torch.full((B,), v), torch.full((B,), 8.0), torch.zeros(B),
            torch.full((B,), SPEC.delay), torch.zeros(B, 2), torch.zeros(B, SPEC.N, 2)]


# ---------------------------------------------------------------- default parity
def test_legacy_path_is_untouched():
    """Every new seam defaults to None and must reproduce the original call exactly."""
    a = _args(curvature=0.6)
    with torch.no_grad():
        u0, z0, r0 = mpc.solve(*a, SPEC, WB, SMAX, VMAX)
        u1, z1, r1 = mpc.solve(*a, SPEC, WB, SMAX, VMAX, consts=None,
                               v_limit=None, bounds=None, a_walk=None)
    assert torch.equal(u0, u1) and torch.equal(z0, z1) and torch.equal(r0, r1)


def test_path_points_default_return_is_unchanged():
    k = torch.randn(B, mpc.N_KNOTS) * 0.5
    Lp = torch.full((B,), 8.0)
    out = mpc.path_points(k, Lp)
    assert len(out) == 4
    x, y, psi, s, kap = mpc.path_points(k, Lp, return_kappa=True)
    assert kap.shape == (B, 25)
    for a_, b_ in zip(out, (x, y, psi, s)):
        assert torch.equal(a_, b_)


# ---------------------------------------------------------------- the two arms agree at equal mu
def test_fixed_and_oracle_are_the_same_algorithm():
    """The comparison is only about the number, so at equal mu the two must be bit-identical."""
    a = _args(curvature=0.8)
    mu_fixed = torch.full((B,), gc.MU_FIXED_LOW)
    sp_f = gc.GripSpec(mode="fixed").validate()
    sp_o = gc.GripSpec(mode="oracle").validate()
    with torch.no_grad():
        uf, zf, rf = gc.solve_grip(*a, mu_fixed, SPEC, WB, SMAX, VMAX, sp_f)
        uo, zo, ro = gc.solve_grip(*a, mu_fixed.clone(), SPEC, WB, SMAX, VMAX, sp_o)
    assert torch.equal(uf, uo) and torch.equal(zf, zo) and torch.equal(rf, ro)


# ---------------------------------------------------------------- budgets
def test_load_transfer_makes_accel_and_brake_differ():
    sp = gc.GripSpec().validate()
    mu = torch.tensor([[gc.MU_FIXED_LOW]])
    lat, acc, brk = gc.budgets(mu, torch.zeros(1, 1), sp)
    assert float(acc) > float(brk), "accel loads the driven axle; braking unloads it"
    q = gc.MU_FIXED_LOW * sp.mu_r_scale
    assert float(acc) == pytest.approx(q * gc.G * sp.lf / (sp.L - q * sp.h), rel=1e-6)
    assert float(brk) == pytest.approx(q * gc.G * sp.lf / (sp.L + q * sp.h), rel=1e-6)
    assert float(lat) == pytest.approx(gc.MU_FIXED_LOW * sp.mu_f_scale * gc.G, rel=1e-6)


def test_lateral_use_eats_into_the_longitudinal_budget():
    sp = gc.GripSpec().validate()
    mu = torch.full((1, 1), 1.0)
    _, acc0, brk0 = gc.budgets(mu, torch.zeros(1, 1), sp)
    _, acc1, brk1 = gc.budgets(mu, torch.full((1, 1), sp.rho_max), sp)
    assert float(acc1) < float(acc0) and float(brk1) < float(brk0)


def test_higher_mu_never_gives_a_smaller_budget():
    sp = gc.GripSpec().validate()
    lo = gc.budgets(torch.tensor([[0.73423]]), torch.zeros(1, 1), sp)
    hi = gc.budgets(torch.tensor([[1.1540]]), torch.zeros(1, 1), sp)
    for a_, b_ in zip(lo, hi):
        assert float(b_) >= float(a_)


# ---------------------------------------------------------------- envelope
def _envelope(curvature, mu_val, v_plan=8.0):
    k = torch.full((1, mpc.N_KNOTS), float(curvature))
    Lp = torch.full((1,), 10.0)
    return gc.speed_envelope(k, Lp, torch.full((1,), 20.0),
                             torch.full((1,), v_plan), torch.full((1,), v_plan),
                             torch.full((1,), float(mu_val)), gc.GripSpec().validate())


def test_straight_path_is_not_limited_by_curvature():
    v, d = _envelope(0.0, gc.MU_FIXED_LOW, v_plan=8.0)
    assert float(d["v_curve"].min()) > 50.0, "a straight plan has no curvature limit"
    assert float(v.max()) == pytest.approx(8.0, abs=1e-4), "the plan's own target survives"


def test_curved_path_is_limited_at_the_derived_speed():
    kappa, mu_val = 0.8, gc.MU_FIXED_LOW
    v, d = _envelope(kappa, mu_val, v_plan=8.0)
    sp = gc.GripSpec()
    expect = math.sqrt(sp.rho_max * mu_val * sp.mu_f_scale * gc.G / kappa)
    assert float(d["v_curve"][0, 0]) == pytest.approx(expect, rel=1e-5)
    assert float(v.max()) <= expect + 1e-4, "nothing on the path exceeds the curvature limit"
    assert float(v.max()) < 8.0, "the envelope actually binds"


def test_more_grip_allows_more_speed_through_the_same_corner():
    v_lo, _ = _envelope(0.8, 0.73423)
    v_hi, _ = _envelope(0.8, 1.1540)
    assert float(v_hi.max()) > float(v_lo.max())


def test_braking_is_anticipated_before_the_corner():
    """Straight first, then a corner: the speed must already be coming down before the curvature."""
    k = torch.zeros(1, mpc.N_KNOTS)
    k[0, -2:] = 1.2                                   # corner only at the far end of the plan
    Lp = torch.full((1,), 12.0)
    v, d = gc.speed_envelope(k, Lp, torch.full((1,), 20.0),
                             torch.full((1,), 8.0), torch.full((1,), 8.0),
                             torch.full((1,), gc.MU_FIXED_LOW), gc.GripSpec().validate())
    n = v.shape[1]
    straight_part = v[0, : n // 2]
    assert float(straight_part[-1]) < float(straight_part[0]) - 0.05, \
        "the profile is flat through the straight; the corner ahead was not anticipated"
    assert float(d["v_curve"][0, : n // 2].min()) > float(v[0, : n // 2].min()), \
        "the reduction came from the backward pass, not from local curvature"


# ---------------------------------------------------------------- bounds reach the solver
def test_per_env_bounds_reach_the_ilqr_controls():
    """Different mu per env must produce different accel clamps in the returned controls."""
    a = _args(B=2, v=1.0)
    a[0][:, mpc.N_KNOTS:] = 1.0                        # ask for full speed -> accel saturates
    mu = torch.tensor([0.40, 1.20])                    # deliberately far apart
    sp = gc.GripSpec(mode="oracle").validate()
    with torch.no_grad():
        u, _, _ = gc.solve_grip(*a, mu, SPEC, WB, SMAX, VMAX, sp)
        _, acc, brk = gc.budgets(mu.reshape(-1, 1), torch.zeros(2, 1), sp)
    assert float(u[0, :, 1].max()) <= float(acc[0]) + 1e-4
    assert float(u[1, :, 1].max()) <= float(acc[1]) + 1e-4
    assert float(u[1, :, 1].max()) > float(u[0, :, 1].max()) + 1e-3, \
        "the higher-grip env is clamped no higher than the lower one"
    assert float(u[:, :, 1].min()) >= -float(brk.max()) - 1e-4


def test_bounds_are_tighter_than_the_spec_at_low_mu():
    sp = gc.GripSpec().validate()
    _, acc, brk = gc.budgets(torch.tensor([[gc.MU_FIXED_LOW]]), torch.zeros(1, 1), sp)
    assert float(acc) < SPEC.a_max and float(brk) < SPEC.a_brake


# ---------------------------------------------------------------- live mu
def test_mu_is_an_argument_so_a_new_value_changes_the_output():
    a = _args(B=2, curvature=0.7)
    sp = gc.GripSpec(mode="oracle").validate()
    with torch.no_grad():
        u_lo, _, r_lo = gc.solve_grip(*a, torch.full((2,), 0.73423), SPEC, WB, SMAX, VMAX, sp)
        u_hi, _, r_hi = gc.solve_grip(*a, torch.full((2,), 1.1540), SPEC, WB, SMAX, VMAX, sp)
    assert not torch.allclose(r_lo[:, :, 3], r_hi[:, :, 3]), "reference speed ignored mu"
    assert not torch.allclose(u_lo, u_hi), "controls ignored mu"


def test_update_writes_the_buffer_in_place():
    """The graph copies from this tensor, so `update` must not rebind it."""
    class _T:
        spec = SPEC
    g = gc.GripMPC(_T(), gc.GripSpec(mode="oracle"), batch=3, device="cpu",
                   wb=WB, s_max=SMAX, v_max=VMAX)
    ptr = g.mu.data_ptr()
    g.update(torch.tensor([0.8, 0.9, 1.0]))
    assert g.mu.data_ptr() == ptr
    assert torch.allclose(g.mu, torch.tensor([0.8, 0.9, 1.0]))
    with pytest.raises(ValueError):
        g.update(torch.tensor([0.8]))


def test_fixed_arm_refuses_update_and_legacy_refuses_install():
    class _T:
        spec = SPEC
    f = gc.GripMPC(_T(), gc.GripSpec(mode="fixed"), 2, "cpu", WB, SMAX, VMAX)
    with pytest.raises(RuntimeError, match="oracle arm"):
        f.update(torch.tensor([1.0, 1.0]))
    l = gc.GripMPC(_T(), gc.GripSpec(mode="legacy"), 2, "cpu", WB, SMAX, VMAX)
    with pytest.raises(RuntimeError, match="installs nothing"):
        l.install()


def test_install_and_release_restore_the_previous_solver():
    class _T:
        spec = SPEC
    t = _T()
    sentinel = object()
    t._solver = sentinel
    g = gc.GripMPC(t, gc.GripSpec(mode="fixed"), 2, "cpu", WB, SMAX, VMAX)
    g.install()
    assert t._solver is not sentinel
    g.release()
    assert t._solver is sentinel
    g.release()                                        # idempotent
    assert t._solver is sentinel


def test_the_passes_never_exceed_the_bound_the_solver_gets():
    """The backward pass must not plan a deceleration the iLQR clamp would refuse to command."""
    k = torch.zeros(1, mpc.N_KNOTS)
    k[0, -2:] = 1.2
    Lp = torch.full((1,), 12.0)
    v, d = gc.speed_envelope(k, Lp, torch.full((1,), 20.0), torch.full((1,), 8.0),
                             torch.full((1,), 8.0), torch.full((1,), gc.MU_FIXED_LOW),
                             gc.GripSpec().validate())
    n = v.shape[1]
    ds = float(Lp) / (n - 1)
    a_brk = float(d["a_brk_bound"])
    a_acc = float(d["a_acc_bound"])
    dec = (v[0, :-1] ** 2 - v[0, 1:] ** 2) / (2 * ds)          # implied decel between samples
    assert float(dec.max()) <= a_brk + 1e-4, "the profile brakes harder than the solver may"
    assert float((-dec).max()) <= a_acc + 1e-4, "and accelerates harder than the solver may"
    assert a_brk == pytest.approx(float(d["a_brk"].min()), rel=1e-6), "bound is the path minimum"


# ---------------------------------------------------------------- initial state is not assumed
def test_the_envelope_takes_no_measured_speed_at_all():
    """It is an upper envelope over the path. Re-introducing a v_meas seed would make it a claim
    about reachability from the current state, which the contract forbids."""
    import inspect
    assert "v_meas" not in inspect.signature(gc.speed_envelope).parameters


def test_an_overspeed_car_gets_a_braking_target_at_the_budget():
    """Three times over the corner speed: the target stays the envelope, and the decel stays the
    bound -- no instantaneous stop is demanded."""
    sp = gc.GripSpec(mode="fixed").validate()
    mu = torch.full((B,), gc.MU_FIXED_LOW)
    a = _args(curvature=0.8, v=3.0)               # the plan asks for 3 m/s through the corner
    a[1] = torch.full((B,), 9.0)                  # the car is doing 9
    with torch.no_grad():
        u, _, r = gc.solve_grip(*a, mu, SPEC, WB, SMAX, VMAX, sp)
        k, Lp, v0, v1 = mpc.decode(a[0], a[1], VMAX, a[2], SPEC)
        env, d = gc.speed_envelope(k, Lp, a[2], v0, v1, mu, sp)
    brk = float(d["a_brk_bound"][0])
    assert float(r[:, :, 3].max()) <= float(env.max()) + 1e-4, "the target rose with the car's speed"
    assert float(r[0, 0, 3]) < 9.0, "no braking was demanded at all"
    assert float(u[:, :, 1].min()) == pytest.approx(-brk, abs=1e-3), \
        "the solver either did not use the brake budget or exceeded it"


def test_from_standstill_and_from_overspeed_the_outputs_stay_finite_and_bounded():
    sp = gc.GripSpec(mode="oracle").validate()
    mu = torch.full((B,), 0.5)                                  # low grip: a tight brake bound
    _, acc, brk = gc.budgets(mu.reshape(-1, 1), torch.zeros(B, 1), sp)
    for v in (0.0, VMAX + 4.0):                                 # standstill, then well past v_max
        a = _args(curvature=0.5, v=v)
        a[1] = torch.full((B,), float(v))
        with torch.no_grad():
            u, z, r = gc.solve_grip(*a, mu, SPEC, WB, SMAX, VMAX, sp)
        assert torch.isfinite(u).all() and torch.isfinite(z).all() and torch.isfinite(r).all()
        assert float(u[:, :, 1].min()) >= -float(brk.max()) - 1e-4, \
            f"v_meas={v}: commanded a decel below the brake budget"
        assert float(u[:, :, 1].max()) <= float(acc.max()) + 1e-4


def test_legacy_still_demands_instantaneous_decel_when_overspeed():
    """The re-applied floor is scoped to the opt-in path; legacy keeps its pre-existing behaviour."""
    a = _args(B=1, v=VMAX + 4.0)
    a[1] = torch.full((1,), VMAX + 4.0)
    with torch.no_grad():
        u, _, _ = mpc.solve(*a, SPEC, WB, SMAX, VMAX)
    assert float(u[0, 0, 1]) < -SPEC.a_brake, "legacy's v_max clamp no longer undercuts a_brake"


# ---------------------------------------------------------------- capture prerequisites
def test_the_installed_solver_binds_prebuilt_ilqr_consts():
    """Inline `torch.tensor(spec.q)` is a pageable H2D copy and fails capture; the bound tuple is
    what makes this capturable, and it must not change the numbers."""
    class _T:
        spec = SPEC
    t = _T()
    g = gc.GripMPC(t, gc.GripSpec(mode="fixed"), B, "cpu", WB, SMAX, VMAX)
    assert g._consts is not None and len(g._consts) == 8
    a = _args(curvature=0.6)
    g.install()
    with torch.no_grad():
        u_bound, _, _ = t._solver(*a, SPEC, WB, SMAX, VMAX)
        u_inline, _, _ = gc.solve_grip(*a, g.mu, SPEC, WB, SMAX, VMAX, g.gspec, consts=None)
    g.release()
    assert torch.equal(u_bound, u_inline), "binding the constants changed the result"


def test_the_lead_fractions_are_reported_not_applied():
    class _T:
        spec = SPEC
    g = gc.GripMPC(_T(), gc.GripSpec(mode="fixed"), B, "cpu", WB, SMAX, VMAX)
    d = g.diagnostics(*_args(curvature=0.4)[:3])
    assert gc.TRANSFER_APPLIED is False
    assert d["lead_fraction_first_step"] == gc.LEAD_FRACTION_FIRST_STEP
    assert d["lead_fraction_steady"] == gc.LEAD_FRACTION_STEADY
    # The budgets in the diagnostics are the tyre numbers, unscaled by either fraction.
    _, acc, _ = gc.budgets(g.mu.reshape(-1, 1), d["rho"], gc.GripSpec().validate())
    assert float(d["a_acc_bound"][0]) == pytest.approx(float(acc.min(1).values[0]), rel=1e-6)
