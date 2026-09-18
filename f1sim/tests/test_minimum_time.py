"""Analytic targets and independent failure tests for the joint minimum-time NLP."""
import numpy as np
import pytest
import torch

from f1sim.minimum_time import _Problem, _audit_export, solve_minimum_time
from f1sim.raceline import Raceline
from f1sim.track import Track


def annulus():
    theta = np.arange(120) * 2 * np.pi / 120
    xy = 6 * np.column_stack([np.cos(theta), np.sin(theta)])
    xx, yy = np.meshgrid(np.arange(-9, 9, .05), np.arange(-9, 9, .05))
    rr = np.hypot(xx, yy)
    return Track.from_occupancy((rr <= 4) | (rr >= 8), .05, (-9., -9.), xy, "analytic-circle")


def test_time_optimum_uses_full_corridor_in_speed_capped_annulus():
    track = annulus()
    result = solve_minimum_time(track.centerline, track, .31, .25, dict(v_max=2., mu=1.0489))
    radius = np.linalg.norm(result.xy, axis=1)
    # At constant capped speed the optimum is the shortest feasible inner circle.
    # The former +/-1m refinement could not even reach this neighbourhood.
    assert radius.mean() < 4.6
    assert radius.min() > 4.35
    assert np.max(result.v) <= 2 + 1e-9
    time = Raceline.from_xy(result.xy, result.v).lap_time
    assert time == pytest.approx(2 * np.pi * radius.mean() / 2, rel=.005)
    assert result.diagnostics["status"] == "converged"
    assert result.diagnostics["kkt_stationarity"] < .005
    assert result.diagnostics["mesh_checks"][-1]["dense_normalized_violation"] <= 2e-5
    assert result.diagnostics["export_audit"]["normalized_physical_violation"] <= 2e-5


def test_joint_constraint_jacobian_matches_independent_directional_difference():
    track = annulus()
    p = _Problem(track.centerline, track, dict(v_max=5.), .31, .1, None, 16, 4)
    x = p.initial.copy()
    x[16:] *= .7
    # Stay inside bilinear cells: the EDT is intentionally nondifferentiable on their edges.
    x[:16] = np.random.default_rng(901).uniform(-.013, .013, 16)
    direction = np.random.default_rng(902).normal(size=len(x))
    step = 1e-6
    numeric = (p.evaluate(x + step * direction) - p.evaluate(x - step * direction)) / (2 * step)
    analytic = p.evaluate(x, True) @ direction
    np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=2e-5)


def test_every_energy_knot_has_both_sides_even_with_float_roundoff():
    track = annulus()
    p = _Problem(track.centerline, track, {}, .31, .1, None, 84, 4)
    right, left = [x.numpy() for x in p.energy_derivatives]
    for knot in range(84):
        i = knot * 4
        assert right[i, knot] == -84
        assert right[i, (knot + 1) % 84] == 84
        assert left[i, (knot - 1) % 84] == -84
        assert left[i, knot] == 84
    assert np.max(np.abs(p.basis[3].numpy()[::4] - p.third_left.numpy()[::4])) > 1


def test_outside_grid_never_extrapolates_free_space():
    track = annulus()
    p = _Problem(track.centerline, track, {}, .31, .1, None, 16, 4)
    q = torch.tensor([[100., 0.], [-100., 0.], [0., 100.], [0., -100.]], dtype=torch.float64)
    np.testing.assert_array_equal(p.field(q).numpy(), np.zeros(4))


@pytest.mark.parametrize("field", ["a_acc", "a_brake", "v_max", "mu", "a_lat"])
def test_explicit_zero_limit_is_rejected(field):
    track = annulus()
    with pytest.raises(ValueError, match="positive"):
        _Problem(track.centerline, track, {field: 0.}, .31, .1, None, 16, 4)


def test_budget_exhaustion_does_not_claim_an_optimized_seed():
    track = annulus()
    with pytest.raises(ValueError, match="converged"):
        solve_minimum_time(track.centerline, track, .31, .25, {}, maxiter=0)
    with pytest.raises(TimeoutError, match="partial geometry discarded"):
        solve_minimum_time(track.centerline, track, .31, .25, {}, max_seconds=0)


def test_export_audit_rejects_geometry_and_speed_changes():
    track = annulus()
    p = _Problem(track.centerline, track, dict(v_max=2.), .31, .1, None, 16, 4)
    with pytest.raises(ValueError, match="audit"):
        _audit_export(track.centerline, np.full(120, 12.), p, np.arange(120)/120)
    blocked = track.centerline * .66
    with pytest.raises(ValueError, match="audit"):
        _audit_export(blocked, np.full(120, 1.), p, np.arange(120)/120)


def test_solver_metadata_round_trips_without_losing_original_fields(tmp_path):
    track = annulus()
    line = Raceline.from_xy(track.centerline, np.ones(len(track.centerline)))
    line.optimization = {"status": "converged", "effective_margin_m": [.04, .4], "global_optimum_proven": False}
    path = tmp_path / "line.csv"
    line.save(path)
    restored = Raceline.load(path)
    assert restored.optimization == line.optimization
    np.testing.assert_array_equal(restored.xy, line.xy)


def test_curvature_is_exact_on_a_circle_with_uneven_sampling():
    from f1sim.raceline import curvature
    theta = np.sort(np.random.default_rng(12).uniform(0, 2*np.pi, 70))
    xy = 3.7 * np.c_[np.cos(theta), np.sin(theta)]
    np.testing.assert_allclose(curvature(xy), 1/3.7, atol=1e-10)


def test_periodic_seam_has_true_left_third_derivative():
    from scipy.interpolate import CubicSpline
    track = annulus()
    p = _Problem(track.centerline, track, {}, .31, .1, None, 84, 4)
    base = np.eye(84)
    spline = CubicSpline(np.arange(85)/84, np.vstack([base, base[0]]), bc_type="periodic")
    np.testing.assert_array_equal(p.third_left[0].numpy(), spline(np.nextafter(1., 0.), nu=3))
    assert not np.array_equal(p.third_left[0].numpy(), p.basis[3][0].numpy())
