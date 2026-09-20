"""Independent contracts for the teacher-only plan speed projection.

The projection is a safety envelope for the plan labels.  These tests deliberately re-compute
the envelope on a much denser grid than the implementation uses; calling the implementation's
predicate would only prove that it agrees with itself.
"""
from __future__ import annotations

import numpy as np
import torch

from f1sim.interactive_teacher import InteractiveTeacher
from f1sim.mpc import PlanSpec, decode, encode
from f1sim.params import VehicleParams
from f1sim.raceline import Raceline
from f1sim.teacher import LABEL_GRIP_CODE, RacelineTeacher
from f1sim.teacher_feasibility import project_speeds


def _dense_margins(k, length, v0, v1, vehicle, spec, mu_front, mu_rear,
                   parameters=None, *, samples=8193):
    """Return independently recomputed constraint margins for one returned profile.

    Positive values are violations.  The equations mirror the physical contract (axle friction
    circles, longitudinal load transfer, actuator limits, steering angle and steering slew), but
    do not call ``project_speeds`` or any helper from that module.
    """
    dtype = torch.float64
    parameters = {} if parameters is None else parameters

    def parameter(name):
        value = parameters.get(name, getattr(vehicle, name))
        return float(torch.as_tensor(value))

    lf, lr = parameter("lf"), parameter("lr")
    height = parameter("h")
    c_roll, c_drag = parameter("c_roll"), parameter("c_drag")
    a_max, a_brake = parameter("a_max"), parameter("a_brake")
    v_switch = parameter("v_switch")
    steer_max, steer_rate = parameter("s_max"), parameter("sv_max")
    drive_split = parameter("drive_split_r") if vehicle.wheel_model else 1.0
    k = torch.as_tensor(k, dtype=dtype)
    length = float(torch.as_tensor(length))
    v0, v1 = (float(torch.as_tensor(v0)), float(torch.as_tensor(v1)))
    mu_front, mu_rear = float(torch.as_tensor(mu_front)), float(torch.as_tensor(mu_rear))
    u = torch.linspace(0.0, 1.0, samples, dtype=dtype)
    pos = u * (k.numel() - 1)
    i0 = pos.floor().long().clamp(max=k.numel() - 2)
    frac = pos - i0.to(dtype)
    curve = k[i0] * (1.0 - frac) + k[i0 + 1] * frac
    dcurve_ds = (k[i0 + 1] - k[i0]) * (k.numel() - 1) / max(length, 1e-12)

    v = v0 + (v1 - v0) * u
    dv_ds = (v1 - v0) / max(length, 1e-12)
    a = v * dv_ds
    resistance = c_roll * torch.tanh(v / 0.05) + c_drag * v.square()
    longitudinal = a + resistance
    lateral = v.square() * curve.abs()

    wb = lf + lr
    front_load_fraction = lr / wb
    rear_load_fraction = lf / wb
    front_normal = 9.81 * front_load_fraction - height / wb * a
    rear_normal = 9.81 * rear_load_fraction + height / wb * a
    front_long = (1.0 - drive_split) * longitudinal
    rear_long = drive_split * longitudinal
    front_force = torch.sqrt(front_long.square() + (front_load_fraction * lateral).square())
    rear_force = torch.sqrt(rear_long.square() + (rear_load_fraction * lateral).square())

    acceleration_limit = min(a_max, spec.a_max) * torch.minimum(
        torch.ones_like(v), torch.as_tensor(v_switch, dtype=dtype) / v.clamp_min(1e-12)
    )
    effective_wheelbase = wb + spec.k_us * v.square()
    steering = torch.atan(effective_wheelbase * curve.abs())
    steering_slew = (
        v
        * (
            effective_wheelbase * dcurve_ds
            + 2.0 * spec.k_us * v * dv_ds * curve
        ).abs()
        / (1.0 + (effective_wheelbase * curve).square())
    )

    return {
        "front_friction": front_force / (mu_front * front_normal) - 1.0,
        "rear_friction": rear_force / (mu_rear * rear_normal) - 1.0,
        "front_normal": -front_normal,
        "rear_normal": -rear_normal,
        "motor": longitudinal - acceleration_limit,
        "brake": -longitudinal - min(a_brake, spec.a_brake),
        "net_acceleration": a - spec.a_max,
        "net_braking": -a - spec.a_brake,
        "steering": steering - steer_max,
        "slew": steering_slew - steer_rate,
    }


def _projection_inputs(batch=1):
    vehicle = VehicleParams()
    spec = PlanSpec()
    k = torch.tensor(
        [
            [0.0, 0.2, 0.65, 0.65, 0.2, 0.0],
            [0.0, -0.4, -0.8, -0.8, -0.4, 0.0],
            [0.0, 0.6, 0.9, 0.9, 0.6, 0.0],
            [0.0, 0.3, 0.85, 0.85, 0.3, 0.0],
        ],
        dtype=torch.float32,
    )[:batch]
    length = torch.tensor([9.0, 9.0, 7.0, 12.0], dtype=torch.float32)[:batch]
    v0 = torch.tensor([4.0, 5.0, 6.0, 1.0], dtype=torch.float32)[:batch]
    v1 = torch.tensor([1.0, 2.0, 3.0, 7.0], dtype=torch.float32)[:batch]
    parameters = {
        name: torch.full((batch,), value, dtype=torch.float32)
        for name, value in {
            "lf": 0.15,
            "lr": 0.18,
            "h": 0.10,
            "c_roll": 0.2,
            "c_drag": 0.015,
            "a_max": 2.5,
            "a_brake": 3.2,
            "v_switch": 5.0,
            "s_max": 0.40,
            "sv_max": 2.5,
            "drive_split_r": 0.65,
        }.items()
    }
    mu_front = torch.tensor([1.0, 0.95, 1.15, 0.85], dtype=torch.float32)[:batch]
    mu_rear = torch.tensor([0.9, 0.8, 1.1, 0.75], dtype=torch.float32)[:batch]
    return k, length, v0, v1, parameters, vehicle, spec, mu_front, mu_rear


def test_feasible_constant_straight_is_returned_bitwise_unchanged():
    vehicle = VehicleParams()
    spec = PlanSpec()
    k = torch.zeros(2, 6)
    length = torch.tensor([6.0, 6.0])
    v0 = torch.tensor([3.0, 0.0])
    v1 = torch.tensor([3.0, 0.0])
    out0, out1, scale = project_speeds(
        k, length, v0, v1, None, vehicle, spec, torch.ones(2), torch.ones(2)
    )

    assert torch.equal(out0, v0)
    assert torch.equal(out1, v1)
    assert torch.equal(scale, torch.ones_like(scale))


def test_zero_targets_are_retained_when_a_curved_profile_is_projected():
    vehicle = VehicleParams()
    spec = PlanSpec()
    k = torch.tensor([[0.0, 0.0, 0.9, 0.9, 0.0, 0.0], [0.0, 0.0, 0.9, 0.9, 0.0, 0.0]])
    length = torch.tensor([8.0, 8.0])
    v0 = torch.tensor([0.0, 4.0])
    v1 = torch.tensor([0.0, 4.0])
    out0, out1, scale = project_speeds(
        k, length, v0, v1, None, vehicle, spec, torch.ones(2), torch.ones(2)
    )

    assert out0[0].item() == 0.0
    assert out1[0].item() == 0.0
    assert 0.0 < scale[1].item() < 1.0
    assert out0[1].item() < v0[1].item()
    assert out1[1].item() < v1[1].item()


def test_interior_curvature_bump_lowers_equal_endpoint_targets():
    vehicle = VehicleParams()
    spec = PlanSpec()
    straight = torch.zeros(1, 6)
    bump = torch.tensor([[0.0, 0.0, 0.7, 0.7, 0.0, 0.0]])
    length = torch.tensor([10.0])
    target = torch.tensor([6.0])
    straight_out = project_speeds(
        straight, length, target, target, None, vehicle, spec, torch.ones(1), torch.ones(1)
    )
    bump_out = project_speeds(
        bump, length, target, target, None, vehicle, spec, torch.ones(1), torch.ones(1)
    )

    assert torch.equal(straight_out[0], target)
    assert torch.equal(straight_out[1], target)
    assert bump_out[0].item() < target.item()
    assert bump_out[1].item() < target.item()
    assert 0.0 < bump_out[2].item() < 1.0


def test_returned_profiles_pass_independent_dense_axle_and_actuator_checks():
    k, length, v0, v1, parameters, vehicle, spec, mu_front, mu_rear = _projection_inputs(batch=4)
    out0, out1, scale = project_speeds(
        k, length, v0, v1, parameters, vehicle, spec, mu_front, mu_rear
    )

    assert torch.isfinite(out0).all() and torch.isfinite(out1).all()
    assert ((scale >= 0.0) & (scale <= 1.0)).all()
    for row in range(k.shape[0]):
        # The returned profile has one common endpoint scale, so checking the actual returned
        # endpoints is sufficient to test the profile the teacher will label.
        margins = _dense_margins(
            k[row], length[row], out0[row], out1[row], vehicle, spec,
            mu_front[row], mu_rear[row],
            {key: value[row] for key, value in parameters.items()}
        )
        for name, margin in margins.items():
            assert float(margin.max()) <= 2e-4, f"{name} violated at row {row}: {margin.max()}"


def test_lower_grip_and_a_weak_rear_axle_both_constrain_the_profile():
    vehicle = VehicleParams()
    spec = PlanSpec()
    k = torch.tensor([[0.0, 0.3, 0.8, 0.8, 0.3, 0.0]])
    length = torch.tensor([8.0])
    v0 = torch.tensor([6.0])
    v1 = torch.tensor([3.0])

    high = project_speeds(k, length, v0, v1, None, vehicle, spec, torch.tensor([1.2]), torch.tensor([1.2]))
    low = project_speeds(k, length, v0, v1, None, vehicle, spec, torch.tensor([0.65]), torch.tensor([0.65]))
    weak_rear = project_speeds(k, length, v0, v1, None, vehicle, spec, torch.tensor([1.2]), torch.tensor([0.55]))

    assert (low[0] < high[0]).item() and (low[1] < high[1]).item()
    assert (weak_rear[0] < high[0]).item() and (weak_rear[1] < high[1]).item()


def _circle_teacher():
    angle = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    line = Raceline.from_xy(
        np.stack([4.0 * np.cos(angle), 4.0 * np.sin(angle)], axis=1),
        np.full(len(angle), 6.0),
    )
    return RacelineTeacher(line)


def test_mixed_true_nominal_and_conservative_labels_use_their_declared_grip_semantics():
    teacher = _circle_teacher()
    teacher.label_grip_codes = torch.tensor([
        LABEL_GRIP_CODE["true"], LABEL_GRIP_CODE["nominal"], LABEL_GRIP_CODE["conservative"]
    ])
    spec = PlanSpec()
    state = torch.zeros(3, 8)
    state[:, 3] = 6.0
    action = encode(
        torch.tensor([[0.0, 0.3, 0.8, 0.8, 0.3, 0.0]]).expand(3, -1),
        torch.full((3,), 7.0), torch.full((3,), 7.0), 8.0, spec
    )
    # The nominal and conservative rows deliberately receive the same low true mu.  Nominal
    # must ignore it, while true must use it; otherwise those two modes silently collapse.
    parameters = {
        "mu": torch.tensor([1.3, 0.7, 0.7]),
        "mu_f_scale": torch.full((3,), 0.92),
        "mu_r_scale": torch.ones(3),
    }
    projected = teacher.project_plan_action(action, state, parameters, 8.0, spec)
    _, _, out0, out1 = decode(projected, state[:, 3], 8.0, torch.full((3,), 8.0), spec)

    assert torch.all(out0 > 0) and torch.all(out1 > 0)
    assert out0[0] > out0[1] > out0[2]
    assert out1[0] > out1[1] > out1[2]
    assert torch.allclose(out0, out1)


def test_interactive_offset_candidates_are_recertified_after_online_endpoint_replacement():
    base = _circle_teacher()
    interactive = InteractiveTeacher(
        base, offsets=(-0.6, 0.0, 0.6), speeds=(1.0, 0.8, 0.55), cand_iters=1
    )
    state = torch.zeros(1, 8)
    state[:, 3] = 6.0
    spec = PlanSpec()
    tid = torch.zeros(1, dtype=torch.long)
    idx = base.project(state[:, :2], tid)[0]
    candidates = interactive._candidates(
        state, None, tid, 8.0, spec, None, idx, plan_speed=torch.tensor([3.0])
    )

    # `_candidates` first replaces every offset's endpoints with the on-line endpoints and then
    # applies the speed multipliers.  Decode the final tensor and certify those final profiles;
    # checking the pre-replacement plans would miss the bug this contract is intended to catch.
    assert candidates.shape == (interactive.n_candidates, 1, 8)
    flat = candidates[:, 0]
    k, length, out0, out1 = decode(
        flat, torch.full((flat.shape[0],), 3.0), 8.0,
        torch.full((flat.shape[0],), 8.0), spec
    )
    muf = torch.full((flat.shape[0],), base.mu_nom * base.mu_f_nom)
    mur = torch.full((flat.shape[0],), base.mu_nom)
    for row in range(flat.shape[0]):
        margins = _dense_margins(k[row], length[row], out0[row], out1[row],
                                 base.vehicle, spec, muf[row], mur[row])
        for name, margin in margins.items():
            assert float(margin.max()) <= 2e-4, f"{name} violated for candidate {row}"


def test_later_bend_does_not_needlessly_slow_feasible_near_endpoint():
    vehicle, spec = VehicleParams(), PlanSpec()
    k = torch.tensor([[0., 0., 0., .6, .6, .6]])
    length, v0, v1 = torch.tensor([9.]), torch.tensor([3.]), torch.tensor([8.])
    mf = torch.tensor([vehicle.mu * vehicle.mu_f_scale])
    mr = torch.tensor([vehicle.mu * vehicle.mu_r_scale])
    out0, out1, scale = project_speeds(k, length, v0, v1, None, vehicle, spec, mf, mr)
    assert torch.equal(out0, v0)
    assert 0 < out1.item() < v1.item() and scale.item() < 1
    for margin in _dense_margins(k[0], length[0], out0[0], out1[0], vehicle, spec, mf[0], mr[0]).values():
        assert float(margin.max()) <= 2e-4
