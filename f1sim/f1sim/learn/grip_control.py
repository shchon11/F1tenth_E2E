"""Friction-informed plan/MPC controller. Opt-in; the legacy path is untouched.

``profile_version=local-v2`` is the automatic runtime: both-axle local speed
planning, an explicitly seeded reachable trajectory, actual-state stage projection,
and nominal VESC command inversion. Historical arms retain the formulas below.

Three arms share this code and differ only in the friction handed to it:

    legacy      nothing installed -- `mpc.solve` as it was
    fixed       mu = 0.73423 everywhere (1.0489 * 0.70, the bottom of the randomization)
    oracle      mu = the episode's true `Simulator.P["mu"]`, per env

The actor is the original frozen checkpoint in all three. Nothing here sees the map, the raceline,
or any future surface: the inputs are the 8D plan the actor emitted and the car's measured speed.

Formulas and their justification are in
`work/learning-next/grip-control/core-status.md`; the short version:

    a_lat_max = mu * mu_f_scale * g                       front saturates first, load share cancels
    rho       = clamp(v^2 |kappa| / a_lat_max, 0, RHO_MAX)  planned lateral utilization, per sample
    q         = mu * mu_r_scale * sqrt(1 - rho^2)           residual coefficient for drive/brake
    a_acc     = min(a_max,   q g lf / (s L - q h))          accel loads the rear
    a_brk     = min(a_brake, q g lf / (s L + q h))          braking unloads it

where s is the rear torque share, one for historical arms and the nominal vehicle's share for auto.

then a curvature cap, a backward braking pass and a forward acceleration pass over the plan's own
dense samples. No second MPC: `mpc.path_points`, `mpc.reference` and `mpc.ilqr` do the work.

**This is a planning approximation, not a safety guarantee**, and not a claim of achieved
feasibility. The profile is an *upper envelope over the path*: it is not seeded from the measured
speed and says nothing about whether the current state can reach it -- see §2b and §3 of the status
note.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch

from .. import mpc as _mpc

G = 9.81

#: The fixed arm's friction: `vehicle.mu` nominal 1.0489 times the bottom of its randomization, 0.70.
MU_FIXED_LOW = 0.73423

#: Planned lateral utilization. Reaching the tyre's peak needs the optimal slip angle, which a
#: tracking controller never targets. One number, used for both the curvature cap and the residual,
#: so the allocation is consistent. Frozen before evaluation.
RHO_MAX = 0.85

#: Lead term only, and only for a step whose `u_prev` is zero: `v_cmd_lead / motor_tau = 0.15/0.20`.
#: It is **not** the fraction of a planned acceleration the whole loop delivers, and must not be read
#: as one.
LEAD_FRACTION_FIRST_STEP = 0.75

#: The same ratio once `u_prev ~= u`, which is the steady case: `z0`'s speed already carries
#: `u_prev * delay`, so the lead is `(v_cmd_lead + delay) / motor_tau = (0.15 + 0.035)/0.20`.
#: Backend's `tracker_chain_probe` measured it directly -- a steady `u = -5` produces a motor request
#: of -4.625 -- and the achieved longitudinal acceleration at low/mid/high grip was
#: -3.300 / -3.977 / -4.429 m/s^2, the mild case slightly *exceeding* |u| because drag also brakes.
#: The remaining shortfall at low mu is tyre saturation, which is the thing the experiment measures.
LEAD_FRACTION_STEADY = 0.925

#: **Neither number is applied anywhere.** They are reported so the command-path mismatch is a
#: measured quantity in the evaluation instead of an assumption inside the controller. Scaling the
#: budgets by one of them would not make the prediction exact -- the response still lags -- it would
#: only make the controller slower while looking principled.
TRANSFER_APPLIED = False

#: `estimated` is the same controller as `oracle`, fed the frozen estimator's filtered lower
#: quantile instead of the truth. It differs from `oracle` only in where `mu` comes from, which is
#: the entire point of the comparison -- so it shares the code path exactly.
MODES = ("legacy", "fixed", "oracle", "estimated")

#: Modes whose friction changes during a run, so `update()` is meaningful.
DYNAMIC_MODES = ("oracle", "estimated")


@dataclass(frozen=True)
class GripSpec:
    """Everything the controller needs besides `mu`, all nominal. Recorded with any result."""
    mode: str = "legacy"
    mu_fixed: float = MU_FIXED_LOW
    rho_max: float = RHO_MAX
    lf: float = 0.15875
    lr: float = 0.17145
    h: float = 0.074
    mu_f_scale: float = 0.92
    mu_r_scale: float = 1.0
    a_max: float = 7.0
    a_brake: float = 5.0
    n_path: int = 25
    # Historical arms assumed rear drive. Only the automatic runtime supplies the nominal
    # vehicle's torque split; old metadata without this field keeps exactly the old budgets.
    drive_split_r: float = 1.0
    profile_version: str = "historical-global-v1"
    c_roll: float = 0.1
    c_drag: float = 0.01
    v_switch: float = 7.319
    sv_max: float = 3.2
    motor_tau: float = 0.20
    actuator_v_max: float = 12.0

    @property
    def L(self) -> float:
        return self.lf + self.lr

    def to_meta(self) -> dict:
        d = asdict(d_self := self)
        d["L"] = d_self.L
        d["lead_fraction_first_step"] = LEAD_FRACTION_FIRST_STEP
        d["lead_fraction_steady"] = LEAD_FRACTION_STEADY
        d["transfer_applied"] = TRANSFER_APPLIED
        d["command_conversion"] = "nominal-vesc-v1" if self.profile_version == "local-v2" else "historical-lead-v1"
        return d

    def validate(self) -> "GripSpec":
        for name, value in asdict(self).items():
            if name not in ("mode", "profile_version") and not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        for name in ("mu_fixed", "lf", "lr", "mu_f_scale", "mu_r_scale", "a_max", "a_brake"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.h < 0 or not isinstance(self.n_path, int):
            raise ValueError("height must be nonnegative and n_path must be an integer")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.profile_version not in ("historical-global-v1", "local-v2"):
            raise ValueError(f"unknown profile_version {self.profile_version!r}")
        if self.n_path < 3 or self.motor_tau <= 0 or self.sv_max <= 0 or self.v_switch <= 0 or self.actuator_v_max <= 0:
            raise ValueError("invalid path, actuator time constant, power or steering limit")
        if self.c_roll < 0 or self.c_drag < 0:
            raise ValueError("resistance coefficients must be nonnegative")
        if not 0.0 < self.rho_max <= 1.0:
            raise ValueError(f"rho_max must be in (0, 1], got {self.rho_max}")
        if not 0.0 < self.drive_split_r <= 1.0:
            raise ValueError(f"drive_split_r must be in (0, 1], got {self.drive_split_r}")
        # `a_acc = q g lf / (L - q h)` diverges at q = L/h; q never gets near it here, but a spec
        # that made it possible would produce a silently enormous budget.
        if self.h > 0 and self.L / self.h <= 2.0:
            raise ValueError(f"L/h = {self.L / self.h:.2f} is too small for the accel formula")
        return self


def budgets(mu: torch.Tensor, rho: torch.Tensor, spec: GripSpec):
    """(a_lat_max, a_acc, a_brk) for a given friction and planned lateral utilization.

    `mu` broadcasts against `rho`, so the same code serves a per-env scalar and a per-sample field.
    """
    a_lat_max = mu * spec.mu_f_scale * G
    q = mu * spec.mu_r_scale * torch.sqrt((1.0 - rho.clamp(0.0, 1.0) ** 2).clamp_min(0.0))
    # The rear tyres supply drive_split_r of the total longitudinal force. The wheel model
    # likewise divides the rear force by this share (dynamics.py). Using share=1 on the nominal
    # 4WD plant incorrectly asks the rear axle to supply every newton of drive and braking.
    driven_length = spec.drive_split_r * spec.L
    a_acc = (q * G * spec.lf / (driven_length - q * spec.h).clamp_min(1e-3)).clamp(max=spec.a_max)
    a_brk = (q * G * spec.lf / (driven_length + q * spec.h)).clamp(max=spec.a_brake)
    return a_lat_max, a_acc, a_brk


def speed_envelope(k: torch.Tensor, Lp: torch.Tensor, speed_cap: torch.Tensor,
                   v0: torch.Tensor, v1: torch.Tensor, mu: torch.Tensor, spec: GripSpec):
    """The friction-limited speed at each dense sample of the plan: (B, n), plus diagnostics.

    Curvature cap, then a backward pass (braking anticipation: the speed allowed here is what can
    still be shed before the corner ahead) and a forward pass (no more speed than can be gained
    along the path). Both loops run over a Python constant, so they unroll and stay graph-safe.

    **An upper envelope over the path, not a trajectory from the current state.** The measured speed
    is deliberately not an input: it is not used to seed sample 0, and nothing here claims the car
    can reach this profile from where it is. A car already faster than the envelope simply gets a
    target below its current speed, which `mpc.reference`'s rate-limited walk turns into a braking
    demand at the budget.
    """
    n = spec.n_path
    _, _, _, s, kap = _mpc.path_points(k, Lp, n=n, return_kappa=True)
    ds = (Lp / (n - 1)).clamp_min(1e-3)[:, None]                            # (B,1)
    mu_c = mu.reshape(-1, 1)                                                # (B,1)

    frac = torch.linspace(0.0, 1.0, n, device=k.device, dtype=k.dtype)[None]
    v_plan = v0[:, None] + (v1 - v0)[:, None] * frac                        # the plan's own profile
    a_lat_max = mu_c * spec.mu_f_scale * G
    v_curve = torch.sqrt((spec.rho_max * a_lat_max) / kap.abs().clamp_min(1e-3))
    cap = torch.minimum(torch.minimum(v_plan, v_curve), speed_cap[:, None]).clamp_min(0.0)

    # Residual longitudinal budget at the lateral use this cap actually implies. One shot, from the
    # capped speed, before the passes -- explicit rather than iterated.
    rho = (cap ** 2 * kap.abs() / a_lat_max.clamp_min(1e-6)).clamp(0.0, spec.rho_max)
    _, a_acc, a_brk = budgets(mu_c, rho, spec)                              # (B,n) each

    # One bound per env, the tightest along this plan, and the passes use exactly what the reference
    # walk and the iLQR clamp will use. Per-sample budgets in the passes would let the backward pass
    # plan a deceleration at a point where the solver is not allowed to command it -- inconsistent
    # even as an approximation. The minimum is the conservative reconciliation.
    a_acc_e = a_acc.min(1).values                                           # (B,)
    a_brk_e = a_brk.min(1).values

    # Columns, not slice-and-concat: the passes are sequential over a Python constant, so they
    # unroll cleanly and stay CUDA-graph-safe.
    cols = list(cap.unbind(1))
    for i in range(n - 2, -1, -1):          # backward: brake in time for what is ahead
        cols[i] = torch.minimum(cols[i], torch.sqrt(cols[i + 1] ** 2 + 2.0 * a_brk_e * ds[:, 0]))
    for i in range(1, n):                   # forward: no more speed than can be gained by here
        cols[i] = torch.minimum(cols[i], torch.sqrt(cols[i - 1] ** 2 + 2.0 * a_acc_e * ds[:, 0]))
    v = torch.stack(cols, 1)

    diag = {"kappa": kap, "v_plan": v_plan, "v_curve": v_curve, "v_limit": v,
            "a_acc": a_acc, "a_brk": a_brk, "a_acc_bound": a_acc_e, "a_brk_bound": a_brk_e,
            "rho": rho, "a_lat_max": a_lat_max, "arc": s}
    return v, diag


def _positive_root(a, b, c):
    """First boundary of a quadratic feasible at zero, including linear/concave cases.

    Torch counterpart of raceline._first_positive_root; all branches stay on device.
    """
    disc = b.square() - 4 * a * c
    den = b + disc.clamp_min(0).sqrt()
    root = -2 * c / den.clamp_min(1e-12)
    root = torch.where((disc >= 0) & (den > 1e-12), root, torch.full_like(c, float('inf')))
    return torch.where(c >= 0, torch.zeros_like(c), root)


def resistance(speed, spec: GripSpec):
    return spec.c_roll * torch.tanh(speed.abs() / .05) + spec.c_drag * speed.square()


def _axles(mu, spec):
    return ((1 - spec.drive_split_r, spec.lr / spec.L, -spec.h / spec.L, mu * spec.mu_f_scale),
            (spec.drive_split_r, spec.lf / spec.L, spec.h / spec.L, mu * spec.mu_r_scale))


def local_accel_bounds(speed, kappa, mu, spec: GripSpec, *, no_resistance=False):
    """Net acceleration interval at the supplied *actual* speed/curvature.

    Each axle obeys (share*(ax+drag))^2+(load*v^2*k)^2 <=
    (mu*(g*load+transfer*ax))^2. The connected interval around ax=0 is solved
    analytically, then intersected with positive axle loads and nominal motor limits.
    Empty intervals/unsustainable lateral states are reported, never called feasible.
    """
    r = torch.zeros_like(speed) if no_resistance else resistance(speed, spec)
    power = spec.a_max * (spec.v_switch / speed.abs().clamp_min(1e-3)).clamp(max=1)
    lo, hi = -spec.a_brake - r, power - r
    bad = torch.zeros_like(speed, dtype=torch.bool)
    for share, load, transfer, grip in _axles(mu, spec):
        aa = share ** 2 - (grip * transfer).square()
        bb = 2 * (share ** 2 * r - grip.square() * G * load * transfer)
        cc = (share * r).square() + (load * speed.square() * kappa).square() - (grip * G * load).square()
        bad = bad | (cc > 1e-5)
        lo = torch.maximum(lo, -_positive_root(aa, -bb, cc))
        hi = torch.minimum(hi, _positive_root(aa, bb, cc))
        if transfer < 0:
            hi = hi.clamp(max=-G * load / transfer)
        elif transfer > 0:
            lo = lo.clamp(min=-G * load / transfer)
    bad = bad | (lo > hi)
    return lo, hi, bad


def _steady_curvature(speed, mu, spec):
    """Curvature preserving the requested lateral reserve and drag at zero net ax."""
    r = resistance(speed, spec)
    cap = torch.full_like(speed, float('inf'))
    for share, load, _, grip in _axles(mu, spec):
        residual = ((spec.rho_max * grip * G * load).square() - (share * r).square()).clamp_min(0)
        cap = torch.minimum(cap, residual.sqrt() / (load * speed.square()).clamp_min(1e-6))
    return cap



def _reachable_step(speed, target, edge_k, ds, mu, spec):
    # Cover the faster endpoint and every resistance between endpoints. Using
    # only the initial speed can exceed the motor power limit after accelerating.
    fastest = torch.maximum(speed, torch.minimum(target,
        (speed.square() + 2 * spec.a_max * ds).sqrt()))
    safe_k = torch.minimum(edge_k, _steady_curvature(fastest, mu, spec))
    lower, upper, _ = local_accel_bounds(fastest, safe_k, mu, spec)
    lower0, upper0, _ = local_accel_bounds(fastest, safe_k, mu, spec, no_resistance=True)
    lower, upper = torch.maximum(lower, lower0), torch.minimum(upper, upper0)
    want = (target.square() - speed.square()) / (2 * ds)
    ax = torch.maximum(torch.minimum(want, upper), lower)
    return (speed.square() + 2 * ax * ds).clamp_min(0).sqrt(), safe_k


def _local_path_limits(cap, kap, mu, ds, spec, k_us, v_max):
    curve = torch.full_like(cap, v_max ** 2)
    # Constant-speed tire demand includes drag, just as in raceline.speed_profile.
    for share, load, _, grip in _axles(mu * spec.rho_max, spec):
        aa = (share * spec.c_drag) ** 2 + (load * kap).square()
        bb = torch.full_like(cap, 2 * share ** 2 * spec.c_roll * spec.c_drag)
        cc = (share * spec.c_roll) ** 2 - (grip * G * load).square()
        curve = torch.minimum(curve, _positive_root(aa, bb, cc.expand_as(cap)))
    curve = curve.clamp_min(0).sqrt()
    cap = torch.minimum(cap, curve)
    # Nominal power must sustain the upper envelope. Fixed device-only iterations.
    low, high = torch.zeros_like(cap), cap.clone()
    for _ in range(12):
        middle = (low + high) * .5
        sustainable = resistance(middle, spec) <= spec.a_max * (spec.v_switch / middle.clamp_min(1e-3)).clamp(max=1)
        low, high = torch.where(sustainable, middle, low), torch.where(sustainable, high, middle)
    cap = torch.minimum(cap, high)
    delta = torch.atan((spec.L + k_us * cap.square()) * kap)
    slew_cap = spec.sv_max * ds[:, None] / (delta[:, 1:] - delta[:, :-1]).abs().clamp_min(1e-6)
    cap = torch.minimum(cap, torch.cat([slew_cap[:, :1], torch.minimum(slew_cap[:, :-1], slew_cap[:, 1:]), slew_cap[:, -1:]], 1))
    lo, hi, _ = local_accel_bounds(cap, kap, mu, spec)
    lo_zero, hi_zero, _ = local_accel_bounds(cap, kap, mu, spec, no_resistance=True)
    # Taking both resistance endpoints also covers any speed lowered by the passes.
    acc = torch.minimum(hi, hi_zero).clamp_min(0)
    brk = (-torch.maximum(lo, lo_zero)).clamp_min(0)
    return cap, curve, acc, brk


def local_speed_profile(k, Lp, speed_cap, v0, v1, mu, v_initial, spec: GripSpec,
                        plan_spec: _mpc.PlanSpec, v_max: float, kernels=None):
    """Open-path desired, feasible and reachable profiles; only local edge budgets.

    The feasible envelope uses the same both-axle physics as raceline.speed_profile.
    Edge budgets conservatively cover both endpoints and all lower resistances, so
    reducing a sample in the passes cannot invalidate a previous edge. Reachability
    starts at the measured/delay-predicted speed, even when already outside the
    envelope. During such recovery the feasible steering cap can differ from the
    requested path; the path-tracking infeasibility is explicit in diagnostics.
    """
    _, _, _, arc, kap = _mpc.path_points(k, Lp, n=spec.n_path, return_kappa=True)
    n = spec.n_path
    ds = (Lp / (n - 1)).clamp_min(1e-3)
    mu_p = mu[:, None]
    frac = torch.linspace(0., 1., n, device=k.device, dtype=k.dtype)[None]
    desired = (v0[:, None] + (v1 - v0)[:, None] * frac).clamp_min(0)
    cap = torch.minimum(desired, speed_cap[:, None]).clamp(max=v_max)
    path_limits = _local_path_limits if kernels is None else kernels["path_limits"]
    cap, curve, acc, brk = path_limits(cap, kap, mu_p, ds, spec, plan_spec.k_us, v_max)
    edge_acc = torch.minimum(acc[:, :-1], acc[:, 1:])
    edge_brk = torch.minimum(brk[:, :-1], brk[:, 1:])
    cols = list(cap.unbind(1))
    for i in range(n - 2, -1, -1):
        cols[i] = torch.minimum(cols[i], (cols[i + 1].square() + 2 * edge_brk[:, i] * ds).sqrt())
    for i in range(1, n):
        cols[i] = torch.minimum(cols[i], (cols[i - 1].square() + 2 * edge_acc[:, i - 1] * ds).sqrt())
    feasible = torch.stack(cols, 1)
    # Actual speed is never overwritten by a slower target. Local recovery curvature
    # keeps longitudinal planning well-defined when the desired turn is unreachable.
    reachable_step = _reachable_step if kernels is None else kernels["reachable_step"]
    reachable = [v_initial.abs()]
    recovery_kappa = []
    for i in range(1, n):
        speed = reachable[-1]
        edge_k = torch.maximum(kap[:, i - 1].abs(), kap[:, i].abs())
        next_speed, safe_k = reachable_step(speed, feasible[:, i], edge_k, ds, mu, spec)
        reachable.append(next_speed)
        recovery_kappa.append(safe_k)
    reachable = torch.stack(reachable, 1)
    return reachable, {
        'v_desired': desired, 'v_feasible': feasible, 'v_reachable': reachable,
        'reachable_kappa_bound': torch.stack(recovery_kappa, 1),
        'v_plan': desired, 'v_curve': curve, 'v_limit': feasible,
        'kappa': kap, 'arc': arc, 'a_acc': acc, 'a_brk': brk,
        'initial_overspeed': v_initial > feasible[:, 0] + 1e-5,
        'path_infeasible': reachable > feasible + 1e-4,
        'a_lat_max': mu_p * min(spec.mu_f_scale, spec.mu_r_scale) * G,
        'rho': reachable.square() * kap.abs() / (mu_p * min(spec.mu_f_scale, spec.mu_r_scale) * G).clamp_min(1e-6),
    }


def project_local_control(state, control, mu, spec: GripSpec, plan_spec, wb, s_max, v_max):
    """Project at the rollout state and selected steer, retaining slew during recovery."""
    speed = state[:, 3]
    le = wb + plan_spec.k_us * speed.square()
    steer_cap = torch.atan(le * _steady_curvature(speed, mu, spec)).clamp(max=s_max)
    slew = spec.sv_max * plan_spec.dt
    steer_lo = torch.maximum(-steer_cap, state[:, 4] - slew)
    steer_hi = torch.minimum(steer_cap, state[:, 4] + slew)
    steer = torch.maximum(torch.minimum(control[:, 0], steer_hi), steer_lo)
    # An initially infeasible steering state cannot jump inside the cap in one step.
    # Recover at the servo rate, and expose the remaining violation below.
    recovery = torch.maximum(torch.minimum(torch.zeros_like(steer), state[:, 4] + slew), state[:, 4] - slew).clamp(-s_max, s_max)
    steer = torch.where(steer_lo <= steer_hi, steer, recovery)
    curvature = torch.tan(steer) / le
    lower, upper, bad = local_accel_bounds(speed, curvature, mu, spec)
    # Physical top speed is a state goal, not permission to exceed available brakes.
    upper = torch.minimum(upper, (spec.actuator_v_max - speed) / plan_spec.dt)
    upper = torch.maximum(upper, lower)
    accel = torch.maximum(torch.minimum(control[:, 1], upper), lower)
    bounded = torch.stack([steer, accel], 1)
    return bounded, lower, upper, bad | (steer_lo > steer_hi)


def stage_diagnostics(u, z, mu, spec, plan_spec, wb):
    speed, steer, ax = z[:, :-1, 3], u[:, :, 0], u[:, :, 1]
    kap = torch.tan(steer) / (wb + plan_spec.k_us * speed.square())
    lo, hi, bad = local_accel_bounds(speed, kap, mu[:, None], spec)
    r = resistance(speed, spec)
    violations = []
    for share, load, transfer, grip in _axles(mu[:, None], spec):
        demand = ((share * (ax + r)).square() + (load * speed.square() * kap).square()).sqrt()
        available = grip * (G * load + transfer * ax).clamp_min(0)
        violations.append((demand - available).clamp_min(0))
    return {'stage_lower': lo, 'stage_upper': hi, 'stage_infeasible': bad,
            'stage_axle_violation': torch.stack(violations, -1),
            'stage_accel_violation': torch.maximum(lo - ax, ax - hi).clamp_min(0),
            'stage_slew_violation': ((steer - z[:, :-1, 4]).abs() - spec.sv_max * plan_spec.dt).clamp_min(0)}


def solve_local_grip(action, v_meas, speed_cap, yaw_rate, delay, u_prev, warm, mu,
                     spec, wb, s_max, v_max, gspec, consts=None, out_bounds=None,
                     out_diagnostics=None, kernels=None):
    k, Lp, v0, v1 = _mpc.decode(action, v_meas, v_max, speed_cap, spec)
    v = v_meas.abs()
    le = wb + spec.k_us * v.square()
    steer_now = torch.atan(yaw_rate * le / v.clamp_min(.5)).clamp(-s_max, s_max)
    psi0 = yaw_rate * delay
    z0 = torch.stack([v * delay, .5 * v * psi0 * delay, psi0,
                      (v + u_prev[:, 1] * delay).clamp_min(0),
                      torch.where(v > .5, steer_now, u_prev[:, 0]), u_prev[:, 1]], 1)
    reachable, diag = local_speed_profile(k, Lp, speed_cap, v0, v1, mu, z0[:, 3], gspec, spec, v_max, kernels=kernels)
    walk = torch.stack([diag['a_brk'], diag['a_acc']], -1)
    ref = _mpc.reference(k, Lp, v0, v1, spec, z0[:, 3], a_walk=walk, v_profile=reachable)
    project_control = project_local_control if kernels is None else kernels["project_control"]
    def project(state, control, stage):
        return project_control(state, control, mu, gspec, spec, wb, s_max, v_max)[0]
    u, z = _mpc.ilqr(z0, ref, warm, spec, wb, s_max, v_max, consts=consts, projector=project)
    diagnostics = stage_diagnostics if kernels is None else kernels["stage_diagnostics"]
    diag.update(diagnostics(u, z, mu, gspec, spec, wb))
    if out_bounds is not None:
        out_bounds.copy_(torch.stack([-diag['stage_lower'][:, 0], diag['stage_upper'][:, 0]], 1))
    if out_diagnostics is not None:
        for name, buffer in out_diagnostics.items():
            if name in diag:
                buffer.copy_(diag[name])
    return u, z, ref



def solve_grip(action, v_meas, speed_cap, yaw_rate, delay, u_prev, warm, mu,
               spec: _mpc.PlanSpec, wb: float, s_max: float, v_max: float,
               gspec: GripSpec, consts=None, out_bounds=None, out_diagnostics=None, kernels=None):
    """`mpc.solve` with a friction-limited reference and per-env control bounds.

    `mu` is a **tensor argument**, so under `GraphedCallable` it becomes a static input buffer that
    is copied per call -- a live graph input, not a scalar frozen into the capture. `PlanSpec` is
    never mutated.

    `consts` is `mpc.build_ilqr_consts`' tuple. Pass it whenever this will be captured: with
    `consts=None` the iLQR builds its cost tensors inline from Python tuples, and the first such
    `torch.tensor(spec.q)` is a pageable host-to-device copy that fails capture -- and a failed
    capture poisons the CUDA context rather than falling back.
    """
    if gspec.profile_version == "local-v2":
        return solve_local_grip(action, v_meas, speed_cap, yaw_rate, delay, u_prev, warm,
                                mu, spec, wb, s_max, v_max, gspec, consts, out_bounds, out_diagnostics, kernels)
    k, Lp, v0, v1 = _mpc.decode(action, v_meas, v_max, speed_cap, spec)
    v_limit, diag = speed_envelope(k, Lp, speed_cap, v0, v1, mu, gspec)

    # The same per-env budgets the envelope's passes used -- the tightest point of this plan, at the
    # lateral utilization actually planned rather than at rho = 0. The solver may not command more
    # than the worst point of the path allows.
    a_acc_e, a_brk_e = diag["a_acc_bound"], diag["a_brk_bound"]
    lo = torch.stack([torch.full_like(a_acc_e, -s_max), -a_brk_e], 1)
    hi = torch.stack([torch.full_like(a_acc_e, s_max), a_acc_e], 1)
    bounds = torch.stack([lo, hi], 1)                                       # (B, 2, 2)
    a_walk = torch.stack([a_brk_e, a_acc_e], 1)                             # (B, 2)
    if out_bounds is not None:
        # The limits this solve actually imposed, written into a caller-owned buffer. A closed-over
        # tensor is written on every graph replay too, so the training loop can log the realised
        # bound without recomputing the envelope or synchronising -- and without confusing it with
        # the straight-line budget, which is a different (larger) number on any curved plan.
        out_bounds.copy_(a_walk)

    return _mpc.solve(action, v_meas, speed_cap, yaw_rate, delay, u_prev, warm, spec, wb, s_max,
                      v_max, consts=consts, v_limit=v_limit, bounds=bounds, a_walk=a_walk)


# Compiled code is shared across tracker/map lifetimes; no graph/input tensors live here.
# Shape/spec warmup status is separate because compilation must finish before CUDA capture.
_POINTWISE_KERNELS = None
_POINTWISE_WARMED = set()
_POINTWISE_FAILURES = {}


def _pointwise_kernels():
    global _POINTWISE_KERNELS
    if _POINTWISE_KERNELS is None:
        def compile_one(fn):
            compiled = torch.compile(fn, fullgraph=True, dynamic=False, mode="default")
            @torch.no_grad()
            def call(*args):
                # Normalize views before entering Dynamo: warm-start controls and path
                # columns otherwise create extra stride specializations at capture time.
                inputs = tuple(a.contiguous() if torch.is_tensor(a) else a for a in args)
                return compiled(*inputs)
            return call
        _POINTWISE_KERNELS = {
            "project_control": compile_one(project_local_control),
            "reachable_step": compile_one(_reachable_step),
            "path_limits": compile_one(_local_path_limits),
            "stage_diagnostics": compile_one(stage_diagnostics),
        }
    return _POINTWISE_KERNELS


class GripMPC:
    """Installs `solve_grip` as the tracker's solver, keeps `mu` live, and puts things back.

    Ordering matters: `prepare_graph_runtime` captures `mpc.solve` and installs it as
    `tracker._solver`, so it would overwrite a hook installed earlier. Install this one **after** it.
    """

    def __init__(self, tracker, gspec: GripSpec, batch: int, device, wb: float, s_max: float,
                 v_max: float):
        self.gspec = gspec.validate()
        self.tracker = tracker
        self.B = int(batch)
        self.device = torch.device(device)
        self.wb, self.s_max, self.v_max = float(wb), float(s_max), float(v_max)
        #: The live friction. `update()` writes this buffer in place; the graph copies from it, so a
        #: new value reaches a captured graph without recapture.
        self.mu = torch.full((self.B,), float(gspec.mu_fixed), device=self.device)
        #: Built once, here, and bound into the solver -- exactly what `graph_fastpath.capture_mpc`
        #: does and for the same reason: `ilqr`'s inline `torch.tensor(spec.q)` is a pageable H2D copy
        #: that fails CUDA-graph capture, and a failed capture is fatal to the process rather than
        #: recoverable. `dtype` follows the default dtype so the bound path is numerically the inline
        #: path. Nothing is cached globally; these live and die with this object.
        self._consts = _mpc.build_ilqr_consts(tracker.spec, self.s_max, self.device,
                                              dtype=torch.get_default_dtype())
        #: The limits the last solve actually imposed, `[a_brk, a_acc]` per env. Written by the
        #: solver (inside the graph too, since it is closed over rather than passed), so a caller can
        #: log the realised bound instead of the straight-line budget, which on a curved plan is a
        #: different and larger number.
        self.last_bounds = torch.zeros(self.B, 2, device=self.device)
        self.last_diagnostics = {}
        if gspec.profile_version == "local-v2":
            shapes = {"v_desired": (self.B, gspec.n_path), "v_feasible": (self.B, gspec.n_path),
                      "v_reachable": (self.B, gspec.n_path), "initial_overspeed": (self.B,),
                      "path_infeasible": (self.B, gspec.n_path),
                      "stage_lower": (self.B, tracker.spec.N), "stage_upper": (self.B, tracker.spec.N),
                      "stage_infeasible": (self.B, tracker.spec.N),
                      "stage_axle_violation": (self.B, tracker.spec.N, 2),
                      "stage_accel_violation": (self.B, tracker.spec.N),
                      "stage_slew_violation": (self.B, tracker.spec.N)}
            flags = {"initial_overspeed", "path_infeasible", "stage_infeasible"}
            self.last_diagnostics = {name: torch.zeros(shape, device=self.device, dtype=torch.bool if name in flags else self.mu.dtype)
                                     for name, shape in shapes.items()}
        if gspec.profile_version == "local-v2":
            for name in ("input_fault", "wheel_feedback_fault", "mu_input_fault"):
                self.last_diagnostics[name] = torch.zeros(self.B, device=self.device, dtype=torch.bool)
        self._solver_diagnostic_names = tuple(self.last_diagnostics)
        self._last_speed = torch.zeros(self.B, device=self.device)
        self._last_yaw = torch.zeros(self.B, device=self.device)
        self._last_delay = torch.full((self.B,), tracker.spec.delay, device=self.device)
        self.wheel_feedback = torch.zeros(self.B, device=self.device)
        self._has_feedback = False
        self._prev_command_hook = None
        self._prev_input_hook = None
        self._prev_reset_hook = None
        self._prev_solver = None
        self._installed = False
        self._graphed = None
        self._kernels = None
        self.acceleration_status = {"backend": "eager", "reason": "not prepared", "cache_hit": False}

    # -- friction ----------------------------------------------------------------
    def update(self, mu: torch.Tensor) -> None:
        """Set the per-env friction. Dynamic arms only; the fixed arm never calls this."""
        if self.gspec.mode not in DYNAMIC_MODES:
            raise RuntimeError(f"update() is for the oracle arm and the estimated arm; "
                               f"this one is {self.gspec.mode!r}")
        m = torch.as_tensor(mu, device=self.device, dtype=self.mu.dtype).reshape(-1)
        if m.numel() != self.B:
            raise ValueError(f"expected {self.B} friction values, got {m.numel()}")
        if self.gspec.profile_version == "local-v2":
            valid = torch.isfinite(m) & (m > 0)
            self.last_diagnostics["mu_input_fault"].copy_(~valid)
            m = torch.where(valid, m, self.mu)
        self.mu.copy_(m)

    def update_feedback(self, wheel_speed: torch.Tensor) -> None:
        """Measured wheel/VESC speed, already in physical m/s; never simulator state."""
        wheel = wheel_speed.reshape(self.B)
        if self.gspec.profile_version == "local-v2":
            valid = torch.isfinite(wheel)
            self.last_diagnostics["wheel_feedback_fault"].copy_(~valid)
            wheel = torch.where(valid, wheel, self.wheel_feedback)
        self.wheel_feedback.copy_(wheel)
        self._has_feedback = True

    def _inputs(self, action, v_meas, speed_cap, yaw_rate, delay):
        """Per-car finite sensor boundary; hold measured values and request a marked stop."""
        state_ok = torch.isfinite(self.tracker.u_prev).all(1) & torch.isfinite(self.tracker.u_seq).all((1, 2))
        self.tracker.u_prev = torch.where(state_ok[:, None], self.tracker.u_prev, torch.zeros_like(self.tracker.u_prev))
        self.tracker.u_seq = torch.where(state_ok[:, None, None], self.tracker.u_seq, torch.zeros_like(self.tracker.u_seq))
        speed_ok = torch.isfinite(v_meas)
        speed = torch.where(speed_ok, v_meas, self._last_speed)
        self._last_speed.copy_(speed)
        if yaw_rate is None:
            yaw_rate = speed.abs() * torch.tan(self.tracker.u_prev[:, 0]) / (self.wb + self.tracker.spec.k_us * speed.square())
        yaw_ok = torch.isfinite(yaw_rate)
        yaw = torch.where(yaw_ok, yaw_rate, self._last_yaw)
        self._last_yaw.copy_(yaw)
        if delay is None:
            delay = torch.full_like(speed, self.tracker.spec.delay)
        elif not torch.is_tensor(delay):
            delay = torch.full_like(speed, float(delay))
        delay_ok = torch.isfinite(delay) & (delay >= 0)
        latency = torch.where(delay_ok, delay, self._last_delay)
        self._last_delay.copy_(latency)
        fault = (~state_ok | ~speed_ok | ~yaw_ok | ~delay_ok |
                 ~torch.isfinite(speed_cap) | (speed_cap < 0) | ~torch.isfinite(action).all(1) |
                 self.last_diagnostics["wheel_feedback_fault"] | self.last_diagnostics["mu_input_fault"])
        self.last_diagnostics["input_fault"].copy_(fault)
        stop = torch.zeros_like(action)
        stop[:, _mpc.N_KNOTS:] = -1
        safe_action = torch.where(fault[:, None], stop, action)
        safe_cap = torch.where(fault, torch.zeros_like(speed_cap), speed_cap)
        return safe_action, speed, safe_cap, yaw, latency

    def _reset_inputs(self, ids):
        self._last_speed[ids] = 0
        self._last_yaw[ids] = 0
        self._last_delay[ids] = self.tracker.spec.delay
        self.wheel_feedback[ids] = 0
        for name in ("input_fault", "wheel_feedback_fault", "mu_input_fault"):
            self.last_diagnostics[name][ids] = False
        if self._prev_reset_hook is not None:
            self._prev_reset_hook(ids)

    def _command(self, u, z, v_meas, speed_cap):
        """Invert the nominal VESC loop once, with final tire/current/target checks.

        Soft policy speed caps affect the trajectory, never instantaneously clamp an
        overspeed recovery command. The outer calibration path remains its sole owner.
        """
        spec = self.gspec
        state = z[:, 0].clone()
        state[:, 3] = v_meas.abs()
        bounded, _, _, infeasible = project_local_control(state, u[:, 0], self.mu, spec,
                                                          self.tracker.spec, self.wb, self.s_max, self.v_max)
        speed = state[:, 3]
        wheel = self.wheel_feedback if self._has_feedback else v_meas
        r = resistance(speed, spec)
        power = spec.a_max * (spec.v_switch / wheel.abs().clamp_min(1e-3)).clamp(max=1)
        requested = bounded[:, 1] + r
        motor = torch.maximum(torch.minimum(requested, power), torch.full_like(power, -spec.a_brake))
        target = (wheel + spec.motor_tau * motor).clamp(0, spec.actuator_v_max)
        nominal_request = (target - wheel) / spec.motor_tau
        motor_violation = torch.maximum(-spec.a_brake - nominal_request, nominal_request - power).clamp_min(0)
        applied = torch.maximum(torch.minimum(nominal_request, power), torch.full_like(power, -spec.a_brake))
        net = applied - r
        # Target saturation can change the longitudinal force and the tire allocation.
        # Project again at the same selected steering, now using the actually converted net ax.
        kap = torch.tan(bounded[:, 0]) / (self.wb + self.tracker.spec.k_us * speed.square())
        lower, upper, tire_bad = local_accel_bounds(speed, kap, self.mu, spec)
        accel_violation = torch.maximum(lower - net, net - upper).clamp_min(0)
        self.last_diagnostics.update({"command_net_accel": net.detach().clone(),
            "command_requested_net_accel": bounded[:, 1].detach().clone(),
            "command_motor_request": nominal_request.detach().clone(),
            "command_target": target.detach().clone(),
            "command_motor_violation": motor_violation.detach().clone(),
            "command_accel_violation": accel_violation.detach().clone(),
            "command_infeasible": (self.last_diagnostics["input_fault"] | infeasible | tire_bad | (motor_violation > 1e-5) | (accel_violation > 1e-5)).detach().clone()})
        self.tracker.u_prev = torch.stack([bounded[:, 0], net], 1)
        return torch.stack([bounded[:, 0], target], 1)

    # -- install / release -------------------------------------------------------
    def _bound(self, *args):
        """The 12-argument solver the graph captures: the tracker's 11, plus `mu`.

        `gspec` and `consts` are closed over rather than passed, so the captured argument list is
        tensors and atoms only -- the same shape `capture_mpc` uses. Eager runs through this too, so
        an eager/graph comparison is a comparison of the same function.
        """
        return solve_grip(*args, self.gspec, consts=self._consts, out_bounds=self.last_bounds,
                          out_diagnostics=self.last_diagnostics, kernels=self._kernels)

    def _prepare_pointwise(self, full):
        """Warm the small math kernels before capture; failures remain explicit/eager."""
        import time
        import warnings
        key = (str(self.device), self.B, str(self.mu.dtype), self.gspec,
               tuple(asdict(self.tracker.spec).items()), self.wb, self.s_max, self.v_max)
        if key in _POINTWISE_FAILURES:
            self.acceleration_status = {"backend": "eager", "reason": _POINTWISE_FAILURES[key],
                                        "cache_hit": True, "warmup_seconds": 0.0}
            warnings.warn("Local controller pointwise fusion unavailable: " + _POINTWISE_FAILURES[key], RuntimeWarning)
            return
        started = time.monotonic()
        cache_hit = key in _POINTWISE_WARMED
        try:
            self._kernels = _pointwise_kernels()
            # Use the actual solver layouts, including first and subsequent iLQR
            # iterations. This is eager warmup, not whole-solver compilation.
            with torch.no_grad():
                self._bound(*full)
            torch.cuda.synchronize(self.device)
        except Exception as exc:
            self._kernels = None
            reason = f"{type(exc).__name__}: {exc}"
            _POINTWISE_FAILURES[key] = reason
            self.acceleration_status = {"backend": "eager", "reason": reason,
                "cache_hit": cache_hit, "warmup_seconds": time.monotonic() - started}
            warnings.warn("Local controller pointwise fusion unavailable: " + reason, RuntimeWarning)
        else:
            _POINTWISE_WARMED.add(key)
            self.acceleration_status = {"backend": "inductor-pointwise", "reason": "qualified warmup before graph capture",
                "cache_hit": cache_hit, "warmup_seconds": time.monotonic() - started}

    def install(self, graph: bool = False, example_args=None, adopt: bool = True) -> "GripMPC":
        if self.gspec.mode == "legacy":
            raise RuntimeError("the legacy arm installs nothing; do not call install()")
        if self._installed:
            raise RuntimeError("already installed")
        if graph:
            if example_args is None:
                raise ValueError("graph=True needs example_args recorded from a real step")
            if self.device.type != "cuda":
                raise RuntimeError(f"graph capture needs CUDA; this one is on {self.device}")
            from ..viewer.graph_fastpath import GraphedCallable
            a = list(example_args)
            if self.gspec.profile_version == "local-v2":
                # Capture examples can themselves come from a faulty sensor frame.
                # Sanitize before warmup/capture, using the same runtime boundary.
                a[:5] = self._inputs(*a[:5])
                a[5] = torch.nan_to_num(a[5], nan=0., posinf=0., neginf=0.)
                a[6] = torch.nan_to_num(a[6], nan=0., posinf=0., neginf=0.)
            full = a[:7] + [self.mu] + a[7:]                   # (…, warm, mu, spec, wb, s_max, v_max)
            if self.gspec.profile_version == "local-v2":
                self._prepare_pointwise(full)
            self._graphed = GraphedCallable(self._bound, full, name="solve_grip", guards={
                "pointwise_kernels": lambda: tuple((name, id(fn)) for name, fn in (self._kernels or {}).items()),
                "physical_spec": lambda: self.gspec,
                "constants": lambda: self._consts,
                "bounds_output": lambda: self.last_bounds,
                "diagnostic_outputs": lambda: {name: self.last_diagnostics.get(name)
                                                for name in self._solver_diagnostic_names},
            })
            # Ownership transfers exactly once. Training installs and steps on one thread, so it
            # adopts here; the viewer captures on its control thread and steps on another, and
            # passes adopt=False to call `adopt()` from the stepping thread instead.
            if adopt:
                self._graphed.adopt()
            fn = lambda *args: self._graphed(*args[:7], self.mu, *args[7:])
        else:
            fn = lambda *args: self._bound(*args[:7], self.mu, *args[7:])
        self._prev_solver = getattr(self.tracker, "_solver", None)
        self.tracker._solver = fn
        if self.gspec.profile_version == "local-v2":
            self._prev_command_hook = getattr(self.tracker, "_command_hook", None)
            self._prev_input_hook = getattr(self.tracker, "_input_hook", None)
            self._prev_reset_hook = getattr(self.tracker, "_reset_hook", None)
            self.tracker._reset_hook = self._reset_inputs
            self.tracker._command_hook = self._command
            self.tracker._input_hook = self._inputs
        self._installed = True
        return self

    def adopt(self) -> None:
        """Hand the captured grip-solver graph to the calling (stepping) thread. No-op when eager."""
        if self._graphed is not None:
            self._graphed.adopt()

    def release(self) -> None:
        """Restore whatever solver was installed before. Safe to call twice."""
        if self._installed:
            self.tracker._solver = self._prev_solver
            if self.gspec.profile_version == "local-v2":
                self.tracker._command_hook = self._prev_command_hook
                self.tracker._input_hook = self._prev_input_hook
                self.tracker._reset_hook = self._prev_reset_hook
            self._installed = False
        self._prev_solver = None
        self._graphed = None

    # -- diagnostics -------------------------------------------------------------
    def diagnostics(self, action, v_meas, speed_cap) -> dict:
        """Recompute the envelope for inspection. Read-only; does not touch the installed solver."""
        if self.gspec.profile_version == "local-v2":
            sp = self.tracker.spec
            k, length, v0, v1 = _mpc.decode(action, v_meas, self.v_max, speed_cap, sp)
            _, d = local_speed_profile(k, length, speed_cap, v0, v1, self.mu, v_meas,
                                       self.gspec, sp, self.v_max)
            d.update({name: value.detach().clone() for name, value in self.last_diagnostics.items()})
            d.update(mode=self.gspec.mode, mu=self.mu.detach().clone(),
                     a_brk_bound=self.last_bounds[:, 0].clone(), a_acc_bound=self.last_bounds[:, 1].clone(),
                     a_acc_path=d["a_acc"], a_brk_path=d["a_brk"],
                     v0_raw=v0, v1_raw=v1, profile_version=self.gspec.profile_version)
            return d
        sp = self.tracker.spec
        with torch.no_grad():
            k, Lp, v0, v1 = _mpc.decode(action, v_meas, self.v_max, speed_cap, sp)
            v_limit, d = speed_envelope(k, Lp, speed_cap, v0, v1, self.mu, self.gspec)
        return {
            "mode": self.gspec.mode, "mu": self.mu.detach().clone(),
            "v0_raw": v0.detach().clone(), "v1_raw": v1.detach().clone(),   # raw decoded plan speeds
            "v_plan": d["v_plan"].detach().clone(),
            "v_curve": d["v_curve"].detach().clone(),
            "v_limit": v_limit.detach().clone(),                            # limited reference speeds
            "a_acc_bound": d["a_acc_bound"].detach().clone(),               # what the iLQR is clamped to
            "a_brk_bound": d["a_brk_bound"].detach().clone(),               # and what the passes used
            "a_acc_path": d["a_acc"].detach().clone(),                      # per-sample, before the min
            "a_brk_path": d["a_brk"].detach().clone(),
            "a_lat_max": d["a_lat_max"].detach().clone(),
            "rho": d["rho"].detach().clone(),
            "kappa": d["kappa"].detach().clone(),
            # Reported, never applied, and neither is a whole-loop delivery fraction: the first is the
            # lead term alone at u_prev = 0, the second the measured steady ratio. Compare the
            # achieved acceleration against them; do not read either as a correction factor.
            "lead_fraction_first_step": LEAD_FRACTION_FIRST_STEP,
            "lead_fraction_steady": LEAD_FRACTION_STEADY,
            "transfer_applied": TRANSFER_APPLIED,
        }
