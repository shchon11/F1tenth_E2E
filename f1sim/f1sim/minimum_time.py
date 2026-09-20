"""Joint periodic path / kinetic-energy minimum-time nonlinear program.

The objective is traversal time only. Offsets and speeds are simultaneous decision
variables, with exact spline geometry derivatives and explicit axle/actuator/track
constraints. This is a quasi-steady model: it does not model slip relaxation, yaw
inertia or wheel transients, and a converged local NLP is not a global optimum.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.optimize import minimize, nnls
import torch

from .params import VehicleParams
from .track import resample_closed


@dataclass
class MinimumTimeResult:
    xy: np.ndarray
    v: np.ndarray
    diagnostics: dict


class _Problem:
    """Periodic cubic position and piecewise-linear squared speed on one mesh."""

    def __init__(self, seed, track, profile_kw, veh_width, margin, width_cap,
                 controls, density, *, deadline=None, sample_phases=None):
        from .raceline import normals, speed_profile, track_widths
        self.track, self.vehicle = track, profile_kw.get("vehicle") or VehicleParams()
        p = self.vehicle
        self.controls = controls
        self.samples = controls * density if sample_phases is None else len(sample_phases)
        self.deadline = deadline
        mu = profile_kw.get("mu")
        mu = p.mu if mu is None else float(mu)
        muf, mur = profile_kw.get("mu_front"), profile_kw.get("mu_rear")
        self.muf = mu * p.mu_f_scale if muf is None else float(muf)
        self.mur = mu * p.mu_r_scale if mur is None else float(mur)
        self.vmax = min(p.v_max, float(profile_kw.get("v_max", p.v_max)))
        self.acc = min(p.a_max, float(p.a_max if profile_kw.get("a_acc") is None else profile_kw["a_acc"]))
        self.brake = min(p.a_brake, float(p.a_brake if profile_kw.get("a_brake") is None else profile_kw["a_brake"]))
        self.lateral = profile_kw.get("a_lat")
        limits = [self.vmax, self.acc, self.brake, self.muf, self.mur, p.s_max, p.sv_max,
                  p.v_switch, veh_width, p.length, p.lf, p.lr]
        if not np.isfinite(limits).all() or min(limits) <= 0:
            raise ValueError("minimum-time vehicle and grip limits must be finite and positive")
        if not np.isfinite(margin) or margin < 0 or not 0 <= p.drive_split_r <= 1:
            raise ValueError("invalid track margin or drivetrain split")
        if self.lateral is not None and (not np.isfinite(self.lateral) or self.lateral <= 0):
            raise ValueError("lateral acceleration cap must be finite and positive")
        ref = resample_closed(seed, controls)
        normal = normals(ref)
        knots = np.arange(controls + 1) / controls
        query = np.arange(self.samples) / self.samples if sample_phases is None else np.asarray(sample_phases)
        self.phases = query
        self.quadrature = torch.tensor((np.mod(np.roll(query, -1) - query, 1.)
                                         + np.mod(query - np.roll(query, 1), 1.)) / 2)
        spline = CubicSpline(knots, np.vstack([np.eye(controls), np.eye(controls)[0]]),
                             bc_type="periodic")
        self.basis = [torch.tensor(spline(query, nu=i), dtype=torch.float64) for i in range(4)]
        left_phase = np.nextafter(query, -np.inf)
        left_phase[query == 0.] = np.nextafter(1., 0.)
        self.third_left = torch.tensor(spline(left_phase, nu=3), dtype=torch.float64)
        self.ref, self.normal = torch.tensor(ref), torch.tensor(normal)
        self.reference_length = np.linalg.norm(np.roll(ref, -1, axis=0) - ref, axis=1).sum()
        u = query * controls
        u = np.where(np.abs(u - np.round(u)) < 1e-10, np.round(u), u)
        index = np.floor(u).astype(int) % controls
        weight = u - np.floor(u)
        energy_basis = np.zeros((self.samples, controls))
        energy_basis[np.arange(self.samples), index] = 1 - weight
        energy_basis[np.arange(self.samples), (index + 1) % controls] += weight
        self.energy_basis = torch.tensor(energy_basis)
        derivative = np.zeros_like(energy_basis)
        derivative[np.arange(self.samples), index] = -controls
        derivative[np.arange(self.samples), (index + 1) % controls] = controls
        previous = derivative.copy()
        at_knot = np.isclose(weight, 0., atol=1e-10)
        rows = np.flatnonzero(at_knot)
        previous[rows] = 0
        previous[rows, (index[rows] - 1) % controls] = -controls
        previous[rows, index[rows]] = controls
        self.energy_derivatives = [torch.tensor(derivative), torch.tensor(previous)]
        self.edt = torch.tensor(track.edt.astype(float) - distance_transform_edt(track.occupancy) * track.resolution)
        wl, wr = track_widths(track, ref)
        if width_cap is not None:
            wl, wr = np.minimum(wl, width_cap), np.minimum(wr, width_cap)
        # Determine narrow-passage reserve from the WHOLE oriented body, not one
        # center normal ray (which misses obstacle corners beside the front/rear).
        # Half the available extra clearance is retained, up to the requested cap.
        body_radius = np.hypot(veh_width / 2, p.length / 6)
        solid_radius = body_radius + track.resolution / np.sqrt(2) + .003
        tangent = np.column_stack([normal[:, 1], -normal[:, 0]])
        fractions = np.linspace(0., 1., max(33, int(np.ceil((wl + wr).max() / (track.resolution / 2))) + 1))
        offsets = -wr[:, None] + (wl + wr)[:, None] * fractions
        centers = ref[:, None, :] + normal[:, None, :] * offsets[:, :, None]
        capacity = np.full(offsets.shape, np.inf)
        for longitudinal in (-p.length / 3, 0., p.length / 3):
            probes = centers + tangent[:, None, :] * longitudinal
            rc = np.stack([(probes[..., 1] - track.origin[1]) / track.resolution,
                           (probes[..., 0] - track.origin[0]) / track.resolution])
            capacity = np.minimum(capacity, map_coordinates(track.edt, rc, order=1, mode="constant", cval=0.))
        reserve = np.minimum(margin, np.maximum(0., capacity.max(1) - solid_radius) / 2)
        self.margin_range = [float(reserve.min()), float(reserve.max())]
        half_width = veh_width / 2
        self.bounds = list(zip(-np.maximum(wr - half_width - reserve, 0),
                               np.maximum(wl - half_width - reserve, 0)))
        self.bounds += [(1e-8, 1.)] * controls
        # Three overlapping circles cover the complete rectangle (including its
        # interior), not just the center/corners. Raster-cell uncertainty is reserved.
        self.body_radius = body_radius
        self.radius = (torch.tensor(energy_basis @ reserve) + self.body_radius
                       + track.resolution / np.sqrt(2) + .003)
        init_kw = dict(profile_kw, vehicle=p, a_lat=self.lateral or 9.81 * min(self.muf, self.mur))
        self.initial = np.r_[np.zeros(controls), (speed_profile(ref, **init_kw) / self.vmax) ** 2]
        self.initial[controls:] = np.clip(self.initial[controls:], 1e-8, 1)
        self.last_x = self.last_y = self.last_jac = None
        self.evaluations = 0

    def field(self, xy):
        """Bilinear EDT, zero outside the map (never extrapolate a free corridor)."""
        t = self.track
        col = (xy[:, 0] - t.origin[0]) / t.resolution
        row = (xy[:, 1] - t.origin[1]) / t.resolution
        valid = (col >= 0) & (row >= 0) & (col <= self.edt.shape[1] - 1) & (row <= self.edt.shape[0] - 1)
        col = col.clamp(0, self.edt.shape[1] - 1)
        row = row.clamp(0, self.edt.shape[0] - 1)
        i = torch.floor(row).to(torch.int64).clamp(0, self.edt.shape[0] - 2)
        j = torch.floor(col).to(torch.int64).clamp(0, self.edt.shape[1] - 2)
        a, b = row - i, col - j
        distance = (self.edt[i, j] * (1 - a) * (1 - b) + self.edt[i + 1, j] * a * (1 - b)
                    + self.edt[i, j + 1] * (1 - a) * b + self.edt[i + 1, j + 1] * a * b)
        return torch.where(valid, distance, torch.zeros_like(distance))

    def geometry(self, z):
        points = self.ref + self.normal * z[:self.controls, None]
        xy, d1, d2, d3 = [basis @ points for basis in self.basis]
        norm = torch.linalg.vector_norm(d1, dim=1).clamp_min(1e-9)
        tangent = d1 / norm[:, None]
        kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / norm ** 3
        ds = torch.linalg.vector_norm(torch.roll(xy, -1, 0) - xy, dim=1).clamp_min(1e-9)
        squared_speed = self.energy_basis @ z[self.controls:] * self.vmax ** 2
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        cross_derivative = d1[:, 0] * d3[:, 1] - d1[:, 1] * d3[:, 0]
        kappa_arc_derivative = (cross_derivative / norm ** 3
                                - 3 * cross * (d1 * d2).sum(1) / norm ** 5) / norm
        d3_left = self.third_left @ points
        cross_left = d1[:, 0] * d3_left[:, 1] - d1[:, 1] * d3_left[:, 0]
        kappa_left = (cross_left / norm ** 3 - 3 * cross * (d1 * d2).sum(1) / norm ** 5) / norm
        return xy, tangent, kappa, ds, squared_speed, norm, kappa_arc_derivative, kappa_left

    def values(self, z):
        p = self.vehicle
        xy, tangent, curvature, ds, u, norm, kappa_s, kappa_left = self.geometry(z)
        v = torch.sqrt(u)
        # Integrate dt/d(reference phase), rather than treating uneven spline
        # stations as equal arc lengths. Energy derivatives enforce both one-sided
        # accelerations at each knot, including the periodic seam.
        traversal_time = (self.quadrature * norm / v).sum()
        delta = torch.atan((p.lf + p.lr) * curvature)
        steering_rate = (p.lf + p.lr) * kappa_s * v / (1 + ((p.lf + p.lr) * curvature) ** 2)
        steering_rate_left = (p.lf + p.lr) * kappa_left * v / (1 + ((p.lf + p.lr) * curvature) ** 2)
        reference_tangent = self.basis[1] @ self.ref
        forward = (tangent * reference_tangent).sum(1) / torch.linalg.vector_norm(reference_tangent, dim=1)
        constraints = [1 - (delta / p.s_max) ** 2,
                       1 - (steering_rate / p.sv_max) ** 2,
                       1 - (steering_rate_left / p.sv_max) ** 2,
                       forward - .05, norm / self.reference_length - .05]
        split = p.drive_split_r if p.wheel_model else 1.
        axles = [(1 - split, p.lr / (p.lf + p.lr), -p.h / (p.lf + p.lr), self.muf),
                 (split, p.lf / (p.lf + p.lr), p.h / (p.lf + p.lr), self.mur)]
        for energy_derivative in self.energy_derivatives:
            ax = (energy_derivative @ z[self.controls:] * self.vmax ** 2) / (2 * norm)
            speed_squared, kappa = u, curvature
            demand = ax + p.c_roll + p.c_drag * speed_squared
            constraints += [(self.acc - demand) / self.acc,
                            (self.acc * p.v_switch - demand * torch.sqrt(speed_squared)) / (self.acc * p.v_switch),
                            (self.brake + demand) / self.brake]
            for share, load, transfer, grip in axles:
                normal = 9.81 * load + transfer * ax
                fx, fy = share * demand, load * speed_squared * kappa
                constraints += [normal / (9.81 * load),
                                ((grip * normal) ** 2 - fx ** 2 - fy ** 2) / (grip * 9.81 * load) ** 2]
        if self.lateral is not None:
            constraints.append(1 - (u * curvature / self.lateral) ** 2)
        for longitudinal in (-p.length / 3, 0., p.length / 3):
            constraints.append(self.field(xy + tangent * longitudinal) - self.radius)
        return torch.cat([traversal_time.view(1), *constraints])

    def evaluate(self, z, jac=False):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError("minimum-time deadline exceeded; partial geometry discarded")
        if self.last_x is None or not np.array_equal(self.last_x, z):
            self.last_x = z.copy()
            self.last_y = self.values(torch.tensor(z)).detach().numpy()
            self.last_jac = None
            self.evaluations += 1
        if jac and self.last_jac is None:
            self.last_jac = torch.func.jacfwd(self.values)(torch.tensor(z)).detach().numpy()
        return self.last_jac if jac else self.last_y

    def restore(self, z):
        """Restore geometry first, then physical feasibility at modest constant speed."""
        energy = (min(2., .4 * self.vmax) / self.vmax) ** 2
        start = z.copy()
        start[self.controls:] = energy

        def loss(x, geometry_only=False):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                raise TimeoutError("minimum-time deadline exceeded; partial geometry discarded")
            value = torch.tensor(x, requires_grad=True)
            complete = (torch.cat([value, torch.full((self.controls,), energy, dtype=torch.float64)])
                        if geometry_only else value)
            g = self.values(complete)[1:]
            if geometry_only:
                n = self.samples
                g = torch.cat([g[:n], g[3*n:5*n], g[-3*n:]])
            cost = torch.minimum(g, torch.zeros_like(g)).square().sum()
            grad, = torch.autograd.grad(cost, value)
            return float(cost.detach()), grad.numpy()

        geometric = minimize(lambda x: loss(x, True), start[:self.controls], jac=True,
                             method="L-BFGS-B", bounds=self.bounds[:self.controls],
                             options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-8, "maxls": 50})
        start[:self.controls] = geometric.x
        if self.evaluate(start)[1:].min() >= -1e-6:
            return start
        result = minimize(loss, start, jac=True, method="L-BFGS-B", bounds=self.bounds,
                          options={"maxiter": 1000, "ftol": 1e-14, "gtol": 1e-8, "maxls": 50})
        return result.x

    def solve(self, z, maxiter):
        initial = z.copy()
        if self.evaluate(z)[1:].min() < -1e-5:
            z = self.restore(z)
        scale = max(1., self.reference_length / self.vmax)
        for attempt in range(2):
            result = minimize(lambda x: self.evaluate(x)[0] / scale, z,
                              jac=lambda x: self.evaluate(x, True)[0] / scale, method="SLSQP", bounds=self.bounds,
                              constraints={"type": "ineq", "fun": lambda x: self.evaluate(x)[1:],
                                           "jac": lambda x: self.evaluate(x, True)[1:]},
                              options={"maxiter": maxiter, "ftol": 1e-8})
            result.fun = float(result.fun * scale)
            violation = float(max(0., -self.evaluate(result.x)[1:].min()))
            if result.success and violation <= 2e-6:
                return result
            z = self.restore(initial)
        raise ValueError(f"{self.track.name}: minimum-time solve did not converge "
                         f"({result.message}; normalized violation {violation:.3g})")


def _stationarity(problem, z):
    """Independent nonnegative multiplier fit for first-order KKT stationarity."""
    values, jac = problem.evaluate(z), problem.evaluate(z, True)
    active = [row for g, row in zip(values[1:], jac[1:]) if g < 1e-5]
    eye = np.eye(len(z))
    for i, (lo, hi) in enumerate(problem.bounds):
        if z[i] - lo < 1e-6:
            active.append(eye[i])
        if hi - z[i] < 1e-6:
            active.append(-eye[i])
    if not active:
        residual = jac[0]
    else:
        matrix = np.asarray(active).T
        try:
            multipliers, _ = nnls(matrix, jac[0], maxiter=max(1000, 20 * matrix.shape[1]))
            residual = jac[0] - matrix @ multipliers
        except RuntimeError:
            return float("inf")
    return float(np.linalg.norm(residual, ord=np.inf) / max(1., np.linalg.norm(jac[0], ord=np.inf)))


def _audit_export(xy, speed, problem, phases):
    """Audit the actual returned samples, independently of the collocation variables."""
    from .raceline import curvature, normals
    p = problem.vehicle
    if not np.isfinite(xy).all() or not np.isfinite(speed).all() or np.min(speed) <= 0:
        raise ValueError("minimum-time export contains invalid geometry or speeds")
    ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
    kappa = curvature(xy)
    u = speed ** 2
    acceleration = (np.roll(u, -1) - u) / (2 * ds)
    steer = np.arctan((p.lf + p.lr) * kappa)
    slew = np.abs(np.roll(steer, -1) - steer) * (speed + np.roll(speed, -1)) / (2 * ds)
    residuals = [float(np.max(speed / problem.vmax - 1)),
                 float(np.max(np.abs(steer) / p.s_max - 1)),
                 float(np.max(slew / p.sv_max - 1))]
    split = p.drive_split_r if p.wheel_model else 1.
    axle_ratio = 0.
    for endpoint_u, endpoint_k in [(u, kappa), (np.roll(u, -1), np.roll(kappa, -1))]:
        demand = acceleration + p.c_roll + p.c_drag * endpoint_u
        residuals.extend([float(np.max(demand / problem.acc - 1)),
                          float(np.max(demand * np.sqrt(endpoint_u) / (problem.acc * p.v_switch) - 1)),
                          float(np.max(-demand / problem.brake - 1))])
        for share, load, transfer, grip in [
                (1 - split, p.lr / (p.lf + p.lr), -p.h / (p.lf + p.lr), problem.muf),
                (split, p.lf / (p.lf + p.lr), p.h / (p.lf + p.lr), problem.mur)]:
            normal = 9.81 * load + transfer * acceleration
            if np.min(normal) <= 0:
                raise ValueError("minimum-time export unloads an axle")
            ratio = ((share * demand) ** 2 + (load * endpoint_u * endpoint_k) ** 2) / (grip * normal) ** 2
            axle_ratio = max(axle_ratio, float(ratio.max()))
            residuals.append(float(ratio.max() - 1))
    if problem.lateral is not None:
        residuals.append(float(np.max(np.abs(u * kappa) / problem.lateral - 1)))
    # Dense interpolation of the returned polyline, including segment interiors.
    count = max(len(xy), int(np.ceil(ds.sum() / (problem.track.resolution / 3))))
    arc = np.r_[0., np.cumsum(ds)]
    query = np.arange(count) * arc[-1] / count
    dense = np.column_stack([np.interp(query, arc, np.r_[xy[:, j], xy[0, j]]) for j in range(2)])
    tangent = np.column_stack([normals(xy)[:, 1], -normals(xy)[:, 0]])
    direction = np.column_stack([np.interp(query, arc, np.r_[tangent[:, j], tangent[0, j]]) for j in range(2)])
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = np.interp(phases, np.r_[problem.phases, 1.], np.r_[problem.radius.numpy(), problem.radius[0].item()]) - .003
    reserve = np.interp(query, arc, np.r_[radius, radius[0]])
    clearance = min(float(np.min(problem.field(torch.tensor(dense + offset * direction)).numpy() - reserve))
                    for offset in (-p.length / 3, 0., p.length / 3))
    violation = max(0., *residuals)
    if violation > 2e-5 or clearance < -1e-4:
        raise ValueError(f"minimum-time export failed physical audit: violation={violation:.3g}, clearance={clearance:.4g}m")
    return dict(normalized_physical_violation=violation, max_axle_friction_ratio=axle_ratio,
                minimum_body_clearance_above_reserve_m=clearance,
                maximum_steering_rad=float(np.max(np.abs(steer))), samples=count)


def solve_minimum_time(seed, track, veh_width, margin, profile_kw, width_cap=None,
                       *, max_seconds=None, maxiter=250, controls=None):
    """Return a converged, mesh-checked local minimum-time line or raise.

    No feasible iterate/seed is silently substituted for an unconverged solution.
    Cache owners must save diagnostics with the geometry. Mesh checks are distinct
    from dynamic closed-loop validation, which this reduced model cannot provide.
    """
    from .raceline import Raceline, speed_profile
    if maxiter <= 0:
        raise ValueError(f"{track.name}: no raceline satisfies a converged minimum-time solve")
    started = time.monotonic()
    deadline = None if max_seconds is None else started + max_seconds
    length = np.linalg.norm(np.roll(seed, -1, axis=0) - seed, axis=1).sum()
    automatic_controls = controls is None
    controls = controls or max(16, int(np.ceil(length / .8)))
    initial_mesh = _Problem(seed, track, profile_kw, veh_width, margin, width_cap,
                            controls, 4, deadline=deadline)
    if automatic_controls and initial_mesh.evaluate(initial_mesh.initial)[1:1 + initial_mesh.samples].min() < -.2:
        controls *= 2
    records, z = [], None
    phases = np.arange(controls * 4) / (controls * 4)
    # Adaptive constraint exchange: add actual between-station violations instead
    # of declaring a coarse feasible trajectory valid or hiding a grip derating.
    check = _Problem(seed, track, profile_kw, veh_width, margin, width_cap,
                     controls, 32, deadline=deadline)
    for refinement in range(10):
        problem = _Problem(seed, track, profile_kw, veh_width, margin, width_cap,
                           controls, 4, deadline=deadline, sample_phases=phases)
        result = problem.solve(problem.initial if z is None else z, maxiter)
        z = result.x
        fine = check.evaluate(z)
        difference = abs(fine[0] - result.fun) / result.fun
        violation = float(max(0., -fine[1:].min()))
        records.append(dict(samples=problem.samples, iterations=int(result.nit),
                            time_s=float(result.fun), dense_time_relative_change=float(difference),
                            dense_normalized_violation=violation))
        if violation <= 2e-5 and difference <= .002:
            break
        by_station = fine[1:].reshape(-1, check.samples).min(0)
        minima = ((by_station <= np.roll(by_station, 1))
                  & (by_station <= np.roll(by_station, -1)) & (by_station < -1e-6))
        selected = np.flatnonzero(minima)
        if not len(selected):
            selected = np.argsort(by_station)[:16]
        phases = np.unique(np.r_[phases, check.phases[selected]])
        if difference > .002:
            phases = np.unique(np.r_[phases, np.arange(controls * 8) / (controls * 8)])
    else:
        raise ValueError(f"{track.name}: minimum-time dense mesh validation failed: {records[-1]}")
    kkt = _stationarity(problem, z)
    if not np.isfinite(kkt) or kkt > .005:
        raise ValueError(f"{track.name}: minimum-time stationarity residual {kkt:.3g} exceeds tolerance")
    # Export a uniform arc representation required by the teacher. Re-evaluate the
    # existing physical speed envelope at this representation and quantify projection
    # loss; never attach the NLP's speed samples to a different resampled geometry.
    dense_xy = check.geometry(torch.tensor(z))[0].numpy()
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.roll(dense_xy, -1, axis=0) - dense_xy, axis=1))]
    output_count = max(len(seed), int(np.ceil(arc[-1] / .04)))
    output_arc = np.arange(output_count) * arc[-1] / output_count
    output_phases = np.interp(output_arc, arc, np.r_[check.phases, 1.])
    knots_xy = (problem.ref + problem.normal * torch.tensor(z[:controls, None])).numpy()
    spline = CubicSpline(np.arange(controls + 1) / controls,
                         np.vstack([knots_xy, knots_xy[0]]), bc_type="periodic")
    xy = spline(output_phases)
    full_kw = dict(profile_kw)
    if full_kw.get("a_lat") is None:
        full_kw["a_lat"] = 9.81 * min(problem.muf, problem.mur)
    normalized_energy = np.interp(output_phases, np.arange(controls + 1) / controls,
                                  np.r_[z[controls:], z[controls]])
    speed = speed_profile(xy, speed_ceiling=np.sqrt(normalized_energy) * problem.vmax, **full_kw)
    exported = Raceline.from_xy(xy, speed)
    loss = (exported.lap_time - float(fine[0])) / float(fine[0])
    if abs(loss) > .02:
        raise ValueError(f"{track.name}: minimum-time export differs {loss:.1%} from the solved lap time")
    export_audit = _audit_export(xy, speed, problem, output_phases)
    diagnostics = dict(model="periodic-quasisteady-axle-nlp-v1", status="converged",
                       global_optimum_proven=False, dynamic_validation=False,
                       objective="integral(norm(dxy/dphase)/speed, phase=0..1)", path_controls=controls,
                       requested_margin_m=float(margin), effective_margin_m=problem.margin_range,
                       solver_message=str(result.message), kkt_stationarity=kkt,
                       normalized_constraint_violation=float(max(0., -problem.evaluate(z)[1:].min())),
                       mesh_checks=records, solved_time_s=float(fine[0]),
                       exported_time_s=exported.lap_time, export_relative_time_change=float(loss),
                       export_audit=export_audit, geometry_discretization_reserve_m=.003,
                       path_control_mesh_convergence_tested=False, elapsed_s=time.monotonic() - started)
    return MinimumTimeResult(xy, speed, diagnostics)
