"""Privileged teacher: pure pursuit on the optimized raceline + speed profile (batched torch).
Uses ground-truth state; only valid inside the simulator. Serves as IL teacher and as the
baseline the RL student must beat."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

import torch

import numpy as np

from .raceline import Raceline, curvature
from .track import resample_closed

#: `label_grip` as an integer, for the per-car form. The order is the one `f1sim.opponent_slots`
#: lists, and it is frozen: a recorded code keeps its meaning.
LABEL_GRIP_NAMES = ("true", "nominal", "conservative")
LABEL_GRIP_CODE = {name: i for i, name in enumerate(LABEL_GRIP_NAMES)}


def plan_geometry_speed(state: torch.Tensor, plan_speed: Optional[torch.Tensor]) -> torch.Tensor:
    """SI speed used to encode/decode plan geometry; standalone callers retain body speed."""
    if plan_speed is None:
        return state[:, 3]
    if (not torch.is_tensor(plan_speed) or plan_speed.shape != (state.shape[0],)
            or not plan_speed.is_floating_point() or plan_speed.device != state.device):
        raise ValueError("plan_speed must be a floating (B,) tensor in m/s on the state device")
    return plan_speed


def fit_knots(k: torch.Tensor, tx: torch.Tensor, ty: torch.Tensor, Lp: torch.Tensor, fr: torch.Tensor,
              k_lim: torch.Tensor, iters: int, fractions: torch.Tensor) -> torch.Tensor:
    """Gauss-Newton fit of the plan's curvature knots so the integrated path passes through the
    body-frame points (tx, ty) (B, M), which sit at fractions `fr` (M,) of the plan length `Lp`.

    `k` (B, N_KNOTS) is the initial guess and also what the ridge pulls towards; `k_lim` (B,) bounds
    every knot; `fractions` are the step lengths tried per iteration. Shared by every teacher that
    expresses a path in the plan action space, so all of them fit it the same way.
    """
    from .mpc import N_KNOTS, path_points
    B, dev = k.shape[0], k.device
    M = fr.shape[0]
    k_init = k.clone()

    lam, mu, eps = 1e-2, 0.3, 0.02                             # GN damping, ridge towards the pure-pursuit / raceline guess
    # The six finite-difference perturbations are independent. Evaluate them
    # with the unperturbed path in one batch instead of launching seven tiny
    # path integrations and constructing six CUDA scalar tensors per iteration.
    basis = torch.eye(N_KNOTS, device=dev, dtype=k.dtype)
    perturb = torch.cat([torch.zeros_like(basis[:1]), eps * basis], 0)
    lengths = Lp[:, None].expand(-1, N_KNOTS + 1).reshape(-1)
    sample_idx = (fr * 24).round().long()
    trial_lengths = Lp[:, None].expand(-1, len(fractions)).reshape(-1)
    rows = torch.arange(B, device=dev)
    for _ in range(iters):
        paths = (k[:, None, :] + perturb[None]).reshape(-1, N_KNOTS)
        x, y, _, _ = path_points(paths, lengths, 25)
        x = x[:, sample_idx].reshape(B, N_KNOTS + 1, M)
        y = y[:, sample_idx].reshape(B, N_KNOTS + 1, M)
        residuals = torch.cat([x - tx[:, None], y - ty[:, None]], 2)
        r0 = residuals[:, 0]
        J = ((residuals[:, 1:] - r0[:, None]) / eps).transpose(1, 2)
        A = J.transpose(1, 2) @ J + (lam + mu) * basis
        g = J.transpose(1, 2) @ r0[..., None] + mu * (k - k_init)[..., None]
        # `solve_ex` is the same LU solve without the error check, which reads the pivots back
        # to the host every call. A carries a 0.31 ridge, so it is never singular.
        step = torch.linalg.solve_ex(A, g)[0].squeeze(-1)
        # Curved minimum-time paths can make a full GN step overshoot and
        # oscillate. Accept only a decrease in the actual fit+ridge objective.
        trials = k[:, None] - fractions[None, :, None] * step[:, None]
        trials = torch.maximum(torch.minimum(trials, k_lim[:, None, None]), -k_lim[:, None, None])
        px, py, _, _ = path_points(trials.reshape(-1, N_KNOTS), trial_lengths, 25)
        px = px[:, sample_idx].reshape(B, len(fractions), M)
        py = py[:, sample_idx].reshape(B, len(fractions), M)
        costs = ((px - tx[:, None]).square() + (py - ty[:, None]).square()).sum(-1)
        costs = costs + mu * (trials - k_init[:, None]).square().sum(-1)
        k = trials[rows, costs.argmin(1)]
    return k


#: The speed profile the teacher drives when no limits are given: the minimum-curvature line's
#: own 6 / 6 / 3, which is what `--teacher-a-*` documents as its default and what every run before
#: the limits became overridable used.
DEFAULT_A_LAT, DEFAULT_A_ACC, DEFAULT_A_BRAKE = 6.0, 6.0, 3.0


class RacelineTeacher:
    """mode "pp" (default): pure pursuit on the raceline with understeer compensation
    (effective wheelbase L + k_us v^2; the simulated car turns ~20 % less than kinematic at
    4 m/s^2 lateral, ~35 % at 6) and privileged latency / servo / calibration / grip compensation.
    mode "stanley": curvature feed-forward + Stanley feedback at the front axle; more accurate
    without latency but loses to pure pursuit under domain randomization (delays up to 80 ms):
    on the 26-track set at 6 m/s, collisions per env per 15 s: pp 0.13 vs stanley 0.39."""

    #: (T, N) largest |lateral offset| each raceline point tolerates, set by the env when opponent
    #: behaviour events are on (f1sim.opponent_events.raceline_offset_limit). None: no clamp, which
    #: is also what every caller that never passes `offset` sees.
    offset_limit: Optional[torch.Tensor] = None

    def __init__(self, raceline, wheelbase: float = 0.3302, device="cpu", mode: str = "pp",
                 lookahead_gain: float = 0.35, lookahead_min: float = 0.6, lookahead_max: float = 2.5,
                 speed_lookahead_time: float = 0.35, lateral_slowdown: float = 0.6,
                 speed_scale: float = 1.0, steer_max: float = 0.4189,
                 k_e: float = 2.5, k_psi: float = 1.0, v_soft: float = 1.0, k_us: float = 0.003,
                 ff_time: float = 0.05, k_e_pp: float = 0.0,
                 mu_nominal: float = 1.0489, mu_f_scale_nominal: float = 0.92,
                 recover_time: float = 0.0, v_recover_min: float = 0.6, a_lat_recover: float = 6.0,
                 v_max_profile: float = 10.0, a_lat: Optional[float] = None,
                 a_acc: Optional[float] = None, a_brake: Optional[float] = None, vehicle=None):
        self.device = torch.device(device)
        from .params import VehicleParams
        nominal = VehicleParams()
        self.vehicle = vehicle if vehicle is not None else replace(
            nominal, lf=wheelbase * nominal.lf / (nominal.lf + nominal.lr),
            lr=wheelbase * nominal.lr / (nominal.lf + nominal.lr),
            s_max=steer_max, mu=mu_nominal, mu_f_scale=mu_f_scale_nominal)
        vehicle = self.vehicle
        # `None` means "the default profile", which is what every caller that omits them expects;
        # they are Optional so an explicit limit can override, not so they can be missing.
        a_lat = DEFAULT_A_LAT if a_lat is None else a_lat
        a_acc = DEFAULT_A_ACC if a_acc is None else a_acc
        a_brake = DEFAULT_A_BRAKE if a_brake is None else a_brake
        rls = [raceline] if isinstance(raceline, Raceline) else list(raceline)     # one per track id
        N = max(len(r.xy) for r in rls)
        # speed profiles per grip level: the teacher is privileged, so it brakes and corners for the
        # friction *this* car has (a_lat and a_acc scale with grip; braking does not, see below)
        from .raceline import speed_profile
        self.grip_levels = np.r_[np.linspace(0.45, 1.0, 12), 1.1, 1.2]
        self.nominal_grip_index = 11
        xy, v, kap = [], [], []
        for r in rls:
            xr = resample_closed(r.xy, N)
            xy.append(xr); kap.append(curvature(xr))
            # Recompute the combined-slip profile at each actual friction level.
            # Motor/regen caps do not scale with grip; the tire constraints decide
            # whether the surface or the actuator is limiting each segment.
            v.append(np.stack([speed_profile(xr, v_max_profile, a_lat, a_acc, a_brake,
                                            mu=mu_nominal * g, vehicle=vehicle)
                               for g in self.grip_levels]))   # (K, N)
        self.a_lat = float(a_lat)                                                          # nominal-grip lateral budget of the profiles
        self.a_acc, self.a_brake = float(a_acc), float(a_brake)                            # the profiles' drive / braking limits
        self.xy = torch.tensor(np.stack(xy), dtype=torch.float32, device=self.device)     # (T, N, 2)
        self.v_grip = torch.tensor(np.stack(v), dtype=torch.float32, device=self.device)  # (T, K, N)
        self.v = self.v_grip[:, self.nominal_grip_index]                                    # nominal grip (T, N)
        self.grip_levels_t = torch.tensor(self.grip_levels, dtype=torch.float32, device=self.device)
        self.kappa = torch.tensor(np.stack(kap), dtype=torch.float32, device=self.device) # (T, N) left +
        tan = torch.roll(self.xy, -1, 1) - torch.roll(self.xy, 1, 1)
        self.tan = tan / tan.norm(dim=2, keepdim=True).clamp_min(1e-9)                     # (T, N, 2)
        self.length = torch.tensor([r.length for r in rls], dtype=torch.float32, device=self.device)
        self.N = N
        self.ds = self.length / N                                                          # (T,)
        self.L = wheelbase
        self.mode = mode
        self.k_ld, self.ld_min, self.ld_max = lookahead_gain, lookahead_min, lookahead_max
        self.k_e, self.k_psi, self.v_soft, self.k_us, self.ff_time, self.k_e_pp = k_e, k_psi, v_soft, k_us, ff_time, k_e_pp
        self.t_v = speed_lookahead_time
        self.lat_slow = lateral_slowdown
        self.speed_scale = speed_scale
        self.steer_max = steer_max
        self.mu_nom, self.mu_f_nom = mu_nominal, mu_f_scale_nominal
        # recover_time > 0: cap the commanded speed by what can still be turned back onto the lane.
        # The existing slowdown reads *lateral* error only, so a car sitting on the line facing
        # backwards is told to drive at full profile speed: measured 3.06 m/s at 180 deg of heading
        # error, on a commanded radius of 0.74 m -- 12.7 m/s^2 of lateral acceleration against the
        # ~6 m/s^2 the profile itself assumes. That is not a recovery demonstration, and DAgger can
        # only teach what the teacher shows, so weighting those samples harder would teach it harder.
        self.recover_time, self.v_recover_min, self.a_lat_recover = recover_time, v_recover_min, a_lat_recover

    @torch.no_grad()
    def plan_action(self, state: torch.Tensor, P=None, tid: Optional[torch.Tensor] = None, v_max: float = 8.0,
                    spec=None, iters: int = 6, offset: Optional[torch.Tensor] = None,
                    idx: Optional[torch.Tensor] = None,
                    plan_speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The teacher as a *planner*: the raceline segment ahead of the car expressed in the plan
        action space (f1sim.mpc: curvature knots along the next L_p of arc + start/end speeds).
        Gauss-Newton fits the knots so the integrated path passes through the raceline points
        ahead (in the body frame), starting from the raceline's own curvature there; this is what
        a plan-space student imitates.

        offset: (B,) metres left of the raceline to plan through (opponent behaviour events), clamped
        by `offset_limit`. None leaves this function exactly as it was.

        idx: the caller's own raceline projection of `state[:, :2]`, when it already has one. Purely
        a saving -- `project` is an argmin over every raceline point, and a caller that evaluates
        several plans from ONE pose (`f1sim.interactive_teacher`) would otherwise pay for the same
        projection once per candidate. None computes it here, as before.

        plan_speed: (B,) incoming tracker odometry speed in m/s. It sets the plan's metric length
        and geometric fit; privileged body state and grip still select the speed profile. None
        preserves standalone body-speed geometry. The caller must pass the same sample to decode.
        """
        from .mpc import N_KNOTS, PlanSpec, encode, encode_envelope, encode_knots, path_points, plan_length
        spec = spec or PlanSpec()
        xy, yaw, vx = state[:, :2], state[:, 2], state[:, 3]
        geometry_speed = plan_geometry_speed(state, plan_speed)
        B = xy.shape[0]; dev = xy.device
        tid = torch.zeros(B, dtype=torch.long, device=dev) if tid is None else tid
        # One projection, used twice below (the lateral error the off-line slowdown reads is the
        # same call's second return). It used to be made twice, which on a long raceline is the
        # single most expensive thing this function does.
        idx, lat_err = (self.project(xy, tid) if idx is None
                        else (idx, (xy - self.xy[tid, idx]).norm(dim=1)))
        if offset is not None:
            offset = self.clamp_offset(offset, tid, idx)
        def normal(j):                                             # left-of-travel unit normal at raceline index j
            t_ = self.tan[tid if j.dim() == 1 else tid[:, None].expand_as(j), j]
            return torch.stack([-t_[..., 1], t_[..., 0]], -1)
        ds = self.ds[tid]
        Lp = plan_length(geometry_speed, spec)
        # Fit the raceline ahead; near-term execution is checked against the
        # installed controller by the environment's teacher guard.
        M = 6
        fr = torch.linspace(0.4, 1.0, M, device=dev)
        pidx = (idx[:, None] + ((Lp[:, None] * fr[None]) / ds[:, None]).round().long()) % self.N        # (B,M)
        pts = self.xy[tid[:, None].expand_as(pidx), pidx] - xy[:, None, :]
        if offset is not None:                                     # plan through the offset line, not the raceline
            pts = pts + offset[:, None, None] * normal(pidx)
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        tx = pts[..., 0] * c[:, None] + pts[..., 1] * s_[:, None]
        ty = -pts[..., 0] * s_[:, None] + pts[..., 1] * c[:, None]
        # initial knots: the raceline curvature at the knot arc distances
        kidx = (idx[:, None] + ((Lp[:, None] * torch.linspace(0, 1, N_KNOTS, device=dev)[None]) / ds[:, None]).round().long()) % self.N
        k_rl = self.kappa[tid[:, None].expand_as(kidx), kidx].clone()
        # initial guess = what pure pursuit would do (its lookahead grows with speed, which is exactly the
        # gain scheduling the direct teacher's rejoin has) blended into the raceline's own curvature ahead:
        # the fit below only refines this, so an off-line car neither snaps to the line at full lock
        # (over-correction crashes) nor drifts along beside it (under-correction crashes)
        ld = (self.k_ld * geometry_speed.abs()).clamp(self.ld_min, self.ld_max)
        ld_idx = (idx + (ld / ds).round().long()) % self.N
        tgt = self.xy[tid, ld_idx] - xy
        if offset is not None:
            tgt = tgt + offset[:, None] * normal(ld_idx)
        alpha = torch.atan2(-tgt[:, 0] * s_ + tgt[:, 1] * c, tgt[:, 0] * c + tgt[:, 1] * s_)
        k_pp = (2.0 * torch.sin(alpha) / ld).clamp(-spec.kappa_max, spec.kappa_max)
        w = torch.clamp(1.0 - torch.linspace(0, 1, N_KNOTS, device=dev) * Lp[:, None] / ld[:, None], 0.0, 1.0)   # PP weight fades over the lookahead
        k = w * k_pp[:, None] + (1 - w) * k_rl
        gb = self.grip_bin(P, B, dev)
        wheelbase = torch.full_like(vx, self.L) if P is None else P.get('lf', self.vehicle.lf) + P.get('lr', self.vehicle.lr)
        steering = torch.full_like(vx, min(self.steer_max, self.vehicle.s_max)) if P is None else torch.as_tensor(P.get('s_max', self.vehicle.s_max), device=dev, dtype=vx.dtype).clamp(max=self.steer_max)
        k_lim = 0.85 * torch.minimum(torch.full_like(vx, spec.kappa_max), torch.tan(steering) / wheelbase)
        k = torch.maximum(torch.minimum(k, k_lim[:, None]), -k_lim[:, None])
        # Cached: a list -> CUDA tensor is a pageable host copy, which waits for the device.
        fractions = getattr(self, "_gn_fractions", None)
        if fractions is None or fractions.device != k.device or fractions.dtype != k.dtype:
            fractions = self._gn_fractions = torch.tensor([1., .5, .25, .125, 0.], device=dev, dtype=k.dtype)
        k = fit_knots(k, tx, ty, Lp, fr, k_lim, iters, fractions)
        # speeds from the profile: 0.15 s ahead and at the end of the plan
        v_idx0 = (idx + ((vx.abs() * spec.v_cmd_lead) / ds).round().long()) % self.N
        v_idx1 = (idx + (Lp / ds).round().long()) % self.N
        if offset is not None:                                     # error against the offset line (see __call__)
            t0, p0 = self.tan[tid, idx], self.xy[tid, idx]
            lat_err = (t0[:, 0] * (xy[:, 1] - p0[:, 1]) - t0[:, 1] * (xy[:, 0] - p0[:, 0]) - offset).abs()
        slow = (1.0 - self.lat_slow * lat_err).clamp(0.3, 1.0)     # off the line: slow down, like the direct teacher
        v0 = self.speed_at(tid, v_idx0, gb) * slow; v1 = self.speed_at(tid, v_idx1, gb) * slow
        if self.speed_lead_s is not None:
            # Where the profile rises, ask for the speed it reaches `speed_lead_s` later: the tracker
            # closes a speed error at about half the rate the profile climbs (see the attribute).
            # A max, so a braking zone keeps the speed of the point the car is at.
            v_lead = self.speed_at(tid, (idx + ((vx.abs() * self.speed_lead_s) / ds).round().long()) % self.N, gb) * slow
            v0 = torch.maximum(v0, v_lead)
        if self.speed_horizon_s is not None and spec.speed_mode == "linear":
            # The end speed as the straight line through the profile `speed_horizon_s` ahead -- the
            # part of a plan the tracker follows before the next one replaces it. Through the
            # profile at the plan's END instead, a plan approaching a corner slows linearly over
            # all of its 1.5 s where the profile holds speed and brakes late (see the attribute).
            s_h = (geometry_speed.abs() * self.speed_horizon_s).clamp(min=0.3)
            v_h = self.speed_at(tid, (idx + (s_h / ds).round().long()) % self.N, gb) * slow
            v1 = (v0 + (v_h - v0) * Lp / torch.minimum(s_h, Lp)).clamp(0.0, v_max)
        if self.speed_error_gain is not None:
            # Proportional feedback on the speed error, through the plan, as one shift of the whole
            # speed line: the tracker approaches a reference step at about 2.5 m/s^2 whatever its
            # size, so a car behind the profile is handed a higher line and one ahead of it (a
            # braking zone) a lower one.
            dv = self.speed_error_gain * (v0 - vx.abs())
            v0 = (v0 + dv).clamp(0.0, v_max); v1 = (v1 + dv).clamp(0.0, v_max)
        cap = self.heading_speed_cap(yaw, tid, idx)
        if spec.speed_mode == "knots":                             # the profile itself, at the curvature knots
            vk = self.speed_at(tid[:, None].expand_as(kidx), kidx, gb[:, None].expand_as(kidx)) * slow[:, None]
            return encode_knots(k, vk if cap is None else torch.minimum(vk, cap[:, None]), v_max, spec,
                                v_meas=geometry_speed)
        if spec.speed_mode == "envelope":
            # What the speed dimensions say here is *why* the profile is what it is: the lateral
            # budget this car's grip gives the profile (a_lat * grip, the one number the student
            # cannot see and has to infer from how the car answered it), and the speed to arrive at
            # the end of the plan with, which carries everything past the plan's own curvature.
            # Speed scales (off-line slowdown, speed_scale, the heading cap) enter squared: v ~ sqrt(a).
            scale = self.speed_scale * slow
            if cap is not None:
                scale = scale * (torch.minimum(v0, cap) / v0.clamp_min(1e-3))
                v1 = torch.minimum(v1, cap)
            return encode_envelope(k, self.a_lat * self.grip_levels_t[gb] * scale ** 2, v1, v_max, spec,
                                   v_meas=geometry_speed)
        if cap is not None:
            v0 = torch.minimum(v0, cap); v1 = torch.minimum(v1, cap)
        action = encode(k, v0, v1, v_max, spec, v_meas=geometry_speed)
        if getattr(self, '_defer_profile_projection', False) or not self.certify_speeds:
            return action  # interactive generation certifies after endpoint replacement/scaling
        return self.project_plan_action(action, state, P, v_max, spec, plan_speed=plan_speed)

    #: [s] When set, the plan's end speed is chosen so its linear speed passes through the profile
    #: this far ahead (see `plan_action`). None: through the profile at the plan's end, every run
    #: before 2026-09-24. Measured on ICCAS with friction pinned, the raceline teacher drove 7.97 s
    #: against its own profile's 6.58 s, and the braking zones held most of the gap: the planned speed
    #: sat up to 2.4 m/s under the profile at the car's point through them.
    speed_horizon_s: Optional[float] = None
    #: [s] opt-in, see plan_action. Off: every run before 2026-09-24.
    speed_lead_s: Optional[float] = None
    #: opt-in, see plan_action. Off: every run before 2026-09-24.
    speed_error_gain: Optional[float] = None
    #: Pass the plan's speeds through `project_plan_action`. True: every run before 2026-09-24.
    certify_speeds: bool = True
    # Measured 2026-09-24, line teacher, ICCAS, friction pinned (lap; profile 6.58 s, policy 7.86 s):
    # defaults 7.98; horizon 0.5 s 7.73 (0.6 s); + certify off 7.34; + lead 0.5 s 7.23; error gain
    # 1-3 7.27-7.34; speed_scale 1.1 7.10, 1.2 7.21. With procedural props and random friction the
    # certify-off variants touched props (4/876 and 8/850 encounters against 0/803), so none is on.

    def project_plan_action(self, action, state, P, v_max, spec, *, plan_speed=None):
        """Certify the final fitted path's desired speeds, independently of its source raceline.

        Endpoint interpolation can accelerate through an interior bend even when
        both endpoints came from a valid global profile. This teacher-only check
        preserves feasible actions and tactical zero targets. It does not certify
        reachability from the actual current speed/slip or replace the controller.
        """
        from .mpc import decode, encode
        from .teacher_feasibility import project_speeds
        speed = plan_geometry_speed(state, plan_speed)
        k, length, v0, v1 = decode(action, speed, v_max, torch.full_like(speed, v_max), spec)
        B = state.shape[0]
        def parameter(name):
            default = self.mu_nom if name == 'mu' else getattr(self.vehicle, name)
            value = P.get(name, default) if P is not None else default
            return torch.as_tensor(value, device=state.device, dtype=state.dtype).expand(B)
        codes = self.label_grip_codes
        if codes is None:
            codes = torch.full((B,), LABEL_GRIP_CODE[self.label_grip], device=state.device, dtype=torch.long)
        elif codes.numel() != B:
            if B % codes.numel():
                raise ValueError('grip codes must match the batch or its candidate tiling')
            codes = codes.repeat(B // codes.numel())
        factor = torch.where(codes == LABEL_GRIP_CODE['conservative'],
                             torch.full_like(speed, float(self.grip_levels[0])), torch.ones_like(speed))
        muf = torch.where(codes == LABEL_GRIP_CODE['true'], parameter('mu') * parameter('mu_f_scale'),
                          factor * self.mu_nom * self.vehicle.mu_f_scale)
        mur = torch.where(codes == LABEL_GRIP_CODE['true'], parameter('mu') * parameter('mu_r_scale'),
                          factor * self.mu_nom * self.vehicle.mu_r_scale)
        out0, out1, scale = project_speeds(k, length, v0, v1, P, self.vehicle, spec, muf, mur)
        self.last_profile_scale = scale
        self.last_profile_initial_overspeed = state[:, 3].abs() > out0 + 1e-6
        self.last_profile_initial_sideslip = torch.atan2(state[:, 4], state[:, 3].abs().clamp_min(1e-6))
        projected = encode(k, out0, out1, v_max, spec, v_meas=speed)
        return torch.where((scale == 1)[:, None], action, projected)

    label_grip = "true"          # "true": per-env grip (privileged); "nominal"/"conservative": constant

    #: (B,) per-car override of `label_grip`, as `LABEL_GRIP_CODE` values, or None for the scalar
    #: above. A race whose opponents were configured one by one (`f1sim.opponent_slots`) can put a
    #: teacher planning on the true friction next to one planning on the nominal profile, which is a
    #: fast car beside a repeatable one -- so the label the profile is chosen by became a property of
    #: the *car* rather than of the teacher. None leaves `grip_bin` the function it was.
    label_grip_codes: Optional[torch.Tensor] = None

    def heading_speed_cap(self, yaw: torch.Tensor, tid: torch.Tensor, idx: torch.Tensor) -> Optional[torch.Tensor]:
        """Speed from which the car can still turn back onto the lane within `recover_time`.

        Heading change available over a time t at the grip-limited curvature a_lat / v^2 is
        a_lat * t / v, so recovering a heading error psi needs v <= a_lat * t / psi. Returns None
        when disabled, leaving the commanded speed exactly as before."""
        if self.recover_time <= 0:
            return None
        tan = self.tan[tid, idx]
        psi = torch.remainder(torch.atan2(tan[:, 1], tan[:, 0]) - yaw + math.pi, 2 * math.pi) - math.pi
        return (self.a_lat_recover * self.recover_time / psi.abs().clamp_min(1e-3)).clamp_min(self.v_recover_min)

    def grip_bin(self, P, B: int, device):
        """Index of the speed profile this teacher drives on.

        "true" reads each env's randomized mu. That makes the teacher fast, but it also makes the
        *label* a function of something the student cannot see: two identical scans get speed labels
        up to 1/0.45 ~ 2.2x apart, and a Huber regression can only learn their conditional mean --
        too fast in low grip, too slow in high grip. "nominal" and "conservative" pin the profile so
        the label is a function of the observation alone (the price is a slower target, and a teacher
        that can over-drive a low-grip car, which is why collection uses `speed_scale` < 1).
        """
        if self.label_grip_codes is not None:
            return self._grip_bin_per_car(P, B, device)
        if self.label_grip == "nominal" or P is None:
            return torch.full((B,), self.nominal_grip_index, dtype=torch.long, device=device)
        if self.label_grip == "conservative":
            return torch.zeros(B, dtype=torch.long, device=device)
        g = ((P["mu"] * P["mu_f_scale"]) / (self.mu_nom * self.mu_f_nom)).clamp(max=float(self.grip_levels[-1]))
        # A nearest-bin lookup can select a profile with more grip than this car
        # actually has. Use the lower envelope, including at the high-grip bins.
        return (torch.searchsorted(self.grip_levels_t, g, right=True) - 1).clamp(0, len(self.grip_levels) - 1)

    def _grip_bin_per_car(self, P, B: int, device) -> torch.Tensor:
        """`grip_bin` when each car carries its own label. All three answers, then select.

        Computed rather than branched because the rows are mixed: there is no "the" mode to test.
        Without `P` the privileged answer does not exist for anybody, which is exactly the case the
        scalar path already turns into `nominal`, so it does the same here."""
        top = self.nominal_grip_index
        codes = self.label_grip_codes
        nominal = torch.full((B,), top, dtype=torch.long, device=device)
        if P is None:
            return nominal
        g = ((P["mu"] * P["mu_f_scale"]) / (self.mu_nom * self.mu_f_nom)).clamp(max=float(self.grip_levels[-1]))
        true_bin = (torch.searchsorted(self.grip_levels_t, g, right=True) - 1).clamp(0, len(self.grip_levels) - 1)
        out = torch.where(codes == LABEL_GRIP_CODE["nominal"], nominal, true_bin)
        return torch.where(codes == LABEL_GRIP_CODE["conservative"],
                           torch.zeros_like(out), out)

    speed_mode = "grip"          # "grip": per-grip profiles (braking points move too); "sqrt": nominal profile x sqrt(grip)

    def speed_at(self, tid: torch.Tensor, idx: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
        if self.speed_mode == "sqrt":
            g = self.grip_levels_t[gb]
            return self.v[tid, idx] * torch.sqrt(g) * self.speed_scale
        return self.v_grip[tid, gb, idx] * self.speed_scale

    def clamp_offset(self, offset: torch.Tensor, tid: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Cut a commanded lateral offset down to what the lane has room for at this raceline point."""
        if self.offset_limit is None:
            return offset
        lim = self.offset_limit[tid, idx]
        return torch.clamp(offset, -lim, lim)

    def project(self, xy: torch.Tensor, tid: Optional[torch.Tensor] = None):
        tid = torch.zeros(xy.shape[0], dtype=torch.long, device=xy.device) if tid is None else tid
        d2 = ((xy[:, None, :] - self.xy[tid]) ** 2).sum(-1)
        idx = d2.argmin(1)
        return idx, d2[torch.arange(len(idx), device=idx.device), idx].sqrt()

    def __call__(self, state: torch.Tensor, P=None, tid: Optional[torch.Tensor] = None,
                 offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        """state (B,8) ground truth -> action (B,2) = (steer [rad], speed [m/s]).
        P: the simulator's per-env parameter dict (privileged). When given, the teacher
        compensates command latency, servo lag, actuator calibration and scales speed with grip.
        tid: per-env track id (which raceline to follow).
        offset: (B,) metres left of the raceline to track instead of the line itself, clamped by
        `offset_limit`. None (every caller that does not script opponent behaviour) is the path this
        function had before offsets existed, instruction for instruction."""
        xy, yaw, vx = state[:, :2], state[:, 2], state[:, 3]
        tid = torch.zeros(xy.shape[0], dtype=torch.long, device=xy.device) if tid is None else tid
        if P is not None:
            # latency compensation: predict the pose at the time the command takes effect
            dt_c = P["cmd_delay"] + 0.5 * P["servo_tau"]
            r = state[:, 5]
            yaw_c = yaw + r * dt_c
            xy = xy + torch.stack([vx * torch.cos(yaw_c), vx * torch.sin(yaw_c)], 1) * dt_c[:, None]
            yaw = yaw_c
        idx, lat_err = self.project(xy, tid)
        if offset is not None:
            # The offset line is the raceline displaced along its own normal, so the car's error
            # against it is its signed error against the raceline minus the offset -- and the
            # off-line speed slowdown below has to read *that*, or a car sitting perfectly on a
            # 0.35 m offset would be told it is 0.35 m off and slowed for it every step it holds.
            offset = self.clamp_offset(offset, tid, idx)
            t0, p0 = self.tan[tid, idx], self.xy[tid, idx]
            lat_err = (t0[:, 0] * (xy[:, 1] - p0[:, 1]) - t0[:, 1] * (xy[:, 0] - p0[:, 0]) - offset).abs()
        ds = self.ds[tid]
        if self.mode == "pp":
            ld = (self.k_ld * vx.abs()).clamp(self.ld_min, self.ld_max)
            tgt_idx = (idx + (ld / ds).round().long()) % self.N
            tgt = self.xy[tid, tgt_idx]
            if offset is not None:
                tn = self.tan[tid, tgt_idx]
                tgt = tgt + offset[:, None] * torch.stack([-tn[:, 1], tn[:, 0]], 1)   # normal, left +
            dx, dy = tgt[:, 0] - xy[:, 0], tgt[:, 1] - xy[:, 1]
            alpha = torch.atan2(dy, dx) - yaw
            alpha = torch.remainder(alpha + math.pi, 2 * math.pi) - math.pi
            ld_act = torch.sqrt(dx ** 2 + dy ** 2).clamp_min(1e-3)
            L_eff = self.L + self.k_us * vx ** 2                                             # understeer
            steer = torch.atan(2 * L_eff * torch.sin(alpha) / ld_act)
            if self.k_e_pp > 0:                                                             # small lateral-error term
                t = self.tan[tid, idx]; p = self.xy[tid, idx]
                e = t[:, 0] * (xy[:, 1] - p[:, 1]) - t[:, 1] * (xy[:, 0] - p[:, 0])
                e = e if offset is None else e - offset
                steer = steer - torch.atan(self.k_e_pp * e / (vx.abs() + self.v_soft))
        else:
            front = xy + self.L * torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
            fidx, _ = self.project(front, tid)
            t = self.tan[tid, fidx]; p = self.xy[tid, fidx]
            e = t[:, 0] * (front[:, 1] - p[:, 1]) - t[:, 1] * (front[:, 0] - p[:, 0])      # left of the line +
            e = e if offset is None else e - offset          # error against the offset line, same normal
            psi = torch.atan2(t[:, 1], t[:, 0]) - yaw
            psi = torch.remainder(psi + math.pi, 2 * math.pi) - math.pi
            k_idx = (fidx + ((vx.abs() * self.ff_time) / ds).round().long()) % self.N
            kap = self.kappa[tid, k_idx]
            ff = torch.atan(kap * (self.L + self.k_us * vx ** 2))                          # + understeer
            steer = ff + self.k_psi * psi - torch.atan(self.k_e * e / (vx.abs() + self.v_soft))
        steer = steer.clamp(-self.steer_max, self.steer_max)
        v_idx = (idx + ((vx.abs() * self.t_v) / ds).round().long()) % self.N
        v_cmd = self.speed_at(tid, v_idx, self.grip_bin(P, xy.shape[0], xy.device))
        v_cmd = v_cmd * (1.0 - self.lat_slow * lat_err).clamp(0.3, 1.0)   # slow down when off-line
        cap = self.heading_speed_cap(yaw, tid, idx)
        if cap is not None:
            v_cmd = torch.minimum(v_cmd, cap)
        if P is not None:
            v_cmd = v_cmd / P["speed_gain"]
            steer = ((steer - P["steer_bias"]) / P["steer_gain"]).clamp(-self.steer_max, self.steer_max)
        return torch.stack([steer, v_cmd], 1)
