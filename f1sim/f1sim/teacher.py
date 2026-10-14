"""Privileged teacher: pure pursuit on the optimized raceline + speed profile (batched torch).
Uses ground-truth state; only valid inside the simulator. Serves as IL teacher and as the
baseline the RL student must beat."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import torch

import numpy as np

from .raceline import Raceline, curvature
from .track import resample_closed

#: `label_grip` as an integer, for the per-car form. The order is the one `f1sim.opponent_slots`
#: lists, and it is frozen: a recorded code keeps its meaning.
LABEL_GRIP_NAMES = ("true", "nominal", "conservative")
LABEL_GRIP_CODE = {name: i for i, name in enumerate(LABEL_GRIP_NAMES)}


@dataclass
class LaneHold:
    """A lateral position held in the lane's own coordinates: the rolling start's hold
    (`gym_env.EnvConfig.spawn_alongside_hold`), installed on the teacher that drives the opponents.

    A car with weight `w` > 0 is steered, at every point the teacher plans through -- each plan
    point, the pure-pursuit target, the error its slowdown reads -- toward the blend
    `w * (the held point) + (1 - w) * (the raceline offset point it was asked for)`. The held point
    is the one at centreline offset `cl` at the same distance ahead along the CENTRELINE from the
    car's own station, so the held line is the lane position itself and not one raceline offset (a
    constant raceline offset follows the raceline across the lane: a car moving sideways while it
    is meant to be still). Everything a caller adds to the offset -- the events' shifts, a planner
    kind's own choice of side -- is inside the second term, so the hold scales all of it by `1 - w`
    without any of them knowing, and at `w` = 0 the plan is the planner's own, bit for bit.

    Measured why (2026-10-14, boot-camp config, 228 abreast opponents): the first version held one
    raceline offset fitted at the pure-pursuit lookahead and summed it with the planners' offsets.
    The lane clamp cut that offset at the car's own raceline point (`RacelineTeacher.clamp_offset`),
    where the room is measured from a different raceline position -- the 83 cars cut by more than
    5 cm moved 0.21-0.26 m toward the learner in 1 s (median, p90 0.59) -- and a planner's choice
    rode on top of the held offset (`OffsetPlanner._merge_offset`). Here the held offset is cut
    where it is applied, against the lane's own room at that station (`room`, the same test the
    abreast grid was placed with), and nothing is summed with it. The raceline is not the frame
    either: a raceline point's foot on the centreline jumps across a sharp corner (the line cuts
    inside the bend's centre), so the held line is sampled from the car's centreline station.

    A lane position is held at the speed it allows (`RacelineTeacher._hold_speed_cap`): the
    raceline's profile is for the raceline's radius, and the inside of a bend at a fixed offset is a
    tighter one. Driven at the profile, a car held on the inside slides out across the lane -- into
    the car beside it. Measured (2026-10-14, blackbox2022_1): a car held 0.36 m inside a 75-degree,
    0.8 m-radius bend at 2.9 m/s accelerated with the profile to 3.6 m/s and slid 0.6 m out in
    0.6 s at full lock. Where the held line is tighter than the car can steer at all it is driven
    at full lock: the hold can delay a merge, not repeal the geometry.

    The held line is followed by its own arc length: the plan's path reaches its points at
    fractions of the plan length, and on the inside of a bend the held line is shorter than the
    centreline by the factor 1 - c kappa (16 % at 0.35 m inside a 2.3 m radius) -- points placed by
    centreline arc there left a fit 0.5 m off them, the car 0.13 m off its lane in 0.5 s.

    Per car (written in place by the env every step of a hold): `w` the weight, `cl` the centreline
    offset held (left +), `ci` the car's centreline index (`Simulator.cl_idx`). Per station of the
    track's resampled centreline (`TrackTensors.cl`): `cxy` (T, n, 2) the points, `ctan` (T, n, 2)
    the unit tangents, `seg` (T,) the spacing, `room` (T, n, 2) the free lateral room left (+) /
    right (magnitude) of the centreline for the car's body and margin, `kap` (T, n) the centreline's
    curvature (left +)."""
    w: torch.Tensor
    cl: torch.Tensor
    ci: torch.Tensor
    cxy: torch.Tensor
    ctan: torch.Tensor
    seg: torch.Tensor
    room: torch.Tensor
    kap: torch.Tensor
    #: The held line is walked ahead of each car in `step` centreline points a sample, `n_steps`
    #: samples (`RacelineTeacher._hold_line`): fixed per track set, so every call has one shape.
    step: int = 1
    n_steps: int = 120

    def rows(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "LaneHold":
        """The same hold with its per-car tensors passed through `f` -- a cut to the rows a call
        plans for, or a tile over a candidate family; the per-station tables are shared."""
        return LaneHold(f(self.w), f(self.cl), f(self.ci), self.cxy, self.ctan, self.seg, self.room, self.kap,
                        self.step, self.n_steps)


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
    #: behaviour events are on (f1sim.opponent_events.raceline_offset_limit_sided: (T, N, 2) left / right
    #: room; a (T, N) symmetric bound is also accepted). None: no clamp, which
    #: is also what every caller that never passes `offset` sees.
    offset_limit: Optional[torch.Tensor] = None

    #: The rolling start's lane hold (`LaneHold`), installed by the env only where one can happen
    #: (`F1VecEnv.set_teacher`). It acts on a call that passes an `offset` -- the opponent command
    #: always does there -- of the batch it was built for; a call without one, or of another size (a
    #: label for some rows, a candidate tile), plans as if it were None. None: every call exactly as
    #: before it existed.
    lane_hold: Optional[LaneHold] = None

    def _hold_for(self, offset: Optional[torch.Tensor], B: int) -> Optional[LaneHold]:
        h = self.lane_hold
        if h is None or offset is None or h.w.shape[0] != B:
            return None
        return h

    #: [m] how far ahead a held car looks for the tightest bend of the lane position it holds, and
    #: how finely. Braking from 6 m/s to 2.5 m/s at the profile's 4 m/s^2 takes 3.7 m; a held car
    #: spawns at up to 3 m/s and the hold is 1 s.
    HOLD_PREVIEW_M = 6.0
    HOLD_PREVIEW_STEP_M = 0.25
    #: [m] how far the car may be from its centreline index's vertex, along the lane, before the
    #: station is read as that far and no further (the index is the last physics step's, the pose a
    #: latency-compensated one).
    HOLD_STATION_SLACK_M = 1.0

    def _hold_station(self, h: LaneHold, tid: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """(B,) each car's station on the centreline, as a fractional index of `h.cxy`."""
        ci = h.ci
        along = ((xy - h.cxy[tid, ci]) * h.ctan[tid, ci]).sum(-1)
        slack = self.HOLD_STATION_SLACK_M
        return ci.to(xy.dtype) + along.clamp(-slack, slack) / h.seg[tid]

    def _hold_at(self, h: LaneHold, tid: torch.Tensor, f: torch.Tensor):
        """(held point (..., 2), the held offset c (...), the centreline index (...), tid matching)
        at fractional centreline index `f` (B,) or (B, M). c is `h.cl` cut to the room there."""
        n = h.cxy.shape[1]
        tj = tid if f.dim() == 1 else tid[:, None].expand_as(f)
        fl = torch.floor(f)
        i = torch.remainder(fl.long(), n)
        t = h.ctan[tj, i]
        p = h.cxy[tj, i] + ((f - fl) * h.seg[tj])[..., None] * t
        cl = h.cl if f.dim() == 1 else h.cl[:, None]
        room = h.room[tj, i]
        c = torch.maximum(torch.minimum(cl, room[..., 0]), -room[..., 1])
        return p + c[..., None] * torch.stack([-t[..., 1], t[..., 0]], -1), c, i, tj

    def _hold_kappa(self, h: LaneHold, tid: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Signed curvature (left +) of the held lane position at fractional centreline index `f`:
        the centreline's carried to the held offset c as a parallel curve, kappa / (1 - c kappa), and
        no tighter than the car can steer -- past that, or past the bend's centre, it is a full-lock
        bend, which is what the car will be driving."""
        _, c, i, tj = self._hold_at(h, tid, f)
        kc = h.kap[tj, i]
        den = 1.0 - c * kc
        k_car = math.tan(self.steer_max) / self.L
        k = torch.where(den > 1e-3, kc / den.clamp_min(1e-3), torch.sign(kc) * k_car)
        return k.clamp(-k_car, k_car)

    def _hold_line(self, h: LaneHold, tid: torch.Tensor, xy: torch.Tensor):
        """The held line ahead of each car: (points (B, K+1, 2), their arc along it from the first
        (B, K+1), their fractional centreline index (B, K+1)) for K = `h.n_steps` samples
        `h.step` centreline points apart, starting at the car's own station."""
        f0 = self._hold_station(h, tid, xy)
        k = torch.arange(h.n_steps + 1, device=xy.device, dtype=f0.dtype) * h.step
        f = f0[:, None] + k[None]
        pts = self._hold_at(h, tid, f)[0]
        arc = torch.cat([torch.zeros_like(f[:, :1]), (pts[:, 1:] - pts[:, :-1]).norm(dim=-1).cumsum(1)], 1)
        return pts, arc, f

    @staticmethod
    def _hold_sample(line, d: torch.Tensor):
        """(points (B, M, 2), fractional centreline indices (B, M)) of the held line at arc `d`
        (B, M) along it from the car, linearly between its samples; past the walk, its last one."""
        pts, arc, f = line
        K = arc.shape[1] - 1
        j = torch.searchsorted(arc, d.contiguous()).clamp(1, K)
        a0, a1 = arc.gather(1, j - 1), arc.gather(1, j)
        u = ((d - a0) / (a1 - a0).clamp_min(1e-6)).clamp(0.0, 1.0)
        p0 = pts.gather(1, (j - 1)[..., None].expand(*j.shape, 2))
        p1 = pts.gather(1, j[..., None].expand(*j.shape, 2))
        f0, f1 = f.gather(1, j - 1), f.gather(1, j)
        return p0 + u[..., None] * (p1 - p0), f0 + u * (f1 - f0)

    def _hold_speed_cap(self, h: LaneHold, tid: torch.Tensor, line, gb: torch.Tensor) -> torch.Tensor:
        """(B,) the fastest a held car may go now, +inf for a car that is not held: the speed from
        which it can still brake (at the profile's `a_brake`) to what each bend of its held lane
        position allows over the next `HOLD_PREVIEW_M` (the profile's own lateral budget, `a_lat`
        x grip). Divided by sqrt(w), so it fades out with the hold over the ramp."""
        arc = line[1]
        d = getattr(self, "_hold_preview", None)
        if d is None or d.device != arc.device or d.dtype != arc.dtype:
            n = int(round(self.HOLD_PREVIEW_M / self.HOLD_PREVIEW_STEP_M)) + 1
            d = self._hold_preview = torch.arange(n, device=arc.device, dtype=arc.dtype) * self.HOLD_PREVIEW_STEP_M
        f = self._hold_sample(line, d[None].expand(arc.shape[0], -1))[1]
        k = self._hold_kappa(h, tid, f).abs().clamp_min(1e-3)
        a = (self.a_lat * self.grip_levels_t[gb])[:, None]
        v = torch.sqrt(a / k + 2.0 * self.a_brake * d[None]).amin(1)
        return torch.where(h.w > 0, v / h.w.clamp_min(1e-6).sqrt(), torch.full_like(v, math.inf))

    def _hold_cap_fn(self, h: Optional[LaneHold], tid: torch.Tensor, line, gb: torch.Tensor):
        """`v -> v` capped by `_hold_speed_cap` on the held rows, untouched (bit for bit) on the
        others and everywhere when there is no hold. `v` is (B,) or (B, K)."""
        if h is None:
            return lambda v: v
        cap = self._hold_speed_cap(h, tid, line, gb)
        held = h.w > 0

        def apply(v):
            if v.dim() == 2:
                return torch.where(held[:, None], torch.minimum(v, cap[:, None]), v)
            return torch.where(held, torch.minimum(v, cap), v)
        return apply

    def _hold_error(self, h: LaneHold, tid: torch.Tensor, held: torch.Tensor, j: torch.Tensor,
                    offset: torch.Tensor) -> torch.Tensor:
        """(B,) how far the blended target beside the car (the held point `held` (B, 2), raceline
        index `j`) sits from the offset point there, along the raceline normal at `j`: what a lateral
        error measured against the offset line is off by."""
        tn = self.tan[tid, j]
        n_rl = torch.stack([-tn[:, 1], tn[:, 0]], 1)
        off_pt = self.xy[tid, j] + offset[:, None] * n_rl
        return h.w * ((held - off_pt) * n_rl).sum(1)

    def __init__(self, raceline, wheelbase: float = 0.3302, device="cpu", mode: str = "pp",
                 lookahead_gain: float = 0.35, lookahead_min: float = 0.6, lookahead_max: float = 2.5,
                 speed_lookahead_time: float = 0.35, lateral_slowdown: float = 0.6,
                 speed_scale: float = 1.0, steer_max: float = 0.4189,
                 k_e: float = 2.5, k_psi: float = 1.0, v_soft: float = 1.0, k_us: float = 0.02,
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
        from .mpc import (N_ACC, N_KNOTS, PlanSpec, encode, encode_budget, encode_envelope, encode_knots, encode_timed,
                          encode_tspeed, path_points, plan_length)
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
        hold = self._hold_for(offset, B)
        if hold is not None:
            line = self._hold_line(hold, tid, xy)                  # the held lane position ahead of each car
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
            if hold is not None:
                # Each plan point blended toward the held lane position the same distance ahead along
                # the centreline (`LaneHold`). A `where` and not a plain sum, so a car with no hold
                # plans bit for bit as before.
                held = self._hold_sample(line, Lp[:, None] * fr[None])[0]
                ww = hold.w[:, None, None]
                pts = torch.where(ww > 0, pts + ww * ((held - xy[:, None, :]) - pts), pts)
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        tx = pts[..., 0] * c[:, None] + pts[..., 1] * s_[:, None]
        ty = -pts[..., 0] * s_[:, None] + pts[..., 1] * c[:, None]
        # initial knots: the raceline curvature at the knot arc distances
        kidx = (idx[:, None] + ((Lp[:, None] * torch.linspace(0, 1, N_KNOTS, device=dev)[None]) / ds[:, None]).round().long()) % self.N
        k_rl = self.kappa[tid[:, None].expand_as(kidx), kidx].clone()
        if hold is not None:
            # a held car's own bend, not the raceline's, is where its fit starts (and what the ridge
            # in `fit_knots` pulls towards)
            k_h = self._hold_kappa(hold, tid, self._hold_sample(
                line, Lp[:, None] * torch.linspace(0, 1, N_KNOTS, device=dev)[None])[1])
            k_rl = torch.where(hold.w[:, None] > 0, k_rl + hold.w[:, None] * (k_h - k_rl), k_rl)
        # initial guess = what pure pursuit would do (its lookahead grows with speed, which is exactly the
        # gain scheduling the direct teacher's rejoin has) blended into the raceline's own curvature ahead:
        # the fit below only refines this, so an off-line car neither snaps to the line at full lock
        # (over-correction crashes) nor drifts along beside it (under-correction crashes)
        ld = (self.k_ld * geometry_speed.abs()).clamp(self.ld_min, self.ld_max)
        ld_idx = (idx + (ld / ds).round().long()) % self.N
        tgt = self.xy[tid, ld_idx] - xy
        if offset is not None:
            tgt = tgt + offset[:, None] * normal(ld_idx)
            if hold is not None:
                held = self._hold_sample(line, ld[:, None])[0][:, 0]
                tgt = torch.where(hold.w[:, None] > 0, tgt + hold.w[:, None] * ((held - xy) - tgt), tgt)
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
            e_off = t0[:, 0] * (xy[:, 1] - p0[:, 1]) - t0[:, 1] * (xy[:, 0] - p0[:, 0]) - offset
            lat_err = e_off.abs()
            if hold is not None:
                # ... and against the held position where a hold applies: a car sitting on the lane
                # position it is told to hold is not off its line, and slowed for it it would drop
                # back out of the abreast row it is holding.
                lat_err = torch.where(hold.w > 0, (e_off - self._hold_error(hold, tid, line[0][:, 0], idx, offset)).abs(),
                                      lat_err)
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
        hc = self._hold_cap_fn(hold, tid, line if hold is not None else None, gb)   # a held car: what its lane position allows
        if spec.speed_mode == "knots":                             # the profile itself, at the curvature knots
            vk = self.speed_at(tid[:, None].expand_as(kidx), kidx, gb[:, None].expand_as(kidx)) * slow[:, None]
            return encode_knots(k, hc(vk if cap is None else torch.minimum(vk, cap[:, None])), v_max, spec,
                                v_meas=geometry_speed)
        if spec.speed_mode == "budget":
            # The grip budget is the floor's and this teacher's global pace only; every reason to go
            # slower than it (off the line, a heading the cap distrusts) is a ceiling on the profile.
            a_hat = self.a_lat * self.grip_levels_t[gb] * self.speed_scale ** 2
            # The ceiling is the fastest this teacher wants to be anywhere along the plan (its
            # profile there, off-line slowdown included), not its speed right now: a ceiling at the
            # current speed would pin the profile under it and delay every exit by a step.
            span = ((v_idx1 - v_idx0) % self.N)[:, None]
            frac = torch.linspace(0.0, 1.0, 9, device=span.device)[None]
            i_s = (v_idx0[:, None] + (span * frac).round().long()) % self.N
            v_top = (self.speed_at(tid[:, None].expand_as(i_s), i_s, gb[:, None].expand_as(i_s)).amax(1)) * slow
            v_cap = hc(v_top if cap is None else torch.minimum(v_top, cap))
            v_end = hc(v1 if cap is None else torch.minimum(v1, cap))
            return encode_budget(k, a_hat.expand_as(v0) if torch.is_tensor(a_hat) else torch.full_like(v0, a_hat),
                                 v_end, v_cap, v_max, spec, v_meas=geometry_speed)
        if spec.speed_mode in ("timed", "tspeed"):
            # Drive this teacher's own profile (off-line slowdown and the heading cap included) forward
            # in time from the speed the car has: each step asks for the speed the profile holds where
            # the car will be one step on, within the car's drive limit and the profile's braking.
            # The profile already brakes for what is ahead, so the acceleration it implies is the
            # label; nothing about the plan's end enters.
            a_hat = self.a_lat * self.grip_levels_t[gb] * self.speed_scale ** 2
            a_hat = a_hat.expand_as(v0) if torch.is_tensor(a_hat) else torch.full_like(v0, a_hat)
            M = spec.n_time; dt = spec.horizon_s / M
            v = geometry_speed.abs().clone(); s_arc = torch.zeros_like(v)
            accs, vs = [], [v]
            for _ in range(M):
                ahead = (idx + ((s_arc + v * dt) / ds).round().long()) % self.N
                tgt = self.speed_at(tid, ahead, gb) * slow
                if cap is not None:
                    tgt = torch.minimum(tgt, cap)
                tgt = hc(tgt)
                drive = spec.a_drive_profile * (spec.v_switch_profile / v.clamp_min(1e-3)).clamp(max=1.0)
                acc = ((tgt - v) / dt).clamp(min=-self.a_brake)
                acc = torch.minimum(acc, drive)
                accs.append(acc)
                v_new = (v + acc * dt).clamp_min(0.0)
                s_arc = s_arc + 0.5 * (v + v_new) * dt
                v = v_new
                vs.append(v)
            if spec.speed_mode == "tspeed":
                V = torch.stack(vs, 1)                                      # (B, M+1) at t = 0, dt, ..
                cols = []
                for j in range(N_ACC):
                    q = (j + 1) / N_ACC * M; i0 = min(int(q), M - 1); w = q - i0
                    cols.append(V[:, i0] * (1 - w) + V[:, i0 + 1] * w)
                return encode_tspeed(k, torch.stack(cols, 1), a_hat, v_max, spec, v_meas=geometry_speed)
            A = torch.stack(accs, 1)                                        # (B, M)
            at = [min(int(round(j / (N_ACC - 1) * M)), M - 1) for j in range(N_ACC)]
            return encode_timed(k, A[:, at], a_hat, spec, v_meas=geometry_speed)
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
            if hold is not None:
                v0h = v0 if cap is None else torch.minimum(v0, cap)
                scale = torch.where(hold.w > 0, scale * (hc(v0h) / v0h.clamp_min(1e-3)), scale)
                v1 = hc(v1)
            return encode_envelope(k, self.a_lat * self.grip_levels_t[gb] * scale ** 2, v1, v_max, spec,
                                   v_meas=geometry_speed)
        if cap is not None:
            v0 = torch.minimum(v0, cap); v1 = torch.minimum(v1, cap)
        v0, v1 = hc(v0), hc(v1)
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
        if spec.speed_mode != "linear":
            # A profile mode's speeds are built from the plan's own curvature, the grip budget and
            # the car's drive / brake limits (`mpc.decode_profile`): feasible by construction, and
            # its dimensions are not the two linear speeds this check certifies.
            return action
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
        v = self.v_grip[tid, gb, idx]
        sc = self.speed_scale
        if torch.is_tensor(sc) and sc.ndim == 1 and v.ndim == 2:
            sc = sc[:, None]                      # a per-car scale over several lookups per car
        return v * sc

    def clamp_offset(self, offset: torch.Tensor, tid: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Cut a commanded lateral offset down to what the lane has room for at this raceline point.
        `offset_limit` (T, N) is one bound for both sides; (T, N, 2) is (left, right) room."""
        if self.offset_limit is None:
            return offset
        lim = self.offset_limit[tid, idx]
        if lim.ndim == offset.ndim + 1:
            return torch.maximum(torch.minimum(offset, lim[..., 0]), -lim[..., 1])
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
        hold, hold_e = None, None
        if offset is not None:
            # The offset line is the raceline displaced along its own normal, so the car's error
            # against it is its signed error against the raceline minus the offset -- and the
            # off-line speed slowdown below has to read *that*, or a car sitting perfectly on a
            # 0.35 m offset would be told it is 0.35 m off and slowed for it every step it holds.
            offset = self.clamp_offset(offset, tid, idx)
            hold = self._hold_for(offset, xy.shape[0])
            t0, p0 = self.tan[tid, idx], self.xy[tid, idx]
            e_off = t0[:, 0] * (xy[:, 1] - p0[:, 1]) - t0[:, 1] * (xy[:, 0] - p0[:, 0]) - offset
            lat_err = e_off.abs()
            if hold is not None:
                # the same against the held lane position (`LaneHold`), where a hold applies
                line = self._hold_line(hold, tid, xy)
                hold_e = self._hold_error(hold, tid, line[0][:, 0], idx, offset)
                lat_err = torch.where(hold.w > 0, (e_off - hold_e).abs(), lat_err)
        ds = self.ds[tid]
        if self.mode == "pp":
            ld = (self.k_ld * vx.abs()).clamp(self.ld_min, self.ld_max)
            tgt_idx = (idx + (ld / ds).round().long()) % self.N
            tgt = self.xy[tid, tgt_idx]
            if offset is not None:
                tn = self.tan[tid, tgt_idx]
                tgt = tgt + offset[:, None] * torch.stack([-tn[:, 1], tn[:, 0]], 1)   # normal, left +
                if hold is not None:
                    held = self._hold_sample(line, ld[:, None])[0][:, 0]
                    tgt = torch.where(hold.w[:, None] > 0, tgt + hold.w[:, None] * (held - tgt), tgt)
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
                if hold is not None:
                    e = torch.where(hold.w > 0, e - hold_e, e)
                steer = steer - torch.atan(self.k_e_pp * e / (vx.abs() + self.v_soft))
        else:
            front = xy + self.L * torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
            fidx, _ = self.project(front, tid)
            t = self.tan[tid, fidx]; p = self.xy[tid, fidx]
            e = t[:, 0] * (front[:, 1] - p[:, 1]) - t[:, 1] * (front[:, 0] - p[:, 0])      # left of the line +
            e = e if offset is None else e - offset          # error against the offset line, same normal
            if hold is not None:
                front_held = self._hold_sample(line, torch.full_like(vx, self.L)[:, None])[0][:, 0]
                e = torch.where(hold.w > 0, e - self._hold_error(hold, tid, front_held, fidx, offset), e)
            psi = torch.atan2(t[:, 1], t[:, 0]) - yaw
            psi = torch.remainder(psi + math.pi, 2 * math.pi) - math.pi
            k_idx = (fidx + ((vx.abs() * self.ff_time) / ds).round().long()) % self.N
            kap = self.kappa[tid, k_idx]
            ff = torch.atan(kap * (self.L + self.k_us * vx ** 2))                          # + understeer
            steer = ff + self.k_psi * psi - torch.atan(self.k_e * e / (vx.abs() + self.v_soft))
        steer = steer.clamp(-self.steer_max, self.steer_max)
        v_idx = (idx + ((vx.abs() * self.t_v) / ds).round().long()) % self.N
        gb = self.grip_bin(P, xy.shape[0], xy.device)
        v_cmd = self.speed_at(tid, v_idx, gb)
        v_cmd = v_cmd * (1.0 - self.lat_slow * lat_err).clamp(0.3, 1.0)   # slow down when off-line
        cap = self.heading_speed_cap(yaw, tid, idx)
        if cap is not None:
            v_cmd = torch.minimum(v_cmd, cap)
        v_cmd = self._hold_cap_fn(hold, tid, line if hold is not None else None, gb)(v_cmd)   # a held car: what its lane position allows
        if P is not None:
            v_cmd = v_cmd / P["speed_gain"]
            steer = ((steer - P["steer_bias"]) / P["steer_gain"]).clamp(-self.steer_max, self.steer_max)
        return torch.stack([steer, v_cmd], 1)
