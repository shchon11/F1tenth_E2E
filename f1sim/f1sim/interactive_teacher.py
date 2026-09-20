"""A privileged teacher that races the other car instead of ignoring it.

`RacelineTeacher` follows a line computed once from the occupancy grid. It has never heard of the
other cars: in a race it is kept off them by the env's follow-gap cap, which only ever slows it
down. A student distilled from it can therefore learn "hold the line, and brake for what is in
front of you now" and nothing else -- which is exactly the memory note *DAgger cannot teach
overtaking*, and why every pass the policies make has had to come out of PPO.

This teacher is the other half of that sentence. It keeps the raceline teacher as its reference --
the line, the grip-aware speed profile, the latency compensation, the plan fit -- and puts a small
**best-response search** on top of it:

    1. a family of candidate plans in the SAME 8-D plan space the student outputs
       (`f1sim.mpc`): `offsets` x `speeds` around the raceline teacher's own plan;
    2. each rolled out to a time-indexed trajectory by `mpc.reference`, i.e. by the geometry the
       plan tracker walks as a target (not a dynamically simulated execution);
    3. each scored against the walls, the lane margin, the raceline's own progress, and -- the
       point of the whole exercise -- the opponents' **future** positions, matched sample for
       sample in time (`gym_env.F1VecEnv.car_future`);
    4. argmin is the label.

Why time-indexed matters, and why a present-tense cost cannot do it: the three situations the
contract names are all invisible at t = 0. A car drifting right is, right now, straight ahead; a
car braking is, right now, at a comfortable distance; a car about to move into the gap is, right
now, leaving it open. `tests/test_interactive_teacher.py` builds those three as synthetic scenes
and asserts which candidate wins.

Everything is batched over B envs and over the candidates at once -- one `plan_action` call on a
(K_off * B) tile and one scoring pass -- because the whole point is to be able to run it inside
DAgger's collection loop. `work/bench` measures it against `RacelineTeacher`.

**Privileged, simulator-only.** It reads the other cars' state out of the simulator. Nothing it
produces is an observation; it produces labels.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch

from .mpc import ACT_DIM, N_KNOTS, PlanSpec, decode, reference
from .teacher import RacelineTeacher, plan_geometry_speed

#: Lateral offsets [m left of the raceline] the candidate family spans. Symmetric, and wide enough
#: that the outermost pair is a pass rather than a lean: the half-lane on the traffic family's real
#: floors is about 0.70 m, so +-0.6 m is the outside of the road and +-0.3 m is half a car width of
#: line change. Each is still clamped per raceline point against the lane's own free space
#: (`RacelineTeacher.clamp_offset`), so a wide offset through a narrow section becomes as much of
#: one as fits.
DEFAULT_OFFSETS = (-0.6, -0.3, 0.0, 0.3, 0.6)

#: Multipliers on the plan's two speed targets. 1.0 is the raceline teacher's own plan, 0.8 is
#: following, 0.55 and 0.3 are braking -- the candidate a car that has stopped in the lane has to
#: make winnable, because the alternative at 8 m/s is a rear-end the cost function cannot avoid by
#: steering alone.
DEFAULT_SPEEDS = (1.0, 0.8, 0.55, 0.3)


@dataclass
class TeacherCost:
    """Weights of `C = C_progress + w_wall C_wall + w_opp C_opp_future + w_clear C_clear + w_smooth C_smooth`.

    Every term is built to live in roughly [0, 1] per candidate before its weight, so the weights
    read as a ratio between reasons rather than as unit conversions:

    * `C_progress` is the arc the plan takes out of the raceline over the horizon, divided by what
      the speed limit could have taken: 0 for a perfect lap, -1 is impossible, and the useful range
      between two candidates is a few tenths.
    * `C_wall` is the mean depth the body would be inside a wall over the horizon, as a fraction of
      its own half-width. Weighted an order above everything else: a candidate that hits a wall is
      not a trade-off.
    * `C_opp_future` is the mean over horizon samples, summed over opponents, of an elliptical
      proximity kernel around the other car's *predicted* position (see `_opp_cost`).
    * `C_clear` is the mean squared shortfall of free space against `wall_margin`, the same 0.25 m
      the plan-clearance penalty uses -- the plans that end races graze walls
      (`docs/research/failure-attribution-2026-09-13.md` section 2).
    * `C_smooth` is how far the candidate's trajectory runs from the raceline teacher's own, measured
      sideways and in units of `SMOOTH_REF_M`. It is what makes "stay on the line" the default, and
      it has real work to do: measured with nothing to race, cutting to the inside of the raceline
      buys about 0.01 of the progress term, because the arc is scored against a line whose speed
      profile was computed for the line's own curvature and not for a shorter one. Without this
      term the argmin of an empty road is a permanent 0.6 m inside line.
    """
    progress: float = 1.0
    wall: float = 30.0
    opp: float = 8.0
    clear: float = 1.0
    smooth: float = 0.35


#: Semi-axes [m] of the two ellipses `_opp_cost` measures car-to-car separation with, longitudinal
#: first, in the frame of the candidate plan at that instant.
#:
#: Elliptical and not a circle, and this is the single most consequential choice in the file. The
#: T family's own note records that on a 0.70 m half-lane a completed pass is rare; two cars side
#: by side are 0.31 m of body apart across a lane that leaves at most ~0.6 m of centre-to-centre
#: room. A circular safety radius large enough to stop a rear-end (the bodies are 0.58 m long) is
#: therefore also large enough to forbid every pass the track allows, and a teacher built on one is
#: a slower raceline teacher rather than a racing one. The ellipses provide soft proximity preferences only. Actual feasibility uses
#: oriented chassis SAT including the simulator rear extension; an ellipse misses box corners.
OPP_HARD = (0.62, 0.34)
OPP_SOFT = (1.10, 0.50)

#: [m] the lateral departure from the reference plan that `C_smooth` calls "one". Half a metre is
#: about the widest line change these lanes have room for, so a full-width move costs one unit of
#: that term and a half-width move a quarter of it.
SMOOTH_REF_M = 0.5

#: Weight of the hard ellipse inside the kernel, relative to the soft one's maximum of 1.
OPP_HARD_WEIGHT = 3.0


class InteractiveTeacher:
    """Best response to the opponents' motion, in the student's own plan space.

    Drop-in for `RacelineTeacher` wherever a *label* is wanted (`F1VecEnv.teacher_label`,
    `learn.evaluate --teacher`): `plan_action` has the same signature and returns the same
    (B, ACT_DIM) normalized plan in the same range. `attach(env)` gives it the race it is racing
    in; without one -- a solo env, or a caller that never attached -- the opponent term is
    identically zero and what is left is the raceline teacher with a wall-margin and progress
    search over the same family, which is a different thing from this class's purpose and is
    reported as such rather than pretended about.
    """

    def __init__(self, base: RacelineTeacher, env=None,
                 offsets: Sequence[float] = DEFAULT_OFFSETS,
                 speeds: Sequence[float] = DEFAULT_SPEEDS,
                 horizon_s: float = 1.0, cost: Optional[TeacherCost] = None,
                 cand_iters: int = 2,
                 wall_margin: float = 0.25, future_model: Optional[str] = None,
                 lane_clamp: bool = False):
        if not isinstance(base, RacelineTeacher):
            raise TypeError("InteractiveTeacher wraps a RacelineTeacher: the raceline, the speed "
                            "profile and the plan fit are all its reference, not a reimplementation")
        if not offsets or not speeds:
            raise ValueError("the candidate family needs at least one offset and one speed")
        if 0.0 not in tuple(float(o) for o in offsets):
            raise ValueError(f"offsets {list(offsets)} do not contain 0.0: the raceline teacher's "
                             f"own plan has to be in the family, or the argmin is a search over "
                             f"alternatives to a plan that was never a candidate")
        if horizon_s <= 0:
            raise ValueError(f"horizon_s must be positive, got {horizon_s}")
        self.base = base
        self.device = base.device
        self.offsets = torch.tensor([float(o) for o in offsets], dtype=torch.float32, device=self.device)
        self.speeds = torch.tensor([float(v) for v in speeds], dtype=torch.float32, device=self.device)
        self.horizon_s = float(horizon_s)
        self.cost = cost or TeacherCost()
        self.cand_iters = int(cand_iters)
        self.wall_margin = float(wall_margin)
        self.future_model = future_model
        #: Whether the candidate offsets go through `RacelineTeacher.offset_limit`, the isotropic
        #: per-raceline-point lane bound the scripted opponent events are clamped by.
        #:
        #: OFF by default, and the reason is measured: that bound is `EDT - half width - margin`,
        #: the free space in EVERY direction, so it cannot tell the wide side of the lane from the
        #: narrow one. On the tracks this teacher races on, the smallest such bound over the next
        #: 16 m is 0.17-0.33 m on every map tried -- which would clamp every pass to less than a
        #: car's width of line change whichever way the room actually is. What replaces it is
        #: strictly better information: `C_wall` and `C_clear` read the distance field along the
        #: candidate's OWN trajectory, on the side it actually goes. The clamp exists for the
        #: opponents because nothing else was watching their line; here something is.
        self.lane_clamp = bool(lane_clamp)
        #: The teacher object the candidate family is generated through. A shallow copy of `base`:
        #: every tensor is shared, so it costs nothing, and the two differ only in `offset_limit`.
        #: A copy rather than a save-and-restore on `base` itself, because in a race `base` is very
        #: often the object driving the OPPONENTS (`env.teacher`), and briefly clearing its lane
        #: clamp is a hazard that would only ever show up as an opponent in a wall.
        self._gen = copy.copy(base)
        self._gen._defer_profile_projection = True
        self.env = None
        self.track = None
        self.half_width = 0.155
        self.half_length = 0.29
        self._grid = None
        #: Index of the raceline teacher's own plan in the family (offset 0, speed 1.0), or the
        #: nearest speed to 1.0 when the caller did not include it. `C_smooth` is measured against
        #: it and it is what the argmin falls back to when nothing else separates the candidates.
        #: The on-line offset's row. Read once here: `nonzero` is a host sync, and `_candidates`
        #: used to pay it on every call.
        self.i0 = int((self.offsets == 0).nonzero()[0, 0])
        self.base_index = (self.i0 * self.speeds.numel()
                           + int((self.speeds - 1.0).abs().argmin()))
        #: Per-step diagnostics of the last `plan_action`, for the tests, the benchmark and the
        #: census: which candidate won each row, and the cost of each.
        self.last_choice: Optional[torch.Tensor] = None
        self.last_cost: Optional[torch.Tensor] = None
        # Last scoring pass: sampled chassis/props/walls and opponent body feasibility.
        # This is a sampled reference check, not a dynamics or swept-volume guarantee.
        # A blocked row attempts braking and is excluded from valid supervision.
        self.last_hard_clear: Optional[torch.Tensor] = None
        self.last_blocked: Optional[torch.Tensor] = None
        self.last_label_valid: Optional[torch.Tensor] = None
        if env is not None:
            self.attach(env)

    # ------------------------------------------------------------------ wiring
    def attach(self, env) -> "InteractiveTeacher":
        """Bind the race this teacher is racing in: the track (walls) and the other cars."""
        self.env = env
        self.track = env.sim.track
        self.half_width = 0.5 * float(env.cfg.vehicle.width)
        self.half_length = 0.5 * float(env.cfg.vehicle.length)
        return self

    @property
    def n_candidates(self) -> int:
        return int(self.offsets.numel() * self.speeds.numel()) + 5  # on-line brake plus four immediate evasions

    #: `label_grip` and `speed_scale` live on the reference teacher, so that every caller that sets
    #: them (`learn.common.make_teacher`, `learn.dagger`) reaches the thing that actually uses them.
    @property
    def label_grip(self) -> str:
        return self.base.label_grip

    @label_grip.setter
    def label_grip(self, v: str):
        self.base.label_grip = v

    @property
    def speed_scale(self) -> float:
        return self.base.speed_scale

    @speed_scale.setter
    def speed_scale(self, v: float):
        self.base.speed_scale = float(v)

    @property
    def offset_limit(self):
        return self.base.offset_limit

    @offset_limit.setter
    def offset_limit(self, v):
        self.base.offset_limit = v

    def project(self, xy, tid=None):
        return self.base.project(xy, tid)

    def __call__(self, state, P=None, tid=None, offset=None):
        """The direct (steer, speed) action space, which is the reference teacher's unchanged.

        Deliberately NOT a search. The candidate family is a family of *plans*; a one-step steer
        and speed cannot carry a manoeuvre that only pays off half a second later, so a
        best-response teacher in that action space would be a different design and this one would
        be quietly pretending to be it. `action_mode="plan"` is what this class is for.
        """
        return self.base(state, P, tid, offset=offset)

    # ------------------------------------------------------------------ the label
    @torch.no_grad()
    def plan_action(self, state: torch.Tensor, P=None, tid: Optional[torch.Tensor] = None,
                    v_max: float = 8.0, spec=None, iters: int = 6,
                    offset: Optional[torch.Tensor] = None,
                    idx: Optional[torch.Tensor] = None,
                    plan_speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, ACT_DIM) the argmin candidate, in the same normalized plan space `RacelineTeacher`
        returns. Every argument means what it means there, and `offset` shifts the whole candidate
        family (a scripted lane change the caller wants held) -- which is what keeps this
        signature-compatible with the teacher it replaces.

        `iters` is accepted and **not used**: the family is generated at `cand_iters`, and the
        reference plan this search is relative to is the offset-0 candidate OF that family. Fitting
        it to a different budget would make the thing `C_smooth` is measured against a plan that is
        not in the family, and the argmin's default would stop being a candidate.
        """
        spec = spec or PlanSpec()
        B = state.shape[0]
        dev = state.device
        tid = torch.zeros(B, dtype=torch.long, device=dev) if tid is None else tid
        idx = self.base.project(state[:, :2], tid)[0] if idx is None else idx
        cand = self._candidates(state, P, tid, v_max, spec, offset, idx,
                                plan_speed=plan_speed)                       # (K, B, ACT_DIM)
        cost = self.score(cand, state, tid, v_max, spec, idx,
                          plan_speed=plan_speed)                             # (K, B)
        j = cost.argmin(0)                                                       # (B,)
        # All-blocked does not imply braking is safest: a rear-closing car can
        # make it the first collision. Delay the first predicted contact, then
        # minimize contact duration; retain the original cost as deterministic tie-break.
        j = torch.where(self.last_blocked, self.last_fallback_choice, j)
        self.last_choice, self.last_cost = j, cost
        chosen = cand.gather(0, j[None, :, None].expand(1, B, cand.shape[2]))[0]
        return chosen

    # ------------------------------------------------------------------ pieces
    def _candidates(self, state, P, tid, v_max, spec, offset, idx, plan_speed=None) -> torch.Tensor:
        """(K, B, ACT_DIM), offset-major: candidate k*len(speeds)+m is offset k at speed m.

        The offsets go through `RacelineTeacher.plan_action` itself, on one (n_off * B) tile, so
        each candidate IS the raceline teacher's plan through the offset line -- lane clamp,
        understeer, latency compensation and all -- rather than a reimplementation that would drift
        away from it. The speeds are then a scaling of the plan's two normalized speed targets,
        which is the same arithmetic `gym_env._opponent_actions` uses for `opp_scale`, and costs
        nothing: the path is unchanged, the profile along it is not.

        `cand_iters` is the Gauss-Newton budget per candidate. The fit starts from the pure-pursuit
        / raceline blend, which is already close; `tests/test_interactive_teacher.py` measures how
        far a 2-iteration candidate is from the 6-iteration one, because the whole family has to be
        affordable inside DAgger's collection loop.
        """
        B = state.shape[0]
        n_off, n_spd = self.offsets.numel(), self.speeds.numel()
        rep = lambda t: t[None].expand(n_off, *t.shape).reshape(-1, *t.shape[1:])
        off = self.offsets[:, None].expand(n_off, B).reshape(-1)
        if offset is not None:
            off = off + rep(offset)
        Pt = None if P is None else {k: rep(v) if torch.is_tensor(v) and v.shape[:1] == (B,) else v
                                     for k, v in P.items()}
        g = self._gen                                      # the base teacher's settings, live
        scale = self.base.speed_scale
        g.speed_scale = rep(scale) if torch.is_tensor(scale) and scale.shape[:1] == (B,) else scale
        g.label_grip = self.base.label_grip
        codes = self.base.label_grip_codes
        g.label_grip_codes = None if codes is None else rep(codes)
        g.offset_limit = self.base.offset_limit if self.lane_clamp else None
        a = g.plan_action(rep(state), Pt, rep(tid), v_max, spec,
                          iters=self.cand_iters, offset=off, idx=rep(idx),
                          plan_speed=None if plan_speed is None else rep(plan_geometry_speed(state, plan_speed)))
        a = a.view(n_off, B, -1)
        # The two axes have to be independent, and `plan_action` does not leave them so: it reads
        # the car's lateral error against the line it is asked to plan through, and a car sitting on
        # the raceline is by definition 0.6 m off a 0.6 m offset line. Its off-line slowdown then
        # cuts the commanded speed by up to 30 % for every offset candidate, so "go left" would
        # arrive already carrying "and slow down" and the argmin could never separate the two. The
        # speed the family starts from is therefore the ON-LINE plan's, for every offset.
        i0 = self.i0
        a = torch.cat([a[..., :-2], a[i0:i0 + 1, :, -2:].expand(n_off, B, 2)], 2)
        out = a[None].expand(n_spd, n_off, B, a.shape[2]).clone()
        sc = self.speeds.view(n_spd, 1, 1, 1)
        out[..., -2:] = ((out[..., -2:] + 1.0) * sc - 1.0).clamp(-1.0, 1.0)
        # offset-major so that a choice index reads as (offset, speed) without a stride argument
        out = out.permute(1, 0, 2, 3).reshape(n_off * n_spd, B, a.shape[2])
        brake = out[self.base_index:self.base_index + 1].clone()
        brake[..., -2:] = -1.0
        evade = self._evasion_candidates(out[self.base_index], state, P, v_max, spec, plan_speed)
        # Preserve every regular index and the automatic raceline-brake index.
        out = torch.cat([out, brake, evade], 0)
        K = out.shape[0]
        tile = lambda t: t[None].expand(K, *t.shape).reshape(-1, *t.shape[1:])
        params = None if P is None else {key: tile(value) if torch.is_tensor(value) and value.shape[:1] == (B,) else value
                                       for key, value in P.items()}
        final = self.base.project_plan_action(out.reshape(K * B, -1), tile(state), params, v_max, spec,
                                             plan_speed=tile(plan_geometry_speed(state, plan_speed)))
        return final.reshape_as(out)

    def _evasion_candidates(self, base, state, P, v_max, spec, plan_speed=None):
        """Immediate left/right, each coasting or braking, then rejoin base geometry.

        Raceline offsets are fitted far down the road; they may all begin with
        the same turn when an adjacent car requires a different action now.
        These four bounded alternatives use current axle grip and steering
        authority. The common speed projector and collision preview still decide
        whether any is usable; their existence is not a safety certificate.
        """
        speed = plan_geometry_speed(state, plan_speed).abs()
        vehicle = self.base.vehicle
        def param(name):
            default = self.base.mu_nom if name == "mu" else getattr(vehicle, name)
            value = default if P is None else P.get(name, default)
            return torch.as_tensor(value, device=state.device, dtype=state.dtype).expand_as(speed)
        lf, lr = param("lf"), param("lr")
        wb = lf + lr
        resistance = param("c_roll") * torch.tanh(speed / .05) + param("c_drag") * speed.square()
        split = param("drive_split_r") if vehicle.wheel_model else torch.ones_like(speed)
        muf, mur = param("mu") * param("mu_f_scale"), param("mu") * param("mu_r_scale")
        front_load, rear_load = lr / wb, lf / wb
        front = ((muf * 9.81 * front_load).square() - ((1 - split) * resistance).square()).clamp_min(0).sqrt() / front_load
        rear = ((mur * 9.81 * rear_load).square() - (split * resistance).square()).clamp_min(0).sqrt() / rear_load
        grip_limit = torch.minimum(front, rear) / speed.square().clamp_min(1e-6)
        steer_limit = torch.tan(param("s_max").clamp(max=self.base.steer_max)) / (wb + spec.k_us * speed.square())
        limit = torch.minimum(grip_limit, steer_limit).clamp(max=spec.kappa_max) / spec.kappa_max
        out = base[None].expand(4, -1, -1).clone()
        # Cached: `new_tensor` from a list is a pageable host copy, i.e. a device sync per call.
        signs = getattr(self, "_evade_signs", None)
        if signs is None or signs.device != state.device or signs.dtype != state.dtype:
            signs = self._evade_signs = state.new_tensor([1., 1., -1., -1.])
        # The first two knots provide immediate authority; later knots retain
        # the existing route, avoiding a second long-horizon path optimizer.
        fade = (1 - torch.arange(N_KNOTS, device=state.device, dtype=state.dtype) / 2).clamp_min(0)
        out[..., :N_KNOTS] = base[None, :, :N_KNOTS] * (1 - fade) + signs[:, None, None] * limit[None, :, None] * fade
        coast = (2 * speed / v_max - 1).clamp(-1, 1)
        targets = torch.stack((coast, -torch.ones_like(coast), coast, -torch.ones_like(coast)))
        out[..., -2:] = targets[..., None]
        return out

    def horizon_times(self, spec: PlanSpec) -> torch.Tensor:
        """The instants the trajectory and the opponents' futures are both sampled at [s]."""
        n = max(2, int(round(self.horizon_s / spec.dt)) + 1)
        return torch.arange(n, device=self.device, dtype=torch.float32) * spec.dt

    def rollout(self, cand: torch.Tensor, state: torch.Tensor, v_max: float, spec: PlanSpec,
                plan_speed: Optional[torch.Tensor] = None):
        """(world xy, world heading, speed) of each candidate, each (K, B, H+1) over the
        `horizon_times` samples -- the xy with a trailing 2.

        `mpc.reference` and nothing else: it is the function the plan tracker walks its own
        reference with, so the trajectory scored here is the tracking reference, not the executed dynamics --
        the speed profile along the arc, the walk from the measured speed under the acceleration
        bounds, the run-on past the end of a short plan. Scoring a plan by re-integrating its
        curvature at the commanded speed instead would score a manoeuvre the tracker cannot
        perform.
        """
        K, B, D = cand.shape
        flat = cand.reshape(K * B, D)
        rep = lambda t: t[None].expand(K, *t.shape).reshape(-1, *t.shape[1:])
        v_meas = rep(plan_geometry_speed(state, plan_speed))
        cap = torch.full_like(v_meas, float(v_max))
        if self.env is not None:
            cap = rep(self.env.speed_cap.to(state.dtype))
        k, Lp, v0, v1 = decode(flat, v_meas, v_max, cap, spec)
        H = int(self.horizon_times(spec).numel()) - 1
        ref = reference(k, Lp, v0, v1, replace(spec, N=H), v_meas)              # (K*B, H+1, 4)
        c, sn = rep(torch.cos(state[:, 2])), rep(torch.sin(state[:, 2]))
        x = rep(state[:, 0])[:, None] + ref[:, :, 0] * c[:, None] - ref[:, :, 1] * sn[:, None]
        y = rep(state[:, 1])[:, None] + ref[:, :, 0] * sn[:, None] + ref[:, :, 1] * c[:, None]
        world = torch.stack([x, y], 2).view(K, B, H + 1, 2)
        psi = (rep(state[:, 2])[:, None] + ref[:, :, 2]).view(K, B, H + 1)
        return world, psi, ref[:, :, 3].view(K, B, H + 1)

    def _tracker_rollout(self, cand, state, v_max, spec, plan_speed=None):
        """Preview the installed legacy MPC without changing its warm start.

        Reference geometry can move laterally or brake faster than the controller
        actually predicts. Use the same solve, measured inputs, latency and shifted
        warm controls. Beyond its solve horizon, hold the final predicted control;
        that continuation is a forecast, not a full-plant safety guarantee.
        Unsupported controller hooks retain an explicitly named reference fallback.
        """
        from .mpc import PlanTracker, solve, solve_fast, _dyn
        from .viewer.graph_fastpath import GraphedCallable
        tracker = getattr(self.env, "tracker", None)
        solver = getattr(tracker, "_solver", None)
        known_solver = (solver is None or solver is solve or solver is solve_fast
                        or (isinstance(solver, GraphedCallable) and solver.name == "mpc.solve"))
        supported = (type(tracker) is PlanTracker and known_solver
                     and tracker._input_hook is None and tracker._plan_hook is None
                     and tracker._command_hook is None)
        self.last_preview_model = "legacy_mpc" if supported else "reference"
        if not supported:
            return self.rollout(cand, state, v_max, spec, plan_speed=plan_speed)
        K, B, D = cand.shape
        tile = lambda x: x[None].expand(K, *x.shape).reshape(-1, *x.shape[1:])
        v = tile(plan_geometry_speed(state, plan_speed))
        cap = tile(self.env.speed_cap.to(state.dtype))
        lr = self.env.last_result
        yaw_rate = (lr.imu[:, :, 2].mean(1) if lr is not None and lr.imu is not None and lr.imu.shape[1] > 0
                    else v.new_empty(0))
        if yaw_rate.numel() == 0:
            yaw_rate = (plan_geometry_speed(state, plan_speed).abs() * torch.tan(tracker.u_prev[:, 0])
                        / (tracker.wb + spec.k_us * plan_geometry_speed(state, plan_speed).square()))
        delay = self.env.tracker_delay
        delay = state.new_full((B,), spec.delay) if delay is None else torch.as_tensor(delay, device=state.device, dtype=state.dtype).expand(B)
        warm = torch.cat((tracker.u_seq[:, 1:], tracker.u_seq[:, -1:]), 1)
        u, z, _ = solve(cand.reshape(K * B, D), v, cap, tile(yaw_rate), tile(delay),
                        tile(tracker.u_prev), tile(warm), spec, tracker.wb, tracker.s_max, tracker.v_max)
        times = self.horizon_times(spec).to(state.dtype)
        needed = int(times.numel()) + 1
        points = list(z.unbind(1))
        for _ in range(max(0, needed - len(points))):
            points.append(_dyn(points[-1], u[:, -1], spec, tracker.wb))
        pred = torch.stack(points, 1)
        # Each env has its own delay; interpolate on that clock rather than
        # pretending the MPC's first state is at time zero.
        phase = (times[None] - tile(delay)[:, None]) / spec.dt
        lo = phase.floor().long().clamp(0, pred.shape[1] - 2)
        hi = lo + 1
        frac = (phase - lo).clamp(0, 1)[..., None]
        gather = lambda j: pred.gather(1, j[..., None].expand(-1, -1, pred.shape[-1]))
        result = gather(lo) * (1 - frac) + gather(hi) * frac
        start = torch.zeros_like(pred[:, 0])
        start[:, 3] = v.abs()
        early = (times[None] / tile(delay)[:, None].clamp_min(1e-6)).clamp(0, 1)[..., None]
        before = start[:, None] * (1 - early) + pred[:, :1] * early
        result = torch.where((phase < 0)[..., None], before, result)
        c, sn = tile(state[:, 2].cos()), tile(state[:, 2].sin())
        x = tile(state[:, 0])[:, None] + result[..., 0] * c[:, None] - result[..., 1] * sn[:, None]
        y = tile(state[:, 1])[:, None] + result[..., 0] * sn[:, None] + result[..., 1] * c[:, None]
        world = torch.stack((x, y), -1).reshape(K, B, -1, 2)
        yaw = (tile(state[:, 2])[:, None] + result[..., 2]).reshape(K, B, -1)
        return world, yaw, result[..., 3].reshape(K, B, -1)

    @torch.no_grad()
    def score(self, cand: torch.Tensor, state: torch.Tensor, tid: torch.Tensor, v_max: float,
              spec: PlanSpec, idx: Optional[torch.Tensor] = None,
              parts: bool = False, plan_speed: Optional[torch.Tensor] = None):
        """(K, B) total cost of each candidate, or (total, {name: (K, B)}) with `parts=True`.

        `C = C_progress + w_wall C_wall + w_opp C_opp_future + w_clear C_clear + w_smooth C_smooth`,
        each term as `TeacherCost` documents it. Exposed rather than private because every test in
        `tests/test_interactive_teacher.py` and the report's tables are about *which term* moved a
        decision, and a choice index alone cannot say.

        If any candidate clears walls, props and oriented opponent bodies at every sample,
        an explicit finite feasibility term ranks all intersecting candidates below all clear ones.
        If none clears it, `last_blocked` is true and plan_action delays the first
        predicted contact without claiming its fallback valid. The ellipse is only a soft preference.
        """
        tid = torch.zeros(state.shape[0], dtype=torch.long, device=state.device) if tid is None else tid
        idx = self.base.project(state[:, :2], tid)[0] if idx is None else idx
        world, psi, _v = self._tracker_rollout(cand, state, v_max, spec, plan_speed=plan_speed)
        w = self.cost
        wall, clear, wall_clear = self._wall_costs(world, tid, psi, return_clear=True)
        opp, hard_clear = self._opp_cost(world, psi, state, spec, return_clear=True)
        hard_clear = hard_clear & wall_clear
        terms = {"progress": w.progress * self._progress_cost(world, tid, idx, v_max),
                 "wall": w.wall * wall,
                 "opp": w.opp * opp,
                 "clear": w.clear * clear,
                 "smooth": w.smooth * self._smooth_cost(world, psi)}
        total = sum(terms.values())
        self.last_hard_clear = hard_clear
        self.last_blocked = ~hard_clear.any(0)
        self.last_label_valid = ~self.last_blocked
        hits = self._last_wall_hits | self._last_opponent_hits
        self.last_collision_samples = hits
        self.last_fallback_choice = self._contact_fallback(hits, total)
        # A data-dependent gap, not another tuned collision weight: even the lowest-cost violating
        # plan must cost more than the highest-cost clear plan. Include the cost magnitude so the
        # strict gap survives rounding when progress weights are large. All-blocked rows stay finite
        # and unchanged so diagnostics remain meaningful even when braking is the fallback.
        gap = total.amax(0) - total.amin(0) + total.abs().amax(0).clamp_min(1.0)
        terms["hard_feasibility"] = torch.where(
            (~hard_clear) & (~self.last_blocked)[None], gap[None], torch.zeros_like(total))
        # Immediate evasions are an escape family, not a faster replacement for
        # a clear ordinary plan. Keep geometric diagnostics intact; eligibility
        # is a separate finite priority with the same strict ranking guarantee.
        regular = int(self.offsets.numel() * self.speeds.numel()) + 1
        ordinary_clear = hard_clear[:regular].any(0)
        emergency = torch.arange(cand.shape[0], device=cand.device)[:, None] >= regular
        eligible = ~(emergency & ordinary_clear[None])
        self.last_candidate_eligible = eligible
        terms["emergency_priority"] = torch.where(~eligible, gap[None], torch.zeros_like(total))
        total = total + terms["hard_feasibility"] + terms["emergency_priority"]
        return (total, terms) if parts else total

    @staticmethod
    def _contact_fallback(hits, cost):
        """Lexicographic risk ordering for blocked candidates; never a valid label."""
        steps = torch.arange(hits.shape[-1], device=hits.device)[None, None]
        first = torch.where(hits, steps, hits.shape[-1]).amin(-1)
        eligible = first == first.amax(0, keepdim=True)
        count = hits.sum(-1)
        fewest = torch.where(eligible, count, hits.shape[-1] + 1).amin(0, keepdim=True)
        eligible = eligible & (count == fewest)
        return torch.where(eligible, cost, torch.full_like(cost, float("inf"))).argmin(0)

    # ------------------------------------------------------------------ costs
    def _progress_cost(self, world, tid, idx, v_max) -> torch.Tensor:
        """-(arc gained along the raceline) / (what the speed limit could have gained).

        The endpoint is projected onto a window of the raceline ahead of where the car is now --
        a window rather than the whole line, because two points a lap apart are not a candidate's
        two possible futures -- and refined along the tangent so the answer is not quantised by the
        window's own spacing. Arc along the LINE and not the path length: a candidate that spends
        its speed going sideways has travelled further and progressed less, which is the trade the
        whole search is about.
        """
        K, B = world.shape[:2]
        base = self.base
        ds = base.ds[tid]                                                       # (B,)
        grid = self._arc_grid(world.dtype)
        j = (idx[:, None] + (grid[None, :] / ds[:, None]).round().long()) % base.N   # (B, W)
        pw = base.xy[tid[:, None].expand_as(j), j]                              # (B, W, 2)
        end = world[:, :, -1]                                                   # (K, B, 2)
        d2 = ((end[:, :, None, :] - pw[None]) ** 2).sum(-1)                     # (K, B, W)
        m = d2.argmin(2)                                                        # (K, B)
        ar = torch.arange(B, device=world.device)
        near = pw[ar[None, :].expand(K, B), m]                                  # (K, B, 2)
        tanv = base.tan[tid[None, :].expand(K, B), j[ar[None, :].expand(K, B), m]]
        arc = grid[m] + ((end - near) * tanv).sum(-1)
        return -arc / max(float(v_max) * self.horizon_s, 1e-6)

    def _arc_grid(self, dtype) -> torch.Tensor:
        """Arc offsets [m] the endpoint projection searches over, from just behind the car to past
        the longest plan a 10 m/s car can have."""
        if getattr(self, "_grid", None) is None or self._grid.dtype != dtype:
            self._grid = torch.linspace(-1.0, 18.0, 96, device=self.device, dtype=dtype)
        return self._grid

    def _corners(self, world, psi):
        """Simulator chassis footprint in perimeter order, including its orientation."""
        ax = torch.stack((psi.cos(), psi.sin()), -1) * self.half_length
        ay = torch.stack((-psi.sin(), psi.cos()), -1) * self.half_width
        return torch.stack((world + ax + ay, world + ax - ay,
                            world - ax - ay, world - ax + ay), -2)

    def _wall_costs(self, world, tid, psi=None, return_clear=False):
        """Oriented chassis against wall EDT and exact analytic prop sections.

        The wall check matches the simulator's four corners and half-cell surface
        correction; props use its polygon SAT, so containment of small assets is
        visible even when every chassis corner is clear. Per-env prop identities
        are preserved when tiling candidates and horizon samples.
        """
        K, B, H, _ = world.shape
        z = torch.zeros((K, B), device=world.device, dtype=world.dtype)
        if self.track is None:
            self._last_wall_hits = torch.zeros((K, B, H), dtype=torch.bool, device=world.device)
            return (z, z, torch.ones_like(z, dtype=torch.bool)) if return_clear else (z, z)
        psi = torch.zeros(world.shape[:-1], device=world.device, dtype=world.dtype) if psi is None else psi
        corners = self._corners(world, psi)
        t = tid[None, :, None].expand(K, B, H)
        edt = self.track.sample_edt(corners, t[..., None])
        free = edt.amin(-1) - 0.5 * self.track.t_res[t]
        penetration = (-free).clamp_min(0.0)
        hit = free < 0.0
        if self.track.has_props:
            from .prop_math import prism_contacts
            eid = torch.arange(B, device=world.device)[None, :, None].expand(K, B, H).reshape(-1)
            sim = self.env.sim if self.env is not None else None
            reach = getattr(sim, "prop_reach", 0.0)
            props = self.track.props_near(t.reshape(-1), world.reshape(-1, 2), eid, reach)
            heights = getattr(sim, "car_dims", None)
            heights = 0.25 if heights is None else heights[eid, 2]
            depth, _, _ = prism_contacts(corners.reshape(-1, 4, 2), *props, heights)
            depth = depth.reshape(K, B, H)
            penetration = torch.maximum(penetration, depth)
            hit = hit | (depth > 0.0)
        wall = penetration.mean(2) / max(self.half_width, 1e-6)
        clear = ((1.0 - free / max(self.wall_margin, 1e-6)).clamp_min(0.0) ** 2).mean(2)
        self._last_wall_hits = hit
        return (wall, clear, ~hit.any(2)) if return_clear else (wall, clear)

    def _body_overlap(self, world, psi, fut, yaw):
        """SAT for the same rear-extended rectangle used by sim._car_contacts."""
        B = world.shape[1]
        sim = getattr(self.env, "sim", None)
        rear = getattr(sim, "car_rear", None)
        other = getattr(sim, "other_idx", None)
        ego_rear = world.new_zeros(B) if rear is None else rear[:, 0]
        opp_rear = world.new_zeros(fut.shape[:2]) if rear is None else rear[other, 0]
        a = torch.stack((psi.cos(), psi.sin()), -1)[:, :, None]
        b = torch.stack((yaw.cos(), yaw.sin()), -1)[None]
        ap = torch.stack((-a[..., 1], a[..., 0]), -1)
        bp = torch.stack((-b[..., 1], b[..., 0]), -1)
        ca = world[:, :, None] - 0.5 * ego_rear[None, :, None, None, None] * a
        cb = fut[None] - 0.5 * opp_rear[None, :, :, None, None] * b
        delta = cb - ca
        ha = self.half_length + 0.5 * ego_rear[None, :, None, None]
        hb = self.half_length + 0.5 * opp_rear[None, :, :, None]
        overlap = torch.ones(delta.shape[:-1], dtype=torch.bool, device=world.device)
        for axis in (a, ap, b, bp):
            distance = (delta * axis).sum(-1).abs()
            radius = (ha * (a * axis).sum(-1).abs() + self.half_width * (ap * axis).sum(-1).abs()
                      + hb * (b * axis).sum(-1).abs() + self.half_width * (bp * axis).sum(-1).abs())
            overlap = overlap & (distance <= radius)
        return overlap

    def _swept_body_overlap(self, world, psi, fut, yaw):
        """Conservative interval SAT with common translation cancelled.

        Bounding each car's independent world-space sweep blocks safe following:
        both cars occupy the same road at different instants. Instead subtract
        ego's translation and bound only relative motion. Rotation padding is the
        corner's maximum travel from the interval midpoint orientation; it comes
        from dimensions and measured forecast yaw change, not a clearance knob.
        This guards between forecast samples, not arbitrary unmodelled dynamics.
        """
        B = world.shape[1]
        sim = getattr(self.env, "sim", None)
        rear = getattr(sim, "car_rear", None)
        other = getattr(sim, "other_idx", None)
        er = world.new_zeros(B) if rear is None else rear[:, 0]
        pr = world.new_zeros(fut.shape[:2]) if rear is None else rear[other, 0]
        pa = psi[:, :, None]
        pb = yaw[None]
        da = torch.atan2((pa[..., 1:] - pa[..., :-1]).sin(), (pa[..., 1:] - pa[..., :-1]).cos())
        db = torch.atan2((pb[..., 1:] - pb[..., :-1]).sin(), (pb[..., 1:] - pb[..., :-1]).cos())
        ma, mb = pa[..., :-1] + da * .5, pb[..., :-1] + db * .5
        axis = lambda angle: torch.stack((angle.cos(), angle.sin()), -1)
        ax, bx = axis(ma), axis(mb)
        ay = torch.stack((-ax[..., 1], ax[..., 0]), -1)
        by = torch.stack((-bx[..., 1], bx[..., 0]), -1)
        ca = world[:, :, None] - .5 * er[None, :, None, None, None] * axis(pa)
        cb = fut[None] - .5 * pr[None, :, :, None, None] * axis(pb)
        rel = cb - ca
        center = (rel[..., 1:, :] + rel[..., :-1, :]) * .5
        move = rel[..., 1:, :] - rel[..., :-1, :]
        ha = self.half_length + .5 * er[None, :, None, None]
        hb = self.half_length + .5 * pr[None, :, :, None]
        ra = (ha.square() + self.half_width ** 2).sqrt()
        rb = (hb.square() + self.half_width ** 2).sqrt()
        apad = 2 * ra * (.25 * da.abs()).sin()
        bpad = 2 * rb * (.25 * db.abs()).sin()
        # Rear-envelope centres rotate about the CoG. Include arc/chord sag,
        # together with the constant-turn CoG arc between forecast endpoints.
        a_sag = (.5 * er[None, :, None, None] * (1 - (.5 * da).cos())
                 + .5 * (world[..., 1:, :] - world[..., :-1, :]).norm(dim=-1)[:, :, None] * (.25 * da.abs()).tan())
        b_sag = (.5 * pr[None, :, :, None] * (1 - (.5 * db).cos())
                 + .5 * (fut[..., 1:, :] - fut[..., :-1, :]).norm(dim=-1)[None] * (.25 * db.abs()).tan())
        al, aw = ha + apad, self.half_width + apad
        bl = hb + bpad + a_sag + b_sag + .5 * (move * bx).sum(-1).abs()
        bw = self.half_width + bpad + a_sag + b_sag + .5 * (move * by).sum(-1).abs()
        overlap = torch.ones(center.shape[:-1], dtype=torch.bool, device=world.device)
        for direction in (ax, ay, bx, by):
            separation = (center * direction).sum(-1).abs()
            reach = (al * (ax * direction).sum(-1).abs() + aw * (ay * direction).sum(-1).abs()
                     + bl * (bx * direction).sum(-1).abs() + bw * (by * direction).sum(-1).abs())
            overlap = overlap & (separation <= reach)
        return overlap

    def _current_motion_future(self, state, times):
        """Opponent continuation before assuming its next command is achieved.

        Keep measured body velocity and yaw rate. The exact constant-turn arc
        reduces continuously to constant world velocity at zero yaw rate. This
        independent guard prevents a stale/optimistic tracker forecast from being
        the only reason a candidate is called clear.
        """
        other = getattr(getattr(self.env, "sim", None), "other_idx", None)
        if other is None:
            return None
        st = state[other]
        theta = st[..., 5, None] * times
        along = times * torch.sinc(theta / math.pi)
        across = .5 * times * theta * torch.sinc(theta / (2 * math.pi)).square()
        c, sn = st[..., 2].cos(), st[..., 2].sin()
        vx = st[..., 3] * c - st[..., 4] * sn
        vy = st[..., 3] * sn + st[..., 4] * c
        x = st[..., 0, None] + vx[..., None] * along - vy[..., None] * across
        y = st[..., 1, None] + vy[..., None] * along + vx[..., None] * across
        return torch.stack((x, y), -1), st[..., 2, None] + theta

    def _opp_cost(self, world, psi, state, spec, return_clear: bool = False):
        """The term the whole teacher exists for: the candidate's own position at each instant
        against each opponent's *predicted* position at that same instant.

        phi is an elliptical hinge in the candidate's own frame at that instant -- longitudinal and
        lateral separation are not the same quantity to a car that is 0.58 m long and 0.31 m wide,
        and a circular one wide enough to stop a rear-end also forbids every pass a 1.4 m lane
        allows (see `OPP_HARD` / `OPP_SOFT`). Summed over opponents, averaged over the horizon,
        and zero for an opponent outside `overtake_range` -- a car that far away is not being
        raced, and reaching for it would move the plan for something that is not there.
        """
        K, B, H, _ = world.shape
        if self.env is None or self.env.M < 2:
            self._last_opponent_hits = torch.zeros((K, B, H), dtype=torch.bool, device=world.device)
            cost = torch.zeros(K, B, device=world.device, dtype=world.dtype)
            return (cost, torch.ones_like(cost, dtype=torch.bool)) if return_clear else cost
        times = self.horizon_times(spec)
        fut, present, yaw = self.env.opponent_future(times, model=self.future_model, state=state, return_yaw=True)
        fut = fut.to(world.dtype)                                               # (B, C, H, 2)
        d = fut[None, :, :, :, :] - world[:, :, None, :, :]                     # (K, B, C, H, 2)
        c, sn = torch.cos(psi), torch.sin(psi)                                  # (K, B, H)
        dl = d[..., 0] * c[:, :, None, :] + d[..., 1] * sn[:, :, None, :]
        dt = -d[..., 0] * sn[:, :, None, :] + d[..., 1] * c[:, :, None, :]
        e_hard = torch.sqrt((dl / OPP_HARD[0]) ** 2 + (dt / OPP_HARD[1]) ** 2)
        e_soft = torch.sqrt((dl / OPP_SOFT[0]) ** 2 + (dt / OPP_SOFT[1]) ** 2)
        phi = (1.0 - e_soft).clamp_min(0.0) ** 2 + OPP_HARD_WEIGHT * (1.0 - e_hard).clamp_min(0.0)
        phi = phi * present.to(phi.dtype)[None, :, :, None]
        cost = phi.mean(3).sum(2)
        if not return_clear:
            return cost
        endpoint = self._body_overlap(world, psi, fut, yaw.to(world.dtype))
        swept = self._swept_body_overlap(world, psi, fut, yaw.to(world.dtype))
        continued = self._current_motion_future(state, times.to(world.dtype))
        if continued is not None:
            continued_xy, continued_yaw = continued
            endpoint = endpoint | self._body_overlap(world, psi, continued_xy, continued_yaw)
            swept = swept | self._swept_body_overlap(world, psi, continued_xy, continued_yaw)
        # Charge a crossed interval at its start, so the risk fallback never
        # claims the whole sample period remains free before a between-tick hit.
        swept = torch.cat((swept, torch.zeros_like(endpoint[..., :1])), -1)
        self._last_opponent_hits = ((endpoint | swept)
                                    & present.bool()[None, :, :, None]).any(2)
        return cost, ~self._last_opponent_hits.any(2)

    def _smooth_cost(self, world: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Mean squared LATERAL departure of the trajectory from the raceline teacher's own, in
        units of `SMOOTH_REF_M`. Exactly 0 for the reference candidate itself.

        Measured on the trajectory and not on the curvature knots, which is what this was first
        written as. A lateral shift of D over a plan of length L needs a knot amplitude of about
        6 D / L^2, so the knot form of this term varies by a factor of forty between a car at 2 m/s
        and one at 8 m/s -- and no single weight is then right at both ends. The lateral departure
        is the quantity that actually matters and it is the same number at every speed.

        Lateral and not total distance, so that the speed family does not pay it: a slower
        candidate is *behind* the reference along the same line, which is a decision for the
        progress term to price and not this one.
        """
        b = self.base_index
        d = world - world[b:b + 1]
        c, sn = torch.cos(psi[b:b + 1]), torch.sin(psi[b:b + 1])
        lat = -d[..., 0] * sn + d[..., 1] * c
        return ((lat / SMOOTH_REF_M) ** 2).mean(2)
