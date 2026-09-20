"""What every published F1TENTH local planner needs from this simulator, in one place.

The planners this project reproduces as comparison opponents -- ForzaETH's `spliner`, the
fixed-lane switchers several teams race -- differ in exactly one thing: how they choose a lateral
offset from the reference line when there is a car in the way. Everything underneath is the same
question asked of the same simulator, and it is not the interesting part:

* where the other cars are in Frenet coordinates (arc ahead, metres left of the line);
* how much room a given lateral offset has at a given station;
* how a chosen offset reaches `RacelineTeacher`, which owns the line, the grip-aware speed profile,
  the latency compensation and the plan fit;
* the wiring that keeps a slot's speed band and grip label on that reference teacher, where the
  per-car tensors live.

This class is that part. A planner subclasses it and writes `decide`. Two consequences worth
stating, because both are load-bearing:

**The reference teacher is never reimplemented.** A subclass returns an offset; the line that
comes out is `RacelineTeacher`'s line through a displaced reference. A baseline that drew its own
line would be measured against a different driver than the solo control, and the comparison would
stop meaning anything.

**Nothing here is a host sync.** These planners run inside the env's per-step opponent command for
every row of the batch, in training as well as evaluation. A `.item()` or a `bool(t.any())` in
this path costs a device round trip on every step of every race, so the branches are `torch.where`
and the "nothing to plan against" case is decided from Python state, never from a tensor.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .teacher import RacelineTeacher


class FrenetOpponentPlanner:
    """The Frenet view of a race, for a planner that chooses a lateral offset.

    Subclasses implement `decide(state, tid) -> (offset | None, state_code, speed_scale)` and may
    define `STATE_NAMES` for whatever their state machine calls its codes.
    """

    #: What `decide`'s second return means, index by code. Subclasses override.
    STATE_NAMES: Tuple[str, ...] = ("racing",)

    def __init__(self, base: RacelineTeacher, env=None):
        if not isinstance(base, RacelineTeacher):
            raise TypeError(f"{type(self).__name__} wraps a RacelineTeacher: the reference line, "
                            f"the speed profile and the plan fit are all its reference, not a "
                            f"reimplementation")
        self.base = base
        self.device = base.device
        self.env = None
        self.track = None
        self.half_width = 0.155
        self.v_max = 8.0
        #: Which opponent column `_opponents` last picked per row. `_opponent_speed` reads it, and
        #: the tests assert on it.
        self._nearest: Optional[torch.Tensor] = None
        #: Last `decide()`, for the tests, the census and anything plotting a state machine.
        self.last_state: Optional[torch.Tensor] = None
        self.last_offset: Optional[torch.Tensor] = None
        if env is not None:
            self.attach(env)

    # ------------------------------------------------------------------ wiring
    def attach(self, env) -> "FrenetOpponentPlanner":
        """Bind the race this planner is driving in: the walls, the cars, the speed ceiling."""
        self.env = env
        self.track = env.sim.track
        self.half_width = 0.5 * float(env.cfg.vehicle.width)
        self.v_max = float(env.ecfg.v_max_policy)
        return self

    #: `label_grip` and `speed_scale` live on the reference teacher, so that every caller which
    #: sets them (a slot's speed band, `learn.common.make_teacher`) reaches the thing that uses
    #: them. Same contract as `InteractiveTeacher`.
    @property
    def label_grip(self) -> str:
        return self.base.label_grip

    @label_grip.setter
    def label_grip(self, v: str):
        self.base.label_grip = v

    @property
    def speed_scale(self):
        return self.base.speed_scale

    @speed_scale.setter
    def speed_scale(self, v):
        self.base.speed_scale = v

    @property
    def offset_limit(self):
        return self.base.offset_limit

    @offset_limit.setter
    def offset_limit(self, v):
        self.base.offset_limit = v

    def project(self, xy, tid=None):
        return self.base.project(xy, tid)

    # ------------------------------------------------------------------ the command
    @torch.no_grad()
    def plan_action(self, state: torch.Tensor, P=None, tid: Optional[torch.Tensor] = None,
                    v_max: float = 8.0, spec=None, iters: int = 6,
                    offset: Optional[torch.Tensor] = None,
                    idx: Optional[torch.Tensor] = None,
                    plan_speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, ACT_DIM) the reference teacher's plan through the offset this planner chose.

        Every argument means what it means on `RacelineTeacher.plan_action`. `offset` is added to
        the planner's own rather than replacing it: it is how a scripted lane change reaches the
        teacher, and a car doing both is doing both.
        """
        d, _mode, scale = self.decide(state, tid)
        a = self.base.plan_action(state, P, tid, v_max, spec, iters=iters,
                                  offset=self._merge_offset(offset, d),
                                  idx=idx, plan_speed=plan_speed)
        return self._scale_plan_speed(a, scale)

    @torch.no_grad()
    def __call__(self, state: torch.Tensor, P=None, tid: Optional[torch.Tensor] = None,
                 offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The direct (steer, speed) action space, through the same offset.

        The offset is a *path*, so it carries into this space unchanged; only the speed scaling
        has to be applied to a column instead of to the plan's two knots.
        """
        d, _mode, scale = self.decide(state, tid)
        cmd = self.base(state, P, tid, offset=self._merge_offset(offset, d))
        return torch.stack([cmd[:, 0], cmd[:, 1] * scale.to(cmd.dtype)], 1)

    def decide(self, state: torch.Tensor, tid: Optional[torch.Tensor] = None):
        raise NotImplementedError(f"{type(self).__name__} must choose a lateral offset")

    @staticmethod
    def _merge_offset(caller, chosen):
        """The offset to hand the reference teacher: the caller's, this planner's, or neither.

        `None` has to survive when neither has one, and not become a tensor of zeros, because the
        teacher does not treat them the same: given any offset it measures its own off-line error
        as the signed cross-track distance to that line, and given none it uses the Euclidean
        distance to the nearest reference point. The two differ by the line's discretisation, so a
        zeros tensor would make a car with nothing to race drive a hair off the raceline teacher's
        own speed -- and a baseline's solo pace is exactly what its traffic pace is read against.
        """
        if chosen is None:
            return caller
        return chosen if caller is None else caller + chosen

    @staticmethod
    def _scale_plan_speed(action: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Scale a normalized plan's two speed targets per row -- the arithmetic
        `F1VecEnv._opponent_actions` uses for a slot's speed band, for the same reason: the path
        is unchanged and the profile along it is not.

        An unscaled row is left alone rather than multiplied by one: `(x + 1) * 1 - 1` is not `x`
        in float32. The row test does it without a host sync, which a `scale.all()` short circuit
        would cost on every step of every race.
        """
        out = action.clone()
        s = scale.to(out.dtype)[:, None]
        scaled = ((out[:, -2:] + 1.0) * s - 1.0).clamp(-1.0, 1.0)
        out[:, -2:] = torch.where(s == 1.0, out[:, -2:], scaled)
        return out

    # ------------------------------------------------------------------ the Frenet view
    def _stretch(self, state: torch.Tensor) -> torch.Tensor:
        """`clip(1 + v / v_max, 1.0, 1.5)`: how much more road the same manoeuvre needs at pace.

        ForzaETH's scaling, and it is not specific to their planner -- every one of these grows its
        distances with speed, so it lives here.
        """
        return (1.0 + state[:, 3].to(state.dtype) / max(self.v_max, 1e-6)).clamp(1.0, 1.5)

    def _clearance_at(self, idx: torch.Tensor, tid: torch.Tensor, d) -> torch.Tensor:
        """Free space at the reference point `idx` displaced `d` metres to its left [m].

        The simulator's own distance field, at the point the displaced line actually passes
        through. Better information than a lane half-width, which cannot tell the wide side of a
        corridor from the narrow one: this is the clearance on the side the car really goes.
        """
        base = self.base
        p = base.xy[tid, idx]                                          # (..., 2) reference point
        n = self._normal(tid, idx)
        d = torch.as_tensor(d, device=p.device, dtype=p.dtype)
        return self.track.sample_edt(p + d[..., None] * n, tid)

    def _normal(self, tid: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Left-of-travel unit normal of the reference line, the direction `offset` counts in."""
        t_ = self.base.tan[tid, idx]
        return torch.stack([-t_[..., 1], t_[..., 0]], -1)

    def _lateral(self, xy: torch.Tensor, tid: torch.Tensor):
        """(signed metres left of the reference line, nearest reference index) of each point.

        Computed here because `RacelineTeacher.project` returns an *unsigned* distance, and which
        side of the line the other car is on is the entire question these planners answer.
        """
        idx = self.base.project(xy, tid)[0]
        return ((xy - self.base.xy[tid, idx]) * self._normal(tid, idx)).sum(-1), idx

    def _opponents(self, state: torch.Tensor, tid: Optional[torch.Tensor], rear: torch.Tensor):
        """(gap, d_opp, idx_opp, tid, ego_d) for the car each row should plan against, or None.

        `gap` is the signed arc to it and `d_opp` its lateral offset from the reference line. The
        longitudinal half comes from the env's own `signed_gaps`, which already wraps at the start
        line and is already batched.

        `rear` is how far behind a car may be and still matter, per row: a manoeuvre is not over
        when the other car's gap goes negative, it is over when the line has rejoined. The car
        picked is the one with the smallest arc that is not behind that bound, which finishes the
        pass in progress before starting the next -- what a driver does.

        None whenever there is nothing to plan against: no race, or a caller that passed a state
        this planner cannot match to the simulator's rows. Driving the raceline in that case is
        the honest answer; guessing where the other cars are is not.
        """
        env = self.env
        if env is None or int(getattr(env, "M", 1)) < 2 or self.track is None:
            return None
        if env.sim.other_idx is None or state.shape[0] != env.sim.state.shape[0]:
            return None
        B = state.shape[0]
        dev, dt = state.device, state.dtype
        tid_b = torch.zeros(B, dtype=torch.long, device=dev) if tid is None else tid

        gaps = env.signed_gaps(env.sim.s, env.sim.tid).to(dt)          # (B, C) + is ahead of me
        other = env.sim.other_idx                                       # (B, C) row of each
        C = gaps.shape[1]
        tid_o = tid_b[:, None].expand(B, C).reshape(-1)
        d_o = self._lateral(state[other.reshape(-1), :2], tid_o)[0].view(B, C)

        rank = torch.where(gaps > rear[:, None], gaps, torch.full_like(gaps, float("inf")))
        j = rank.argmin(1)
        ar = torch.arange(B, device=dev)
        far = torch.full((B,), float("inf"), device=dev, dtype=dt)
        g = torch.where(torch.isfinite(rank[ar, j]), gaps[ar, j], far)
        idx_opp = self.base.project(state[other[ar, j], :2], tid_b)[0]
        ego_d = self._lateral(state[:, :2], tid_b)[0]
        self._nearest = j
        return g, d_o[ar, j], idx_opp, tid_b, ego_d

    def _opponent_speed(self, state: torch.Tensor) -> torch.Tensor:
        """Speed of the car `_opponents` picked, per row."""
        ar = torch.arange(state.shape[0], device=state.device)
        return state[self.env.sim.other_idx[ar, self._nearest], 3].to(state.dtype)
