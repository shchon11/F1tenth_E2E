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
        self.props = None
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
        #: The procedural obstacle layouts, if this env draws any. They are props, not grid, so the
        #: distance field knows nothing about them -- which is exactly why the layouts have had to
        #: be laid outside the racing line for every driver so far. A planner that asks
        #: `ProceduralObstacles.clearance` does not need that, and `EnvConfig` can then let a crate
        #: stand on the line (`procedural_raceline_corridor`).
        self.props = getattr(env, "procedural", None)
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

    def reset_rows(self, rows: torch.Tensor) -> None:
        """Forget whatever this planner was in the middle of, for the rows that just reset.

        A planner carries state across steps on purpose -- which side a pass is committed to, which
        lane is held -- and a reset puts a different race on that row. Without this the new race
        inherits the old one's commitment, which is a car that will not change sides for a reason
        that no longer exists. Subclasses with such state override it; the default has none.
        """
        return None

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

    def _clearance_at(self, idx: torch.Tensor, tid: torch.Tensor, d, props_only: bool = False) -> torch.Tensor:
        """Free space at the reference point `idx` displaced `d` metres to its left [m].

        The simulator's own distance field, at the point the displaced line actually passes
        through. Better information than a lane half-width, which cannot tell the wide side of a
        corridor from the narrow one: this is the clearance on the side the car really goes.

        And, when the env draws procedural obstacles, the nearest of those too. They are props and
        not grid, so the distance field alone reports a clear lane through a crate; a planner that
        believed it would drive into one. Taking the smaller of the two is what makes these
        baselines able to race a layout that stands on the racing line, which is the only way the
        policy ever meets an obstacle there.
        """
        base = self.base
        p = base.xy[tid, idx]                                          # (..., 2) reference point
        n = self._normal(tid, idx)
        d = torch.as_tensor(d, device=p.device, dtype=p.dtype)
        q = p + d[..., None] * n
        if props_only:
            # The distance field is not asked at all: the question is what is in the road that the
            # reference line does not already account for, and the walls are not that. Skipping the
            # lookup matters -- this path runs over every station and probe of `_blockage_ahead`,
            # for every planner, on every step.
            if self.props is None:
                return torch.full(q.shape[:-1], float("inf"), device=q.device, dtype=q.dtype)
            rows = torch.arange(q.shape[0], device=q.device)
            rows = rows.view(-1, *([1] * (q.dim() - 2))).expand(q.shape[:-1])
            return self.props.clearance(q.reshape(-1, 2), rows.reshape(-1)).view(q.shape[:-1])
        edt = self.track.sample_edt(q, tid)
        if self.props is None:
            return edt
        # Row b of the batch drives in layout b: one layout per env row, as `redraw` writes them.
        rows = torch.arange(q.shape[0], device=q.device)
        rows = rows.view(-1, *([1] * (q.dim() - 2))).expand(q.shape[:-1])
        prop = self.props.clearance(q.reshape(-1, 2), rows.reshape(-1)).view(edt.shape).to(edt.dtype)
        return torch.minimum(edt, prop)

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

    #: Lateral probes across the lane when looking for a prop blocking the line, and how far apart
    #: the stations ahead are sampled [m]. Half a body width in both directions, so a crate cannot
    #: sit between two probes and be missed.
    BLOCK_PROBES = (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9)
    BLOCK_STEP = 0.25

    def _blockage_ahead(self, state: torch.Tensor, tid: torch.Tensor, reach: torch.Tensor,
                        margin: float = 0.10):
        """(arc, lateral) of the nearest procedural prop standing in the road, per row.

        `inf` where the road is clear, which makes it lose every comparison against a real car and
        costs the callers no branch.

        Why a planner needs this at all: `_clearance_at` made these baselines able to *choose a
        side* around a prop, and that is not the same as noticing one. With no car nearby the
        spliner drives the reference line, and with a crate standing on it -- which is exactly what
        `procedural_raceline_corridor = "off"` arranges -- it drives into it. Measured: the
        opponents' termination rate went from 4.9 to 27.3 per km the moment the layouts were
        allowed onto the line.

        Upstream is on this side of the argument too. The ForzaETH detector reports *obstacles*,
        not cars; a planner that evades only the things with wheels is the narrower reading. So the
        blockage is returned in the same (arc, lateral) shape a car is, and the callers pick
        whichever is nearer and plan around it with the machinery they already have.
        """
        if self.props is None:
            far = torch.full((state.shape[0],), float("inf"), device=state.device,
                             dtype=state.dtype)
            return far, torch.zeros_like(far)
        base = self.base
        B = state.shape[0]
        dev, dt = state.device, state.dtype
        idx = base.project(state[:, :2], tid)[0]
        # `reach` is the caller's manoeuvre span, not its detection range: a crate further ahead
        # than the evasion itself reaches is not yet a thing to plan around, and every extra
        # station here is B x 7 more prop queries on every step of every race.
        K = max(2, int(float(reach.max().item()) / self.BLOCK_STEP))
        ds = base.ds[tid]                                              # [m] per reference index
        step = torch.arange(1, K + 1, device=dev, dtype=dt) * self.BLOCK_STEP     # (K,)
        j = (idx[:, None] + (step[None] / ds[:, None]).round().long()) % base.N   # (B, K)
        probes = torch.tensor(self.BLOCK_PROBES, device=dev, dtype=dt)            # (L,)
        L = probes.numel()
        tid_k = tid[:, None, None].expand(B, K, L)
        room = self._clearance_at(j[:, :, None].expand(B, K, L), tid_k,
                                  probes[None, None].expand(B, K, L), props_only=True)
        blocked = room < (self.half_width + margin)                    # (B, K, L)
        any_k = blocked.any(2)                                          # (B, K)
        # the first station in order that is blocked, and `inf` for a row with none
        first = any_k.to(torch.uint8).argmax(1)
        arc = step[first]
        arc = torch.where(any_k.any(1) & (step[first][None] <= reach[None]).squeeze(0),
                          arc, torch.full_like(arc, float("inf")))
        # where in the lane it is: the tightest probe at that station is where the crate stands
        ar = torch.arange(B, device=dev)
        lat = probes[room[ar, first].argmin(1)]
        return arc, torch.where(torch.isfinite(arc), lat, torch.zeros_like(lat))

    def _opponent_speed(self, state: torch.Tensor) -> torch.Tensor:
        """Speed of the car `_opponents` picked, per row."""
        ar = torch.arange(state.shape[0], device=state.device)
        return state[self.env.sim.other_idx[ar, self._nearest], 3].to(state.dtype)
