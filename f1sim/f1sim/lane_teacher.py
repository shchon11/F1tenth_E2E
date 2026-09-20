"""A fixed-lane switching planner, the other family of F1TENTH overtaking stack.

Why this is here and what it is honestly called
-----------------------------------------------
The request this answers was to reproduce UNIST's **UNICORN** alongside ForzaETH. That cannot be
done at the same fidelity and the difference matters, so it is stated rather than glossed: the
ForzaETH race stack is open source and its `spliner` planner is reproduced from its own code and
paper (`f1sim.spliner_teacher`). UNICORN is not -- what is public is a competition page naming
its components: Cartographer localisation, a rule-based state machine, a **Lane Change Planner**,
Frenet-frame tracking, and an Advanced Pure Pursuit / L1 controller. There are no parameters to
read off and no source to check against.

So this module does not claim to be UNICORN. It implements the thing that composition *is*, and
which several F1TENTH teams race: instead of drawing a fresh curve around each obstacle, the track
carries a small fixed set of lanes offset from the racing line, and the planner picks the cheapest
lane that is clear, transitioning at a bounded rate. That is a genuinely different comparison
point from the spline family -- discrete choice with hysteresis rather than a continuous curve --
which is why it is worth having even though it is not a port of anyone's code.

The numbers are ours and are marked as ours. `LANES` reuses the offsets `f1sim.interactive_teacher`
already chose from this project's own measured lane widths, so the two are comparable; nothing
here is presented as a published value the way `f1sim.spliner_teacher`'s constants are.

The behaviour
-------------
Each step, for every lane:

* **blocked** if the car this row is planning against sits within ``LANE_BLOCK_W`` of it, and is
  ahead within ``LOOKAHEAD``;
* **drivable** if the distance field at the lane's own point leaves the body ``LANE_MARGIN`` of
  room -- asked at the opponent's station, where the pass happens, and at the car's own;
* **cost** = blocked penalty + ``SWITCH_COST`` per metre away from the lane currently held +
  ``OFFLINE_COST`` per metre away from the racing line.

The cheapest lane wins, but only by ``HYSTERESIS``: a planner that re-argmins every step weaves
between two nearly equal lanes at the step rate, which is the failure mode this family has.
The commanded offset then moves toward the chosen lane at no more than ``LANE_RATE`` metres per
second, because the lane is a decision and the line to it is a manoeuvre.

If no lane is clear the state is TRAILING and the line goes back to the racing line; the speed is
left to the env's follow cap, exactly as in `f1sim.spliner_teacher` and for the same reason --
two controllers both deciding how hard to brake would fight.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from .opponent_planner import FrenetOpponentPlanner
from .teacher import RacelineTeacher

#: Lateral offsets of the lanes [m left of the racing line]. OURS, not a published set: these are
#: the offsets `interactive_teacher.DEFAULT_OFFSETS` chose from this project's measured half-lane
#: widths (about 0.70 m on the traffic family's floors), so the two planners span the same road.
LANES = (-0.6, -0.3, 0.0, 0.3, 0.6)

#: A lane is blocked if the other car is within this of it [m]. Two bodies are 0.31 m wide, so
#: half a body plus half a body is 0.31; the rest is the clearance a driver would want.
LANE_BLOCK_W = 0.45

#: Room the body needs beyond its own half-width for a lane to be drivable at all [m].
LANE_MARGIN = 0.15

#: How far ahead a car counts as being in the way [m].
LOOKAHEAD = 10.0

#: How far BEHIND a car still counts, before the speed scaling [m]. A lane change is finished when
#: the other car is clear behind, not when its arc goes negative: the bodies are 0.58 m long, so at
#: 1 m the two cars have barely separated and a planner that stopped counting there would start
#: back across the road while still alongside -- the contact the whole family exists to avoid. At
#: 4 m/s three metres is 0.75 s, against the 0.375 s the widest lane change takes at `LANE_RATE`.
#: Scaled by `_stretch` like every other distance here.
REAR_CLEAR = 3.0

#: Cost per metre of lane change, and per metre of distance from the racing line. The first is
#: what makes the planner keep the lane it has; the second is what makes it come back.
SWITCH_COST = 0.6
OFFLINE_COST = 1.0

#: How much cheaper another lane has to be before the planner will move to it. Without it the
#: argmin flips between two nearly equal lanes at the step rate, which is this family's own
#: characteristic failure and not something worth reproducing faithfully.
#:
#: It is bounded from above, and the bound is not obvious: coming back from lane `d` to the racing
#: line saves `offline_cost * |d|` and costs `switch_cost * |d|`, so the most a return can ever be
#: worth is `|d| * (offline_cost - switch_cost)` -- 0.12 for the innermost lane here. A hysteresis
#: above that makes every lane absorbing, and the car simply never comes back. The constructor
#: checks it, because the symptom is a baseline that drives a permanent 0.6 m offset and looks
#: like a tuning problem rather than an arithmetic one.
HYSTERESIS = 0.05

#: Fastest the commanded offset moves toward the chosen lane [m/s]. The lane is a decision; the
#: line to it is a manoeuvre, and a step change in commanded offset is a step change in the
#: reference the tracker chases.
LANE_RATE = 1.6

#: State machine codes, the same three the spline family reports, so a census can put the two
#: baselines in one table.
RACING, TRAILING, CHANGING = 0, 1, 2


class LaneSwitchTeacher(FrenetOpponentPlanner):
    """Fixed-lane switching with hysteresis and a bounded transition rate.

    Not a port of any one team's code -- see the module docstring, which says exactly what is
    reproduced and what is inferred.
    """

    STATE_NAMES = ("racing", "trailing", "changing")

    def __init__(self, base: RacelineTeacher, env=None, *,
                 lanes: Sequence[float] = LANES,
                 block_w: float = LANE_BLOCK_W,
                 margin: float = LANE_MARGIN,
                 lookahead: float = LOOKAHEAD,
                 rear_clear: float = REAR_CLEAR,
                 switch_cost: float = SWITCH_COST,
                 offline_cost: float = OFFLINE_COST,
                 hysteresis: float = HYSTERESIS,
                 lane_rate: float = LANE_RATE,
                 dt: float = 0.01):
        super().__init__(base, env=None)          # attach last: the fields below are its inputs
        lanes = tuple(float(x) for x in lanes)
        if 0.0 not in lanes:
            raise ValueError(f"lanes {list(lanes)} do not contain the racing line (0.0): a lane "
                             f"planner with nowhere to come back to is a permanent offset")
        self.lanes = torch.tensor(lanes, dtype=torch.float32, device=self.device)
        self.block_w = float(block_w)
        self.margin = float(margin)
        self.lookahead = float(lookahead)
        self.rear_clear = float(rear_clear)
        self.switch_cost = float(switch_cost)
        self.offline_cost = float(offline_cost)
        self.hysteresis = float(hysteresis)
        self.lane_rate = float(lane_rate)
        self.dt = float(dt)
        # The planner has to be able to come back. See HYSTERESIS: the most a return to the racing
        # line can ever be worth is the innermost lane's width times the cost gap, and a
        # hysteresis above it makes every lane absorbing -- silently, as a permanent offset.
        inner = min(abs(x) for x in lanes if x != 0.0)
        worth = inner * (self.offline_cost - self.switch_cost)
        if worth <= self.hysteresis:
            raise ValueError(
                f"hysteresis {self.hysteresis} leaves no way back to the racing line: returning "
                f"from the innermost lane ({inner} m) is worth at most {worth:.3f} "
                f"(offline_cost {self.offline_cost} - switch_cost {self.switch_cost}), so every "
                f"lane would be absorbing and the car would hold an offset for ever")
        #: Which lane each row currently holds, and the offset actually commanded last step. Both
        #: are state: the first is what `switch_cost` and `hysteresis` are measured against, the
        #: second is what `lane_rate` limits the change of.
        self._lane: Optional[torch.Tensor] = None
        self._offset: Optional[torch.Tensor] = None
        if env is not None:
            self.attach(env)

    def attach(self, env):
        out = super().attach(env)
        # The rate limit is per second, and the step it is applied over is the env's.
        self.dt = float(getattr(env.cfg.sim, "dt", self.dt))
        return out

    @torch.no_grad()
    def decide(self, state: torch.Tensor, tid: Optional[torch.Tensor] = None
               ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """(offset [m left of the racing line], state code, speed scale) per row."""
        B = state.shape[0]
        dev, dt = state.device, state.dtype
        zero = torch.zeros(B, device=dev, dtype=dt)
        one = torch.ones(B, device=dev, dtype=dt)
        mode = torch.full((B,), RACING, device=dev, dtype=torch.long)
        if self._lane is None or self._lane.shape != (B,) or self._lane.device != dev:
            on_line = int((self.lanes == 0).nonzero()[0, 0])
            self._lane = torch.full((B,), on_line, device=dev, dtype=torch.long)
            self._offset = torch.zeros(B, device=dev, dtype=dt)

        # A car is in the way until it is clear behind -- see REAR_CLEAR -- and the distance grows
        # with speed, like every other distance in these planners.
        stretch = self._stretch(state)
        rear = -self.rear_clear * stretch
        seen = self._opponents(state, tid, rear)
        if seen is None:
            self._lane = torch.full_like(self._lane, int((self.lanes == 0).nonzero()[0, 0]))
            self._offset = torch.zeros_like(self._offset)
            self.last_state, self.last_offset = mode, zero
            return None, mode, one
        gap, d_opp, idx_opp, tid_b, ego_d = seen

        # A prop standing in the road blocks a lane exactly the way a car does, and it is the case
        # this planner would otherwise drive straight into: with nobody to pass it holds the racing
        # line, and `procedural_raceline_corridor = "off"` puts crates on it.
        b_gap, b_d = self._blockage_ahead(state, tid_b, 2.0 * self.rear_clear * stretch)
        take_prop = b_gap < gap.clamp_min(0.0)
        gap = torch.where(take_prop, b_gap, gap)
        d_opp = torch.where(take_prop, b_d, d_opp)
        idx_opp = torch.where(take_prop,
                              (self.base.project(state[:, :2], tid_b)[0]
                               + (b_gap.nan_to_num(posinf=0.0) / self.base.ds[tid_b]).round().long()
                               ) % self.base.N, idx_opp)
        L = self.lanes.numel()
        lanes = self.lanes.to(dt)                                       # (L,)

        # Blocked: the other car sits in this lane, and is close enough ahead to matter.
        near = (gap < self.lookahead) & (gap > rear)
        blocked = ((d_opp[:, None] - lanes[None]).abs() < self.block_w) & near[:, None]

        # Drivable: the body fits, at the opponent's station and at our own. Two probes rather
        # than a sweep -- the lane is straight in the line's own frame, so the places it can be
        # too narrow are where the road is, and those are the two stations that matter here.
        idx_ego = self.base.project(state[:, :2], tid_b)[0]
        need = self.half_width + self.margin
        room_there = self._clearance_at(idx_opp[:, None].expand(B, L),
                                        tid_b[:, None].expand(B, L), lanes[None].expand(B, L))
        room_here = self._clearance_at(idx_ego[:, None].expand(B, L),
                                       tid_b[:, None].expand(B, L), lanes[None].expand(B, L))
        fits = (room_there >= need) & (room_here >= need)

        # Cost. The blocked and unusable penalties are large and unequal, so that "somebody is in
        # it" and "it is a wall" are not the same lane to a tie-break.
        held = lanes[self._lane]                                        # (B,)
        cost = (self.switch_cost * (lanes[None] - held[:, None]).abs()
                + self.offline_cost * lanes[None].abs())
        cost = cost + torch.where(blocked, torch.full_like(cost, 50.0), torch.zeros_like(cost))
        cost = cost + torch.where(fits, torch.zeros_like(cost), torch.full_like(cost, 200.0))

        # Hysteresis: the lane held wins ties and anything closer than `hysteresis`. Without it
        # the argmin flips between two nearly equal lanes at the step rate.
        best = cost.argmin(1)
        ar = torch.arange(B, device=dev)
        keep = cost[ar, self._lane] <= cost[ar, best] + self.hysteresis
        lane = torch.where(keep, self._lane, best)

        # Trailing: every lane a car could use is blocked. This planner has no answer to that --
        # a fixed-lane stack either has a gap or it does not -- so the line goes back to the racing
        # line and the speed is left to the env's follow cap, which is already holding a gap behind
        # the car ahead for every teacher-driven row. Two controllers braking would fight.
        usable = (~blocked) & fits
        trailing = near & ~usable.any(1)
        on_line = int((self.lanes == 0).nonzero()[0, 0])
        lane = torch.where(trailing, torch.full_like(lane, on_line), lane)
        self._lane = lane

        # The line to the chosen lane, rate limited: the lane is a decision, the line to it is a
        # manoeuvre, and a step change in the commanded offset is a step change in the reference
        # the tracker chases. Trailing gets the same limit, so a car that gave up on a pass comes
        # back across the road at the speed it went out.
        step = self.lane_rate * self.dt
        self._offset = self._offset + (lanes[lane] - self._offset).clamp(-step, step)

        mode = torch.where(trailing, torch.full_like(mode, TRAILING), mode)
        mode = torch.where(near & ~trailing & (self._offset.abs() > 1e-3),
                           torch.full_like(mode, CHANGING), mode)
        self.last_state, self.last_offset = mode, self._offset
        return self._offset, mode, one

    def reset_rows(self, rows: torch.Tensor) -> None:
        """A new race starts on the racing line, not in the lane the last one ended in."""
        if self._lane is None:
            return
        self._lane[rows] = int((self.lanes == 0).nonzero()[0, 0])
        self._offset[rows] = 0.0
