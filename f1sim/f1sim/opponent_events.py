"""Scripted behaviour events for the teacher-driven opponents of a race.

The raceline teacher drives one line at one speed scale and is blind to everything except the
follow-gap slowdown, so every opponent a policy meets presents the same problem: a slightly slower
car holding the racing line. Obstacle avoidance and overtaking are the policy's two weakest
benchmark dimensions, and widening `opp_speed_range` / `spawn_gap` moved neither -- a faster car on
the same line is the same lesson at a different speed.

What is missing is *behaviour*. This module gives each teacher-driven opponent a small per-car state
machine that, at a Poisson rate, drops it into one scripted event:

    brake   command k x its profile speed for d seconds (k ~ U(0, 0.5), d ~ U(0.5, 2.5))
    stop    command 0 for d seconds (d ~ U(1, 4)) -- the stalled car
    shift   track the raceline offset by o metres, ramped in over ~1 s, held, ramped out
            (|o| ~ U(0, 0.35), sign uniform) -- a lane change / blocking line
    weave   sinusoidal offset, amplitude ~ U(0.1, 0.25) m, period ~ U(2, 4) s

The machine is fully batched (B,) tensors -- no Python loop over cars -- and every draw comes from
the simulator's own seeded generator, so a seed reproduces the whole schedule. Two invariants the
rest of the env relies on:

* **Events only ever slow a car down.** `speed_scale` is in [0, 1], and `gym_env` applies it
  *before* the `opp_follow_gap` cap, so the final command is the minimum of the two: an opponent
  already braking for a car ahead never accelerates because an event asked it to.
* **The lateral offset can never reach a wall.** `offset_limit` is the free space at each raceline
  point minus the car's half-width and a margin, precomputed once from the track's distance field;
  the teacher clamps the commanded offset against it at the car's own raceline index.

With `opp_events=()` nothing here runs and nothing is drawn from the generator, so an unflagged run
is bit-identical to the code before this module existed.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import math

import torch

#: Event ids as reported in `info["opp_event"]["id"]`. 0 is "no event"; the rest are these names'
#: 1-based positions, fixed for the life of the feature so a logged id keeps its meaning.
EVENT_NAMES = ("brake", "stop", "shift", "weave")
EVENT_ID = {name: i + 1 for i, name in enumerate(EVENT_NAMES)}
NO_EVENT = 0


def parse_events(events) -> tuple:
    """Normalize an `opp_events` value (tuple/list, or a comma-separated string) and check the names."""
    if events is None:
        return ()
    if isinstance(events, str):
        events = [e for e in events.replace(" ", "").split(",") if e]
    names = tuple(str(e) for e in events)
    bad = [n for n in names if n not in EVENT_ID]
    if bad:
        raise ValueError(f"unknown opponent event(s) {bad}: choose from {list(EVENT_NAMES)}")
    seen = set()
    return tuple(n for n in names if not (n in seen or seen.add(n)))


def raceline_offset_limit(teacher, track, car_half_width: float, margin: float) -> torch.Tensor:
    """(T, N) largest |lateral offset| each raceline point tolerates before the car touches a wall.

    The track's distance field at a raceline point is the free space around it in *every* direction,
    so `clearance - half width - margin` bounds the offset whichever way it is applied -- the
    raceline is not centred in the lane and a per-side bound would need the lane's two edges, which
    the field does not separate. Conservative by construction, which is the right side to be on: the
    cost of clamping a 0.35 m shift to 0.2 m in a narrow section is a smaller lane change, and the
    cost of not clamping it is an opponent parked in the wall.
    """
    xy = teacher.xy.to(track.device)                                   # (T, N, 2)
    T, N, _ = xy.shape
    tid = torch.arange(T, device=track.device)[:, None].expand(T, N)
    clearance = track.sample_edt(xy, tid)
    return (clearance - car_half_width - margin).clamp_min(0.0)


class OpponentEvents:
    """Per-car scripted-event state machine for the teacher-driven cars of a vectorized env.

    Owns six (B,) tensors and is stepped once per env step. `gate` selects the cars it may act on
    (teacher-driven, never the learner); the learner's rows are advanced never and read never.
    """

    def __init__(self, B: int, device, ecfg, control_dt: float, gen: torch.Generator):
        self.B, self.device, self.dt, self.gen = B, torch.device(device), float(control_dt), gen
        self.names = parse_events(ecfg.opp_events)
        self.rate = float(ecfg.opp_event_rate)
        self.enabled = bool(self.names) and self.rate > 0.0
        self.ecfg = ecfg
        z = lambda: torch.zeros(B, device=self.device)
        self.kind = torch.zeros(B, dtype=torch.long, device=self.device)   # 0 = idle, else EVENT_ID
        self.t = z()             # seconds elapsed inside the current event
        self.dur = z()           # its total duration
        self.p0 = z()            # brake: speed scale k. shift: signed offset o [m]. weave: amplitude [m]
        self.p1 = z()            # weave: period [s]. shift: ramp time [s]
        self.gate = torch.zeros(B, dtype=torch.bool, device=self.device)
        # id of the event each configured name maps to, as a tensor to index by the drawn slot
        self.kind_lut = torch.tensor([EVENT_ID[n] for n in self.names] or [0],
                                     dtype=torch.long, device=self.device)
        self.offset_limit: Optional[torch.Tensor] = None
        # P(an idle opponent starts an event this step). `opp_event_rate` is events per 10 s, so the
        # per-step probability is rate * dt / 10. Events do not overlap, so the realized rate is
        # this times the idle fraction -- at rate 1.0 with ~1.5 s events that is ~0.87 per 10 s.
        self.p_start = min(1.0, self.rate * self.dt / 10.0)

    # ------------------------------------------------------------------ lifecycle
    def set_gate(self, gate: torch.Tensor):
        """Which cars are teacher-driven this episode (recomputed at every race reset in mixed mode)."""
        self.gate = gate

    def reset(self, ids: torch.Tensor):
        """Clear the event state of respawning cars. Draws nothing: the next event comes from the
        per-step trigger, so a reset costs the generator nothing whether the feature is on or off."""
        if ids.numel() == 0:
            return
        self.kind[ids] = NO_EVENT
        self.t[ids] = 0.0; self.dur[ids] = 0.0; self.p0[ids] = 0.0; self.p1[ids] = 0.0

    # ------------------------------------------------------------------ the machine
    def step(self):
        """Advance every opponent's event by one control step: expire, then trigger, then sample."""
        if not self.enabled:
            return
        g = self.gate
        active = g & (self.kind != NO_EVENT)
        self.t = torch.where(active, self.t + self.dt, self.t)
        expired = active & (self.t >= self.dur)
        self.kind = torch.where(expired, torch.zeros_like(self.kind), self.kind)
        # Every draw is full-batch, so the generator advances by the same amount on every step
        # regardless of how many cars happen to be idle: the schedule a seed produces does not
        # depend on the order envs finish their events in.
        u = torch.rand(6, self.B, device=self.device, generator=self.gen)
        start = g & (self.kind == NO_EVENT) & (u[0] < self.p_start)
        slot = (u[1] * len(self.names)).long().clamp_(0, len(self.names) - 1)
        kind = self.kind_lut[slot]
        dur, p0, p1 = self._sample(kind, u)
        self.kind = torch.where(start, kind, self.kind)
        self.t = torch.where(start, torch.zeros_like(self.t), self.t)
        self.dur = torch.where(start, dur, self.dur)
        self.p0 = torch.where(start, p0, self.p0)
        self.p1 = torch.where(start, p1, self.p1)

    def _sample(self, kind: torch.Tensor, u: torch.Tensor):
        """(duration, p0, p1) for a freshly drawn event of each kind, from the shared uniforms."""
        e = self.ecfg
        lerp = lambda rng, r: rng[0] + (rng[1] - rng[0]) * r
        z = torch.zeros(self.B, device=self.device)
        dur, p0, p1 = z.clone(), z.clone(), z.clone()
        is_brake = kind == EVENT_ID["brake"]
        dur = torch.where(is_brake, lerp(e.opp_brake_time_range, u[2]), dur)
        p0 = torch.where(is_brake, lerp(e.opp_brake_scale_range, u[3]), p0)
        is_stop = kind == EVENT_ID["stop"]
        dur = torch.where(is_stop, lerp(e.opp_stop_time_range, u[2]), dur)
        # p0 stays 0 for a stop: the speed multiplier *is* zero.
        is_shift = kind == EVENT_ID["shift"]
        ramp = float(e.opp_shift_ramp)
        sign = torch.where(u[4] < 0.5, -torch.ones_like(z), torch.ones_like(z))
        dur = torch.where(is_shift, lerp(e.opp_shift_hold_range, u[2]) + 2 * ramp, dur)
        p0 = torch.where(is_shift, sign * lerp(e.opp_shift_offset_range, u[3]), p0)
        p1 = torch.where(is_shift, torch.full_like(z, ramp), p1)
        is_weave = kind == EVENT_ID["weave"]
        dur = torch.where(is_weave, lerp(e.opp_weave_time_range, u[2]), dur)
        p0 = torch.where(is_weave, sign * lerp(e.opp_weave_amp_range, u[3]), p0)
        p1 = torch.where(is_weave, lerp(e.opp_weave_period_range, u[5]).clamp_min(0.1), p1)
        return dur, p0, p1

    # ------------------------------------------------------------------ what the env reads
    def speed_scale(self) -> Optional[torch.Tensor]:
        """(B,) multiplier on the opponent's commanded speed, 1.0 where no event is slowing it.

        Never above 1: `gym_env` takes the minimum of this and the follow-gap cap, which is only a
        cap on a car that is *already* slowing for a car ahead. A multiplier that could exceed 1
        would let an event undo that slowdown and drive the opponent into the car it is following.
        """
        if not self.enabled:
            return None
        s = torch.ones(self.B, device=self.device)
        s = torch.where(self.gate & (self.kind == EVENT_ID["brake"]), self.p0.clamp(0.0, 1.0), s)
        s = torch.where(self.gate & (self.kind == EVENT_ID["stop"]), torch.zeros_like(s), s)
        return s

    def lateral_offset(self) -> Optional[torch.Tensor]:
        """(B,) metres left of the raceline the opponent should be tracking right now (0 = on it)."""
        if not self.enabled:
            return None
        o = torch.zeros(self.B, device=self.device)
        ramp = self.p1.clamp_min(1e-3)
        # 0 -> 1 over `ramp`, 1 while held, 1 -> 0 over the last `ramp`: a lane change, not a
        # teleport. The car has to be able to follow the line it is given, and pure pursuit on a
        # target that jumps 0.35 m sideways in one step asks for a step steer input.
        up = (self.t / ramp).clamp(0.0, 1.0)
        down = ((self.dur - self.t) / ramp).clamp(0.0, 1.0)
        o = torch.where(self.gate & (self.kind == EVENT_ID["shift"]), self.p0 * torch.minimum(up, down), o)
        phase = 2 * math.pi * self.t / self.p1.clamp_min(1e-3)
        o = torch.where(self.gate & (self.kind == EVENT_ID["weave"]), self.p0 * torch.sin(phase), o)
        return o

    def info(self) -> Dict[str, torch.Tensor]:
        """Per-step view for a viewer or logger: which event each car is running and how long is left.

        Reported for every env row; a car that is not a teacher-driven opponent reads id 0.
        """
        idle = ~self.gate | (self.kind == NO_EVENT)
        return {"id": torch.where(idle, torch.zeros_like(self.kind), self.kind),
                "time_left": torch.where(idle, torch.zeros_like(self.t), (self.dur - self.t).clamp_min(0.0)),
                "offset": self.lateral_offset() if self.enabled else torch.zeros(self.B, device=self.device)}

    # ------------------------------------------------------------------ diagnostics
    def counts(self) -> Dict[str, int]:
        """How many cars are currently running each configured event (host sync: demos and tests)."""
        return {n: int((self.gate & (self.kind == EVENT_ID[n])).sum()) for n in self.names}


def describe(names: Sequence[str], rate: float) -> str:
    return f"opponent events {list(names)} at {rate:g} per opponent per 10 s" if names else "opponent events off"
