"""Behaviour for the teacher-driven opponents of a race: scripted events, and reactive driving.

The raceline teacher drives one line at one speed scale and is blind to everything except the
follow-gap slowdown, so every opponent a policy meets presents the same problem: a slightly slower
car holding the racing line. Obstacle avoidance and overtaking are the policy's two weakest
benchmark dimensions, and widening `opp_speed_range` / `spawn_gap` moved neither -- a faster car on
the same line is the same lesson at a different speed.

What is missing is *behaviour*, and there are two kinds of it. This module has both, named in the
same `opp_events` list and each off by default.

**Timed events** drop a car into a scripted manoeuvre at a Poisson rate (`opp_event_rate`), on its
own, with no idea the learner exists:

    brake   command k x its profile speed for d seconds (k ~ U(0, 0.5), d ~ U(0.5, 2.5))
    stop    command 0 for d seconds (d ~ U(1, 4)) -- the stalled car
    shift   track the raceline offset by o metres, ramped in over ~1 s, held, ramped out
            (|o| ~ U(0, 0.35), sign uniform) -- a lane change / blocking line
    weave   sinusoidal offset, amplitude ~ U(0.1, 0.25) m, period ~ U(2, 4) s

**Reactive behaviours** are dispositions drawn per car per race, each with its own probability, and
they read the learner's position relative to the opponent *every step*. A timer cannot produce
them: they are the situations that only exist because there is another car:

    defend     a learner within `overtake_range` behind moves the opponent's lateral offset toward
               the side the learner is coming down, harder the closer it gets
    yield      a learner alongside moves the opponent away from it
    line       an out-in or in-out offset profile through each corner, mode and amplitude drawn per
               corner: the opponent is simply not on the line the learner's model of it assumes
    oblivious  the follow-gap slowdown is switched off for this car -- it does not brake for a car
               ahead or beside it. This is the car that rams, and the only one in the population
               that reproduces the "the car I just passed drove into me" situation on purpose
               rather than as a teacher bug.

Measured, the traffic failures are side-by-side contacts at +1.5 m/s closing and walls hit during a
pass (`docs/research/failure-attribution-2026-09-13.md` section 5) -- lateral room, from the side.
The reactive behaviours are the ones that put the learner *in* that situation on purpose.

The machine is fully batched (B,) tensors -- no Python loop over cars -- and every draw comes from
the simulator's own seeded generator, so a seed reproduces the whole schedule. Three invariants the
rest of the env relies on:

* **Events only ever slow a car down.** `speed_scale` is in [0, 1], and `gym_env` applies it
  *before* the `opp_follow_gap` cap, so the final command is the minimum of the two: an opponent
  already braking for a car ahead never accelerates because an event asked it to. `oblivious` is
  the deliberate exception and it works the other way round -- it removes the cap rather than
  raising the command through it.
* **The lateral offset can never reach a wall.** `offset_limit` is the free space at each raceline
  point minus the car's half-width and a margin, precomputed once from the track's distance field;
  the teacher clamps the commanded offset -- scripted plus reactive, they are summed -- against it
  at the car's own raceline index. The reactive part is additionally capped at `opp_react_max` and
  rate-limited to `opp_react_slew` m/s, because a target that jumps sideways asks the pure-pursuit
  teacher for a step steer input.
* **With `opp_events=()` nothing here runs** and nothing is drawn from the generator, so an
  unflagged run is bit-identical to the code before this module existed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import math

import numpy as np
import torch

#: Timed event names, in id order. Fixed for the life of the feature so a logged id keeps its
#: meaning; the reactive names are appended after them for the same reason.
EVENT_NAMES = ("brake", "stop", "shift", "weave")

#: Reactive behaviour names, in bit order.
REACTIVE_NAMES = ("defend", "yield", "line", "oblivious")

ALL_EVENT_NAMES = EVENT_NAMES + REACTIVE_NAMES

#: Event ids as reported in `info["opp_event"]["id"]`. 0 is "no event"; the rest are these names'
#: 1-based positions in `ALL_EVENT_NAMES`. Only a timed event ever appears in `id` -- a reactive
#: behaviour is not a state the car is "in" to the exclusion of others, so those are reported as a
#: bitmask in `info["opp_event"]["react"]` instead, under `REACTIVE_BIT`.
EVENT_ID = {name: i + 1 for i, name in enumerate(ALL_EVENT_NAMES)}
NO_EVENT = 0

#: Bit of each reactive behaviour in the `disposition` / `react` masks.
REACTIVE_BIT = {name: 1 << i for i, name in enumerate(REACTIVE_NAMES)}

#: `opp_<name>_prob` is the per-race probability a teacher-driven car is given that disposition.
REACTIVE_PROB_FIELD = {name: f"opp_{name}_prob" for name in REACTIVE_NAMES}


def parse_events(events) -> tuple:
    """Normalize an `opp_events` value (tuple/list, or a comma-separated string) and check the names."""
    if events is None:
        return ()
    if isinstance(events, str):
        events = [e for e in events.replace(" ", "").split(",") if e]
    names = tuple(str(e) for e in events)
    bad = [n for n in names if n not in EVENT_ID]
    if bad:
        raise ValueError(f"unknown opponent event(s) {bad}: choose from {list(ALL_EVENT_NAMES)}")
    seen = set()
    return tuple(n for n in names if not (n in seen or seen.add(n)))


def split_events(events) -> tuple:
    """(timed names, reactive names) of a parsed or unparsed `opp_events` value, in id order."""
    names = parse_events(events)
    return (tuple(n for n in names if n in EVENT_NAMES),
            tuple(n for n in names if n in REACTIVE_NAMES))


def reactive_rates(ecfg) -> Dict[str, float]:
    """The per-race probability configured for each reactive behaviour that is named at all."""
    _, react = split_events(ecfg.opp_events)
    return {n: float(getattr(ecfg, REACTIVE_PROB_FIELD[n])) for n in react}


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


def _circular_runs(mask: np.ndarray):
    """(start, length) of every contiguous True run of a circular mask. One run if all True."""
    n = mask.shape[0]
    if not mask.any():
        return []
    if mask.all():
        return [(0, n)]
    # rotate so index 0 starts a run, then the runs are plain runs
    start0 = int(np.nonzero(mask & ~np.roll(mask, 1))[0][0])
    rolled = np.roll(mask, -start0)
    edges = np.diff(np.concatenate([[0], rolled.view(np.int8), [0]]))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0]
    return [((int(s) + start0) % n, int(e - s)) for s, e in zip(starts, ends)]


def raceline_corners(teacher, kappa_min: float, min_arc_m: float, smooth_m: float) -> Dict[str, torch.Tensor]:
    """Which corner each raceline point belongs to, how far through it, and which way it turns.

    Built once per track set, from the raceline's own curvature: a run of points whose smoothed
    |curvature| is above `kappa_min` and which is at least `min_arc_m` long is a corner. Returns
    three (T, N) tensors:

        id     corner index along the lap, -1 on a straight
        phase  0 -> 1 across the corner (entry to exit)
        sign   +1 where the corner turns left, -1 right: the *inside* of the corner, as a lateral
               offset sign in the teacher's convention (offset is metres left of the line)

    The smoothing matters more than the threshold: the raceline's discrete curvature is noisy at a
    2 cm point spacing, and without it a single corner comes apart into a dozen one-point "corners"
    whose phase sweeps are each a few centimetres long.
    """
    kap = teacher.kappa.detach().cpu().numpy()                          # (T, N), left +
    ds = teacher.ds.detach().cpu().numpy()
    T, N = kap.shape
    cid = np.full((T, N), -1, np.int64)
    phase = np.zeros((T, N), np.float32)
    sign = np.zeros((T, N), np.float32)
    for t in range(T):
        step = max(float(ds[t]), 1e-6)
        w = max(1, int(round(smooth_m / step)))
        k = np.convolve(np.concatenate([kap[t][-w:], kap[t], kap[t][:w]]),
                        np.ones(w) / w, mode="same")[w:w + N]
        runs = _circular_runs(np.abs(k) > kappa_min)
        n = 0
        for start, length in runs:
            if length * step < min_arc_m:
                continue
            idx = (start + np.arange(length)) % N
            cid[t, idx] = n
            phase[t, idx] = (np.arange(length) + 0.5) / length
            s = float(np.sign(k[idx].sum()))
            sign[t, idx] = s if s != 0.0 else 1.0
            n += 1
    dev = teacher.xy.device
    return {"id": torch.as_tensor(cid, device=dev),
            "phase": torch.as_tensor(phase, device=dev),
            "sign": torch.as_tensor(sign, device=dev)}


@dataclass
class LearnerView:
    """Where the policy-driven cars are, seen from every car of the batch, for one step.

    Built by `gym_env._learner_view()` and consumed by `OpponentEvents.step_reactive`. Everything
    is (B, C) over the C other cars of each race, in `sim.other_idx` order, except `rl_idx`:

        gap      signed arc to that car, + ahead of me (`gym_env.signed_gaps`)
        lon      body-frame longitudinal offset to it, + ahead of me
        lat      body-frame lateral offset, + to my left
        closing  line-of-sight closing speed, + while the gap shrinks
        learner  that car is driven by the policy under training
        rl_idx   (B,) raceline index of each car, or None when no behaviour needs it
        tid      (B,) track id of each car, needed with `rl_idx` to index the corner tables

    A view rather than the raw state because the reactive behaviours are *geometric*: what they
    need is "is a learner behind me, and which side is it on", and computing that twice -- once
    here and once in the census that has to verify it happened -- is how two definitions of
    "alongside" start disagreeing.
    """
    gap: torch.Tensor
    lon: torch.Tensor
    lat: torch.Tensor
    closing: torch.Tensor
    learner: torch.Tensor
    rl_idx: Optional[torch.Tensor] = None
    tid: Optional[torch.Tensor] = None


class OpponentEvents:
    """Per-car behaviour state for the teacher-driven cars of a vectorized env.

    Two layers, stepped once per env step: `step()` advances the timed state machine and `step_
    reactive(view)` the reactive one. `gate` selects the cars either may act on (teacher-driven,
    never the learner); the learner's rows are advanced never and read never.
    """

    def __init__(self, B: int, device, ecfg, control_dt: float, gen: torch.Generator):
        self.B, self.device, self.dt, self.gen = B, torch.device(device), float(control_dt), gen
        self.names = parse_events(ecfg.opp_events)
        self.timed, self.react = split_events(self.names)
        self.rate = float(ecfg.opp_event_rate)
        self.probs = {n: float(getattr(ecfg, REACTIVE_PROB_FIELD[n])) for n in self.react}
        #: Only behaviours with a positive probability are on: a named one at probability 0 would
        #: silently be the unflagged run, which is what `learn.opponent_config` refuses at the flag.
        self.react = tuple(n for n in self.react if self.probs[n] > 0.0)
        self.timed_on = bool(self.timed) and self.rate > 0.0
        self.react_on = bool(self.react)
        self.enabled = self.timed_on or self.react_on
        self.ecfg = ecfg
        z = lambda: torch.zeros(B, device=self.device)
        self.kind = torch.zeros(B, dtype=torch.long, device=self.device)   # 0 = idle, else EVENT_ID
        self.t = z()             # seconds elapsed inside the current event
        self.dur = z()           # its total duration
        self.p0 = z()            # brake: speed scale k. shift: signed offset o [m]. weave: amplitude [m]
        self.p1 = z()            # weave: period [s]. shift: ramp time [s]
        self.gate = torch.zeros(B, dtype=torch.bool, device=self.device)
        # id of the event each configured timed name maps to, as a tensor to index by the drawn slot
        self.kind_lut = torch.tensor([EVENT_ID[n] for n in self.timed] or [0],
                                     dtype=torch.long, device=self.device)
        # ---- reactive state. `disp` is the bitmask of dispositions this car was given for this
        # race; `react_now` the bitmask of the ones actually acting this step (what a census reads).
        self.disp = torch.zeros(B, dtype=torch.long, device=self.device)
        self.react_now = torch.zeros(B, dtype=torch.long, device=self.device)
        self.amp_defend, self.amp_yield = z(), z()
        self.react_off = z()             # the reactive lateral offset the car is holding [m]
        self.corner = torch.full((B,), -1, dtype=torch.long, device=self.device)
        self.corner_mode = torch.ones(B, device=self.device)    # +1 out-in, -1 in-out
        self.corner_amp = z()
        self.corners: Optional[Dict[str, torch.Tensor]] = None   # set by the env when `line` is on
        self.offset_limit: Optional[torch.Tensor] = None
        # P(an idle opponent starts an event this step). `opp_event_rate` is events per 10 s, so the
        # per-step probability is rate * dt / 10. Events do not overlap, so the realized rate is
        # this times the idle fraction -- at rate 1.0 with ~1.5 s events that is ~0.87 per 10 s.
        self.p_start = min(1.0, self.rate * self.dt / 10.0)

    # ------------------------------------------------------------------ lifecycle
    @property
    def needs_corners(self) -> bool:
        """Whether `line` is on, i.e. whether the corner tables and the raceline index are needed."""
        return "line" in self.react

    def set_gate(self, gate: torch.Tensor):
        """Which cars are teacher-driven this episode (recomputed at every race reset in mixed mode)."""
        self.gate = gate

    def reset(self, ids: torch.Tensor):
        """Clear the event state of respawning cars, and draw their reactive dispositions.

        With no reactive behaviour configured this draws nothing: the next timed event comes from
        the per-step trigger, so a reset costs the generator nothing whether that feature is on or
        off. With one configured it draws `4 + 2` uniforms per resetting car -- a disposition is a
        property of the car in *this* race, so it has to be redrawn when the car is replaced.
        """
        if ids.numel() == 0:
            return
        self.kind[ids] = NO_EVENT
        self.t[ids] = 0.0; self.dur[ids] = 0.0; self.p0[ids] = 0.0; self.p1[ids] = 0.0
        if not self.react_on:
            return
        n = ids.numel()
        u = torch.rand(len(REACTIVE_NAMES) + 2, n, device=self.device, generator=self.gen)
        e = self.ecfg
        disp = torch.zeros(n, dtype=torch.long, device=self.device)
        for i, name in enumerate(REACTIVE_NAMES):
            if name not in self.react:
                continue
            disp = disp | (u[i] < self.probs[name]).long() * REACTIVE_BIT[name]
        self.disp[ids] = disp
        lerp = lambda rng, r: rng[0] + (rng[1] - rng[0]) * r
        self.amp_defend[ids] = lerp(e.opp_defend_offset_range, u[len(REACTIVE_NAMES)])
        self.amp_yield[ids] = lerp(e.opp_yield_offset_range, u[len(REACTIVE_NAMES) + 1])
        self.react_now[ids] = 0
        self.react_off[ids] = 0.0
        self.corner[ids] = -1
        self.corner_amp[ids] = 0.0
        self.corner_mode[ids] = 1.0

    # ------------------------------------------------------------------ the timed machine
    def step(self):
        """Advance every opponent's timed event by one control step: expire, then trigger, then sample."""
        if not self.timed_on:
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
        slot = (u[1] * len(self.timed)).long().clamp_(0, len(self.timed) - 1)
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

    # ------------------------------------------------------------------ the reactive layer
    def step_reactive(self, view: LearnerView):
        """Update the reactive offset from where the learner actually is, this step.

        Three of the four behaviours are an offset, so they share one slew-limited output: the
        target each of them wants is summed, capped at `opp_react_max`, and then moved toward by at
        most `opp_react_slew * dt`. Defend and yield are geometrically exclusive by construction
        (behind versus alongside), so the only sum that happens in practice is a corner line plus
        one of them. `oblivious` produces no offset and is read straight off the disposition by
        `oblivious_mask()`.
        """
        if not self.react_on:
            return
        e = self.ecfg
        g = self.gate
        target = torch.zeros(self.B, device=self.device)
        acting = torch.zeros(self.B, dtype=torch.long, device=self.device)
        learner = view.learner & g[:, None]
        side = lambda lat: (lat / max(e.car_wid, 1e-6)).clamp(-1.0, 1.0)

        if "defend" in self.react:
            has = g & ((self.disp & REACTIVE_BIT["defend"]) != 0)
            rng = float(e.opp_defend_range if e.opp_defend_range > 0 else e.overtake_range)
            full = min(float(e.opp_defend_full), rng - 1e-3)
            # a learner *behind* me: the arc gap to it is negative, and a car whose bodies already
            # overlap mine is alongside, not behind -- that is yield's situation, not defend's.
            behind = learner & (view.gap < -e.car_len) & (view.gap > -rng)
            cand = torch.where(behind, view.gap, torch.full_like(view.gap, -math.inf))
            j = cand.argmax(1, keepdim=True)
            gap_b = cand.gather(1, j)[:, 0]
            lat_b = view.lat.gather(1, j)[:, 0]
            on = has & behind.any(1)
            # strength rises as it closes: nothing at the far edge of contention, full inside
            # `opp_defend_full`. A block thrown from 12 m away is not a block, it is a car that
            # drives off the racing line for no reason.
            w = ((rng - gap_b.abs()) / max(rng - full, 1e-6)).clamp(0.0, 1.0)
            target = target + torch.where(on, self.amp_defend * w * side(lat_b),
                                          torch.zeros_like(target))
            acting = acting | on.long() * REACTIVE_BIT["defend"]

        if "yield" in self.react:
            has = g & ((self.disp & REACTIVE_BIT["yield"]) != 0)
            along = learner & (view.lon.abs() <= e.opp_alongside_lon) & (view.lat.abs() <= e.opp_alongside_lat)
            cand = torch.where(along, -view.lon.abs(), torch.full_like(view.lon, -math.inf))
            j = cand.argmax(1, keepdim=True)
            lat_a = view.lat.gather(1, j)[:, 0]
            on = has & along.any(1)
            target = target + torch.where(on, -self.amp_yield * side(lat_a), torch.zeros_like(target))
            acting = acting | on.long() * REACTIVE_BIT["yield"]

        if "line" in self.react:
            has = g & ((self.disp & REACTIVE_BIT["line"]) != 0)
            if self.corners is None or view.rl_idx is None or view.tid is None:
                raise RuntimeError("opponent event 'line' needs the corner tables and the raceline "
                                   "index; the env did not provide them")
            cid = self._corner_at(view)
            phase = self._corner_field("phase", view)
            inside = self._corner_field("sign", view)
            # A new corner draws its own line. Full-batch draws, like the timed machine's, so the
            # generator advance does not depend on how many cars happen to be entering a corner.
            u = torch.rand(2, self.B, device=self.device, generator=self.gen)
            fresh = (cid >= 0) & (cid != self.corner)
            lerp = lambda rng, r: rng[0] + (rng[1] - rng[0]) * r
            self.corner_mode = torch.where(fresh, torch.where(u[0] < 0.5, -torch.ones_like(u[0]),
                                                              torch.ones_like(u[0])), self.corner_mode)
            self.corner_amp = torch.where(fresh, lerp(e.opp_line_offset_range, u[1]), self.corner_amp)
            self.corner = torch.where(cid >= 0, cid, torch.full_like(cid, -1))
            on = has & (cid >= 0)
            # out-in (mode +1): outside at entry, inside at exit. in-out (mode -1): the reverse.
            # `inside` is the offset sign that points into the corner, so one expression covers both.
            sweep = self.corner_amp * inside * self.corner_mode * (2.0 * phase - 1.0)
            target = target + torch.where(on, sweep, torch.zeros_like(target))
            acting = acting | on.long() * REACTIVE_BIT["line"]

        if "oblivious" in self.react:
            acting = acting | (g & ((self.disp & REACTIVE_BIT["oblivious"]) != 0)).long() * REACTIVE_BIT["oblivious"]

        cap = float(e.opp_react_max)
        target = torch.where(g, target.clamp(-cap, cap), torch.zeros_like(target))
        slew = float(e.opp_react_slew) * self.dt
        self.react_off = self.react_off + (target - self.react_off).clamp(-slew, slew)
        self.react_now = acting

    def _corner_at(self, view: LearnerView) -> torch.Tensor:
        return self.corners["id"][view.tid, view.rl_idx]

    def _corner_field(self, key: str, view: LearnerView) -> torch.Tensor:
        return self.corners[key][view.tid, view.rl_idx]

    def oblivious_mask(self) -> Optional[torch.Tensor]:
        """(B,) cars whose follow-gap slowdown is switched off, or None when nobody's is.

        None rather than an all-False mask so that the env's `follow & ~mask` never runs at all on
        a configuration without this behaviour: that is the line the whole "off is off" property of
        the follow cap rests on.
        """
        if "oblivious" not in self.react:
            return None
        return self.gate & ((self.disp & REACTIVE_BIT["oblivious"]) != 0)

    # ------------------------------------------------------------------ what the env reads
    def speed_scale(self) -> Optional[torch.Tensor]:
        """(B,) multiplier on the opponent's commanded speed, 1.0 where no event is slowing it.

        Never above 1: `gym_env` takes the minimum of this and the follow-gap cap, which is only a
        cap on a car that is *already* slowing for a car ahead. A multiplier that could exceed 1
        would let an event undo that slowdown and drive the opponent into the car it is following.
        """
        if not self.timed_on:
            return None
        s = torch.ones(self.B, device=self.device)
        s = torch.where(self.gate & (self.kind == EVENT_ID["brake"]), self.p0.clamp(0.0, 1.0), s)
        s = torch.where(self.gate & (self.kind == EVENT_ID["stop"]), torch.zeros_like(s), s)
        return s

    def lateral_offset(self) -> Optional[torch.Tensor]:
        """(B,) metres left of the raceline the opponent should be tracking right now (0 = on it).

        The scripted offset plus the reactive one. Summing them is the honest composition: they are
        independent draws about independent things (a scheduled lane change, and where the learner
        is), and the teacher clamps the sum against the lane's own free space, so the result can no
        more reach a wall than either part could.
        """
        if not self.enabled:
            return None
        o = torch.zeros(self.B, device=self.device)
        if self.timed_on:
            ramp = self.p1.clamp_min(1e-3)
            # 0 -> 1 over `ramp`, 1 while held, 1 -> 0 over the last `ramp`: a lane change, not a
            # teleport. The car has to be able to follow the line it is given, and pure pursuit on a
            # target that jumps 0.35 m sideways in one step asks for a step steer input.
            up = (self.t / ramp).clamp(0.0, 1.0)
            down = ((self.dur - self.t) / ramp).clamp(0.0, 1.0)
            o = torch.where(self.gate & (self.kind == EVENT_ID["shift"]), self.p0 * torch.minimum(up, down), o)
            phase = 2 * math.pi * self.t / self.p1.clamp_min(1e-3)
            o = torch.where(self.gate & (self.kind == EVENT_ID["weave"]), self.p0 * torch.sin(phase), o)
        if self.react_on:
            o = o + torch.where(self.gate, self.react_off, torch.zeros_like(o))
        return o

    def info(self) -> Dict[str, torch.Tensor]:
        """Per-step view for a viewer, a logger or the census: which behaviour each car is running.

        Reported for every env row; a car that is not a teacher-driven opponent reads 0 everywhere.
        `id` stays the first key and the only bool-free scalar id: `traffic_attribution.py` and the
        viewer both find "is this car inside an event" by taking the first tensor of the dict.
        """
        idle = ~self.gate | (self.kind == NO_EVENT)
        z = torch.zeros(self.B, device=self.device)
        return {"id": torch.where(idle, torch.zeros_like(self.kind), self.kind),
                "time_left": torch.where(idle, torch.zeros_like(self.t), (self.dur - self.t).clamp_min(0.0)),
                "offset": self.lateral_offset() if self.enabled else z,
                # reactive: what is acting now, and what this car was given for the whole race
                "react": self.react_now.clone(),
                "disposition": torch.where(self.gate, self.disp, torch.zeros_like(self.disp))}

    # ------------------------------------------------------------------ diagnostics
    def counts(self) -> Dict[str, int]:
        """How many cars are currently running each configured behaviour (host sync: demos and tests)."""
        out = {n: int((self.gate & (self.kind == EVENT_ID[n])).sum()) for n in self.timed}
        out.update({n: int(((self.react_now & REACTIVE_BIT[n]) != 0).sum()) for n in self.react})
        return out


def describe(names: Sequence[str], rate: float, probs: Optional[Dict[str, float]] = None) -> str:
    timed, react = split_events(names)
    parts = []
    if timed:
        parts.append(f"timed {list(timed)} at {rate:g} per opponent per 10 s")
    if react:
        p = probs or {}
        parts.append("reactive " + ", ".join(f"{n} p={p.get(n, 0.0):g}" for n in react))
    return " | ".join(parts) if parts else "opponent events off"
