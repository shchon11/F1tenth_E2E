"""Pass detection: a state machine over unwrapped relative arc progress.

The reward's `overtake_gain` measures arc *taken out of* the opponent (gym_env.py:72) and is not a
pass count, so passes are measured here instead.

Two things make this reset- and wrap-safe. `G` is integrated from wrapped per-step increments, so a
lap wrap -- which moves `signed_gaps` by L -- cannot appear as a crossing: per-step relative motion
is metres. And `G` starts at the real initial gap `g0`, not 0, so a car that begins alongside is not
credited with having arrived there.

Success is binary, at most one per race: on hold completion the outcome is recorded and task
measurement stops. That is the **O** family's question -- "was a pass completed?" -- and it is the
right question only where a pass is something a reference driver has been shown to do.

The **T** (traffic) family asks a different one, because on a 0.70 m half-lane a completed pass is
marginal and a binary that reads 0 everywhere measures nothing. Two things here serve it:

  * `PassDetector(repeat=True)` counts passes instead of latching one. The state machine is the
    same machine -- same arming, same alongside band, same hold -- but a completed hold increments
    `passes` and re-arms, an opponent that gets back ahead is a counted `repass` rather than an
    invalidation, and an opponent respawn re-seeds the pair instead of voiding the trial. With
    `repeat=False`, which is the default, every branch added here is skipped and the O family's
    numbers are the numbers it always had.
  * `TrafficTrace` accumulates the continuous side: seconds in contention, seconds close enough to
    attack, the learner's arc progress against the opponents' over the same steps, and the closest
    arc approach. Those are defined on every trial, including the ones where nobody passes anybody,
    which is what makes a narrow floor measurable at all.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    IDLE = "idle"
    ARMED_BEHIND = "armed_behind"
    ALONGSIDE = "alongside"
    AHEAD = "ahead"
    PASS_HELD = "pass_held"
    #: repeat mode only: ahead, clear, and this lead has already been counted as a pass. Reached
    #: only from AHEAD, and the only way out to a *second* count is through ARMED_BEHIND again --
    #: so a car that drifts alongside and back clear without ever losing the lead is not credited
    #: twice for one pass.
    LED = "led"
    INVALID = "invalid"


def wrap(d: float, length: float) -> float:
    """Wrap an arc *difference* into (-L/2, L/2]."""
    return (d + length / 2.0) % length - length / 2.0


@dataclass
class PassDetector:
    """One learner-opponent pair, one race.

    overlap/clear are metres of arc from `geom.thresholds`; hold_steps is the dwell at `clear`.
    """
    length: float
    overlap: float
    clear: float
    hold_steps: int
    #: T (traffic) mode. False is the O family exactly as it was: one pass, terminal, and a re-pass
    #: voids the race. True counts passes over the whole stint instead -- see the module docstring.
    repeat: bool = False
    #: Keep the per-step (G, state) trace. The O family's `outcome` used to derive `armed` from it;
    #: it now comes from a flag, so the trace is pure diagnostics and a 20 000-step traffic trial
    #: does not have to hold one tuple per step per opponent.
    keep_history: bool = True
    state: State = State.IDLE
    G: float | None = None
    reason: str | None = None
    _held: int = 0
    _prev_g: float | None = None
    #: Times the clear margin was lost while the ego was still ahead. Diagnostic, not a failure.
    interruptions: int = 0
    #: Latched once the ego has held a clear lead this race. After that, the opponent getting
    #: clearly ahead again is a re-pass -- however gradually it happens.
    _led: bool = False
    #: repeat mode: completed, held passes over the whole stint, and the number of times an
    #: opponent got clearly back ahead after one. Both stay at 0 when `repeat` is False.
    passes: int = 0
    repasses: int = 0
    #: repeat mode: how many times this pair was re-seeded because the opponent respawned.
    reseeds: int = 0
    #: A pass has to be *arrived at* from behind. Set on entering ARMED_BEHIND, cleared when a pass
    #: is counted, so regaining a clear lead without ever falling behind again counts once.
    _armed: bool = False
    #: Whether the machine ever left IDLE. Mirrors what scanning `history` for a non-IDLE state
    #: used to compute, so `outcome` no longer needs the trace to exist.
    ever_armed: bool = False
    history: list = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.state is State.PASS_HELD

    @property
    def done(self) -> bool:
        """Measurement is over: success is binary and terminal, as is invalidation."""
        return self.state in (State.PASS_HELD, State.INVALID)

    def invalidate(self, reason: str) -> None:
        """Reset, respawn, contact or termination. Never a success."""
        if self.done:
            return                      # a recorded success is not revoked by later events
        self.state, self.reason = State.INVALID, reason

    def start(self, g0: float) -> None:
        """Seed G at the actual initial gap. No phantom zero offset.

        Fully clears prior state: a detector restarted mid-life must not inherit a stale reason,
        hold count or history from the race before it.
        """
        self.G, self._prev_g = g0, g0
        self.state = State.IDLE
        self.reason = None
        self._held = 0
        self.interruptions = 0
        self._led = False
        self._armed = False
        self.ever_armed = False
        self.passes = 0
        self.repasses = 0
        self.reseeds = 0
        self.history = []
        self._step_machine()

    def reseed(self, g0: float) -> None:
        """Re-aim the pair at a NEW opponent without forgetting what this trial already measured.

        Used in repeat mode when the opponent respawns: the car ahead is gone and the gap it left
        behind is not a manoeuvre, so `G` has to restart -- but the passes already completed in this
        trial happened, and `start()` would erase them. Everything that describes the *current*
        engagement resets; everything that counts the *trial* survives.
        """
        passes, repasses, reseeds, armed_ever = self.passes, self.repasses, self.reseeds, self.ever_armed
        self.start(g0)
        self.passes, self.repasses = passes, repasses
        self.reseeds = reseeds + 1
        self.ever_armed = armed_ever or self.ever_armed

    def update(self, g: float, *, contact: bool = False, opponent_reset: bool = False,
               learner_reset: bool = False, terminated: bool = False) -> State:
        if self.done:
            return self.state
        if learner_reset:
            return self._bail("learner_reset")
        if opponent_reset:
            if self.repeat:
                # The car ahead crashed and came back somewhere else. In O that voids the race,
                # because the one pass being measured can no longer be attributed. In T the trial
                # keeps running against whatever traffic is now there, and the passes already
                # completed are not revoked by a later respawn any more than a success is.
                self.reseed(g)
                return self.state
            return self._bail("opponent_respawn")
        if contact:
            return self._bail("contact")
        if terminated:
            return self._bail("terminated")
        if self.G is None:
            self.start(g)
            return self.state
        self.G += wrap(g - self._prev_g, self.length)
        self._prev_g = g
        self._step_machine()
        return self.state

    def _bail(self, reason: str) -> State:
        self.invalidate(reason)
        return self.state

    def _step_machine(self) -> None:
        g, s = self.G, self.state
        # Checked before the state dispatch, and from any state. A re-pass is usually gradual: the
        # ego loses the margin into ALONGSIDE and only then falls behind, so a check that only ran
        # while in AHEAD would catch nothing but an implausible single-step reversal.
        if self._led and g > self.clear:
            if self.repeat:
                # Being re-passed is an event, not the end of the measurement: the learner is
                # behind a car again and can go again. Counted so the table can say a lead was
                # taken and given back rather than reporting only the net.
                self.repasses += 1
                self._led, self._held = False, 0
                self.state = State.ARMED_BEHIND
                self._armed = True
                self._record()
                return
            self.invalidate("repassed")
            self.history.append((self.G, self.state))
            return
        if s is State.IDLE:
            if g > self.clear:
                self.state = State.ARMED_BEHIND
                self._armed = True
        elif s is State.ARMED_BEHIND:
            if abs(g) <= self.overlap:
                self.state = State.ALONGSIDE
        elif s is State.ALONGSIDE:
            if g < -self.clear:
                # The crossing step already satisfies the margin, so it is hold step 1: `hold_steps`
                # then means exactly that many qualifying steps, not that many plus the crossing.
                self._enter_hold()
            elif g > self.clear:
                self.state = State.ARMED_BEHIND          # dropped back; no credit
                self._armed = True
        elif s is State.AHEAD:
            if g < -self.clear:
                self._held += 1
                if self._held >= self.hold_steps:
                    self._succeed()
            else:
                # Margin lost while the ego is still ahead or alongside -- a dip, not a re-pass.
                # The *continuous* hold restarts; the race stays live and can still succeed.
                self._held = 0
                self.interruptions += 1
                self.state = State.ALONGSIDE
        elif s is State.LED:
            # repeat mode only. Ahead and already credited. Losing the margin drops back to
            # ALONGSIDE, from where re-clearing does NOT count again: `_armed` was cleared when the
            # pass was counted, and only falling behind (the `_led` branch above) sets it once more.
            if abs(g) <= self.overlap:
                self.state = State.ALONGSIDE
                self._held = 0
        self._record()

    def _record(self) -> None:
        """One trace entry, and the `ever_armed` flag it used to be scanned for."""
        if self.state is not State.IDLE:
            self.ever_armed = True
        if self.keep_history:
            self.history.append((self.G, self.state))

    def _enter_hold(self) -> None:
        self.state, self._held, self._led = State.AHEAD, 1, True
        if self._held >= self.hold_steps:
            self._succeed()

    def _succeed(self) -> None:
        """A success carries no failure reason, whatever happened on the way there."""
        if self.repeat:
            # Count it once, then keep driving. The credit is gated on `_armed` so that re-clearing
            # after a dip -- which re-enters ALONGSIDE and can complete another hold -- adds nothing:
            # that is one pass being finished, not a second one being made.
            if self._armed:
                self.passes += 1
                self._armed = False
            self.state, self._held, self._led = State.LED, 0, True
            return
        self.state, self.reason = State.PASS_HELD, None


def outcome(det: PassDetector) -> dict:
    """Per-race record. A race that never armed is still a race: it stays in the denominator.

    `reason` is None on success, the terminal cause on an invalidated race, and `no_pass` when the
    race simply ended without one. `hold_interruptions` is reported alongside either way: it
    describes how the manoeuvre went, and never by itself decides the outcome.
    """
    if det.succeeded:
        reason = None
    elif det.state is State.INVALID:
        reason = det.reason
    else:
        reason = "no_pass"
    return {"success": det.succeeded,
            "state": det.state.value,
            "reason": reason,
            "hold_interruptions": det.interruptions,
            "armed": det.ever_armed,
            "final_G": det.G}


# ------------------------------------------------------------------ the continuous side (T)

#: Below this much opponent progress a pace *ratio* is not a pace: dividing by 0.4 m of arc turns
#: measurement noise into a headline number. The trial still reports both progress figures, so the
#: information is not lost -- only the ratio is withheld, with its reason.
PACE_MIN_OPP_PROGRESS_M = 1.0


@dataclass
class TrafficTrace:
    """What happened in traffic, for one learner over one trial, whether or not anyone passed.

    This is the half of the T metric that a narrow floor cannot zero out. A pass on a 0.70 m
    half-lane may never happen for anybody; time spent in contention, time spent close enough for a
    pass to be on, and ground gained or lost against the cars around you happen on every lap.

    Ranges are metres of signed arc, positive when an opponent is ahead:

      contention  |gap| <= `contention_range_m`  -- in traffic at all, either side
      following   0 < gap <= `contention_range_m` -- behind a car, which is where a pass starts
      attacking   0 < gap <= `attack_range_m`     -- close enough that the pass is actually on
      defending   -`contention_range_m` <= gap < 0 -- ahead of a car that is still in range

    `attack_range_m` exists because `contention_range_m` is the env's own `overtake_range`, 12 m,
    and two of the held-out floors have a 33 m lap: on those, |gap| <= 12 m covers three quarters of
    every gap the two cars can be at, so it saturates and separates nobody. The tight window does
    not, and both are reported rather than one being quietly chosen.
    """
    contention_range_m: float
    attack_range_m: float
    seconds: float = 0.0
    contention_s: float = 0.0
    following_s: float = 0.0
    attack_s: float = 0.0
    defending_s: float = 0.0
    ego_progress_m: float = 0.0
    #: Mean over the opponents, matching `gym_env.overtake_gain`'s convention: `other_idx` is a
    #: fixed roster of the other cars, not "the car ahead", so singling one out would make the
    #: two-opponent cells measure something the one-opponent cells do not.
    opponent_progress_m: float = 0.0
    closest_arc_gap_m: float | None = None

    def update(self, *, dt: float, ego_progress: float, opponent_progress, gaps) -> None:
        """One transition. `gaps` and `opponent_progress` are per opponent, read pre-reset."""
        self.seconds += dt
        self.ego_progress_m += float(ego_progress)
        opp = [float(x) for x in opponent_progress]
        if opp:
            self.opponent_progress_m += sum(opp) / len(opp)
        gs = [float(g) for g in gaps]
        if not gs:
            return
        near = min(abs(g) for g in gs)
        self.closest_arc_gap_m = near if self.closest_arc_gap_m is None else min(
            self.closest_arc_gap_m, near)
        if near <= self.contention_range_m:
            self.contention_s += dt
        if any(0.0 < g <= self.contention_range_m for g in gs):
            self.following_s += dt
        if any(0.0 < g <= self.attack_range_m for g in gs):
            self.attack_s += dt
        if any(-self.contention_range_m <= g < 0.0 for g in gs):
            self.defending_s += dt

    def pace_ratio(self) -> dict:
        """Ego arc per opponent arc over the same steps. 1.0 is holding station, >1 is gaining.

        Measured over the learner's OWN active steps, which is the only window both cars are in:
        after a learner crashes its opponent keeps driving, and charging it for arc it was not
        there to contest would make a crash look like a pace deficit instead of a crash. The crash
        is already reported as a crash.
        """
        if self.opponent_progress_m < PACE_MIN_OPP_PROGRESS_M:
            return {"value": None,
                    "reason": f"opponent covered {self.opponent_progress_m:.2f} m, under the "
                              f"{PACE_MIN_OPP_PROGRESS_M} m a ratio needs to mean anything"}
        return {"value": self.ego_progress_m / self.opponent_progress_m, "reason": None}

    def as_dict(self) -> dict:
        return {"seconds": self.seconds,
                "contention_s": self.contention_s, "following_s": self.following_s,
                "attack_s": self.attack_s, "defending_s": self.defending_s,
                "ego_progress_m": self.ego_progress_m,
                "opponent_progress_m": self.opponent_progress_m,
                "closest_arc_gap_m": self.closest_arc_gap_m,
                "pace_ratio": self.pace_ratio()}


class TrafficMeter:
    """The T metric for a rollout loop this module does not own.

    `f1sim.learn.evaluate` exists for quick checks outside the frozen suite, and until now it could
    put a policy in a race and tell you the collision rate -- nothing about the traffic. This
    produces the same quantities the T family reports, from the same `TrafficTrace` and the same
    `PassDetector(repeat=True)`, so a quick check and a frozen cell are measuring one definition
    rather than two that happen to share names.

    It differs from the benchmark path in one respect, and it has to: `evaluate` runs auto-resetting
    envs where there is no such thing as a trial, so the figures are rates over the whole rollout
    rather than per-trial outcomes. A learner that crashes respawns and keeps being measured; the
    crash is counted, and its detectors are re-seeded rather than retired.

    Usage. The meter folds each transition from inside its own `sim.step` wrapper, so it measures a
    loop it was never handed -- `common.rollout_metrics` owns its loop and is a validated
    implementation of a different measurement that a second caller has no business reaching into:

        with TrafficMeter(env) as meter:
            ...                      # any loop at all, including one inside another function
        meter.report()

    Reading from inside the wrapper is also what makes the measurement pre-reset. The gym layer
    resets ended envs a few lines after `sim.step` returns and the simulator reuses its tensors, so
    a gap read after `env.step` returns belongs to the next episode.
    """

    def __init__(self, env, *, contention_range_m: float = 12.0, attack_range_m: float = 3.0,
                 hold_seconds: float = 1.0, vehicle_length: float | None = None):
        import torch
        from .geom import thresholds
        self.env = env
        self.dt = float(env.sim.control_dt)
        self.enabled = bool(getattr(env, "M", 1) > 1 and env.sim.other_idx is not None)
        self.rows = (torch.nonzero(env.learner).flatten().tolist() if self.enabled else [])
        self.n = len(self.rows)
        self.n_opponents = int(env.M) - 1 if self.enabled else 0
        # One length for every detector, as `run_cell` does. Wrap-safety only needs a length that is
        # not shorter than the real lap; on a mixed-track rollout the longest is the safe choice, and
        # `--per-track` -- which is where these numbers are meant to be read -- has only one anyway.
        self.length = (float(env.sim.track.length[env.sim.tid].max()) if self.enabled else 0.0)
        vl = float(vehicle_length) if vehicle_length is not None else float(env.cfg.vehicle.length)
        overlap, clear = thresholds(vl)
        hold_steps = max(1, int(round(hold_seconds / self.dt)))
        self.traces = [TrafficTrace(contention_range_m=contention_range_m,
                                    attack_range_m=attack_range_m) for _ in range(self.n)]
        self.detectors = [[PassDetector(length=self.length, overlap=overlap, clear=clear,
                                        hold_steps=hold_steps, repeat=True, keep_history=False)
                           for _ in range(self.n_opponents)] for _ in range(self.n)]
        self.car_contacts = 0
        self.wall_collisions = 0
        self.learner_respawns = 0
        self.event_steps = 0
        self.event_in_window_steps = 0
        self.steps = 0
        self._orig = None
        self._pending = None
        self._prev_s = None
        self._prev_opp_s = None
        self._seeded = False

    # -- pre-reset capture ------------------------------------------------------------------------
    def __enter__(self):
        if not self.enabled:
            return self
        env = self.env
        self._orig = env.sim.step

        def wrapped(*a, **kw):
            entry_steps = env.sim.steps.clone()
            r = self._orig(*a, **kw)
            ev = env.events
            self._pending = {
                "entry_steps": entry_steps,
                "s": env.sim.s.clone(),
                "collision": (r.collision.clone() if getattr(r, "collision", None) is not None
                              else None),
                "car_collision": (r.car_collision.clone()
                                  if getattr(r, "car_collision", None) is not None else None),
                # Read off the machine, not off `info`: `_opponent_actions` steps the events and
                # applies them before this call, so this is the schedule THIS transition ran under.
                "event_id": (ev.info()["id"].clone()
                             if ev is not None and ev.enabled else None),
            }
            self._fold()
            return r
        env.sim.step = wrapped
        return self

    def __exit__(self, *exc):
        if self._orig is not None:
            self.env.sim.step = self._orig
            self._orig = None
        return False

    # -- one transition ---------------------------------------------------------------------------
    def observe(self, info=None) -> None:
        """Kept for callers that want to drive the meter explicitly. Folding happens in the wrapper,
        so this is a no-op while attached; it exists so a loop written against it still reads."""
        return

    def _fold(self) -> None:
        """Fold one env step, from the values captured before the auto-reset."""
        if not self.enabled or self._pending is None:
            return
        import torch
        env, p = self.env, self._pending
        idx = torch.tensor(self.rows, device=env.device, dtype=torch.long)
        other = env.sim.other_idx[idx]                                  # (n, M-1)
        s_now = p["s"][idx]
        opp_s = p["s"][other]
        fresh = (p["entry_steps"] == 0)
        if not self._seeded:
            gaps0 = _wrapped_gaps(p["s"], other, idx, self.length)
            for i, row in enumerate(gaps0):
                for j, g in enumerate(row):
                    self.detectors[i][j].start(float(g))
            self._prev_s, self._prev_opp_s, self._seeded = s_now.clone(), opp_s.clone(), True
            self.steps += 1
            return
        ego_prog = _wrap_t(s_now - self._prev_s, self.length)
        opp_prog = _wrap_t(opp_s - self._prev_opp_s, self.length)
        self._prev_s, self._prev_opp_s = s_now.clone(), opp_s.clone()
        gaps = _wrapped_gaps(p["s"], other, idx, self.length)
        contact = (p["car_collision"][idx].tolist() if p["car_collision"] is not None
                   else [False] * self.n)
        hit = (p["collision"][idx].tolist() if p["collision"] is not None
               else [False] * self.n)
        learner_fresh = fresh[idx].tolist()
        opp_fresh = fresh[other].tolist()
        ev_id = p["event_id"]
        ev_on = ((ev_id[other] != 0).any(dim=1).tolist() if ev_id is not None
                 else [False] * self.n)
        # Under `collision_mode="soft"` the simulator reports a contact on every step it lasts --
        # a hose graze is 40 ms, sixteen steps -- and nothing ends the episode, so counting steps
        # would score one graze as sixteen collisions. Count onsets there, which is what the reward
        # charges and what `terminate` counts by construction (its crash is the episode's last step).
        soft = getattr(env.ecfg, "collision_mode", "terminate") == "soft"
        if soft and getattr(self, "_prev_hit", None) is None:
            self._prev_hit = [False] * self.n
        for i in range(self.n):
            touching = bool(hit[i]) and not bool(contact[i])
            onset = touching and not (soft and self._prev_hit[i] and not bool(learner_fresh[i]))
            if soft:
                self._prev_hit[i] = touching
            if bool(contact[i]):
                self.car_contacts += 1
            elif onset:
                self.wall_collisions += 1
            if bool(learner_fresh[i]):
                self.learner_respawns += 1
            row = gaps[i]
            self.traces[i].update(dt=self.dt, ego_progress=float(ego_prog[i]),
                                  opponent_progress=[float(x) for x in opp_prog[i]], gaps=row)
            in_window = any(abs(g) <= self.traces[i].contention_range_m for g in row)
            if ev_on[i]:
                self.event_steps += 1
                if in_window:
                    self.event_in_window_steps += 1
            for j, det in enumerate(self.detectors[i]):
                # A respawn on either side is a new engagement, not a manoeuvre. Re-seeding keeps
                # the passes already counted, which `start()` would throw away.
                if bool(learner_fresh[i]) or bool(opp_fresh[i][j]):
                    det.reseed(float(row[j]))
                else:
                    det.update(float(row[j]), contact=bool(contact[i]))
        self.steps += 1

    def report(self) -> dict:
        """Rates over the rollout. Empty when the env has no opponents to measure against."""
        if not self.enabled:
            return {"traffic": None,
                    "traffic_note": "no opponents: race_size 1, or no other cars in the batch"}
        minutes = max(self.steps * self.dt / 60.0, 1e-9)
        learner_min = minutes * max(self.n, 1)
        passes = sum(d.passes for row in self.detectors for d in row)
        leads_lost = sum(d.repasses for row in self.detectors for d in row)
        ego = sum(t.ego_progress_m for t in self.traces)
        opp = sum(t.opponent_progress_m for t in self.traces)
        exposure = max(sum(t.seconds for t in self.traces), 1e-9)
        return {"traffic": {
            "learners": self.n, "opponents_per_learner": self.n_opponents,
            "steps": self.steps, "learner_minutes": learner_min,
            "passes": passes, "passes_per_learner_min": passes / learner_min,
            "leads_lost": leads_lost,
            "pace_vs_opponent": (ego / opp) if opp > 0 else None,
            "ego_progress_m": ego, "opponent_progress_m": opp,
            "contention_fraction": sum(t.contention_s for t in self.traces) / exposure,
            "following_fraction": sum(t.following_s for t in self.traces) / exposure,
            "attacking_fraction": sum(t.attack_s for t in self.traces) / exposure,
            "defending_fraction": sum(t.defending_s for t in self.traces) / exposure,
            "car_contacts": self.car_contacts,
            "car_contacts_per_learner_min": self.car_contacts / learner_min,
            "wall_collisions": self.wall_collisions,
            "learner_respawns": self.learner_respawns,
            "opp_event_step_fraction": self.event_steps / max(self.steps * max(self.n, 1), 1),
            "opp_event_in_window_fraction": (self.event_in_window_steps / self.event_steps
                                             if self.event_steps else None),
            "contention_range_m": self.traces[0].contention_range_m if self.traces else None,
            "attack_range_m": self.traces[0].attack_range_m if self.traces else None,
        }}


def _wrap_t(d, length: float):
    return (d + length / 2.0) % length - length / 2.0


def _wrapped_gaps(s, other, idx, length: float):
    """(n, M-1) signed arc from each learner to each of its opponents, + when the opponent is ahead."""
    return _wrap_t(s[other] - s[idx][:, None], length).tolist()
