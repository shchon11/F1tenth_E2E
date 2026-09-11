"""Pass detection: a state machine over unwrapped relative arc progress.

The reward's `overtake_gain` measures arc *taken out of* the opponent (gym_env.py:72) and is not a
pass count, so passes are measured here instead.

Two things make this reset- and wrap-safe. `G` is integrated from wrapped per-step increments, so a
lap wrap -- which moves `signed_gaps` by L -- cannot appear as a crossing: per-step relative motion
is metres. And `G` starts at the real initial gap `g0`, not 0, so a car that begins alongside is not
credited with having arrived there.

Success is binary, at most one per race: on hold completion the outcome is recorded and task
measurement stops.
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
        self.history = []
        self._step_machine()

    def update(self, g: float, *, contact: bool = False, opponent_reset: bool = False,
               learner_reset: bool = False, terminated: bool = False) -> State:
        if self.done:
            return self.state
        if learner_reset:
            return self._bail("learner_reset")
        if opponent_reset:
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
            self.invalidate("repassed")
            self.history.append((self.G, self.state))
            return
        if s is State.IDLE:
            if g > self.clear:
                self.state = State.ARMED_BEHIND
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
        self.history.append((self.G, self.state))

    def _enter_hold(self) -> None:
        self.state, self._held, self._led = State.AHEAD, 1, True
        if self._held >= self.hold_steps:
            self._succeed()

    def _succeed(self) -> None:
        """A success carries no failure reason, whatever happened on the way there."""
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
            "armed": any(s is not State.IDLE for _, s in det.history),
            "final_G": det.G}
