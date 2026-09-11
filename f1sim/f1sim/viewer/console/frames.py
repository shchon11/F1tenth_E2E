"""The bounded latest-snapshot channel, on the GUI side, plus the playback clock.

No Qt and no GL in here on purpose: this is where the freshness accounting lives, and it is the
part most worth testing without a window.

Two ideas carry the whole file.

**Bounded, drop-oldest.** The worker produces frames at the simulation's pace and the GUI consumes
them at the display's. Those rates are unrelated, so one of them has to drop. Dropping the *oldest*
is the only choice that keeps latency bounded -- a queue that grows instead means the picture drifts
further behind reality the longer you watch, which is exactly the failure that makes a viewer feel
laggy even at a healthy frame rate. `dropped` counts what was thrown away and the UI shows it.

**Freshness has two axes.** "When did a frame last arrive?" and "how old is the newest frame I
have?" are different questions, and only asking the first hides a backlog: frames keep arriving on
time while every one of them is stale. `Freshness` reports both, plus the sequence gap.

A note on `seq_gap`, because it is easy to over-read. The worker's sequence counter increments per
frame *produced*, so a frame it dropped shows up as a jump, and because this channel is
drop-oldest the newest received frame carries the newest produced sequence -- worker-side drops do
land in the gap. What also lands in it is the interpolator's deliberate lag: playback trails the
newest frame by a frame or two on purpose, so a perfectly healthy session sits at a gap of 1-2
rather than 0. `INTERP_LAG_FRAMES` names that expected offset; anything above it is real backlog.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

#: How many decoded frames the GUI keeps. Three is the minimum the interpolator can bracket with;
#: more than a handful is latency being stored up rather than smoothness being bought.
DEFAULT_CAPACITY = 4

#: Beyond this the data is called stale in the UI (only while the session is actually RUNNING).
STALE_AFTER_S = 0.5

#: How far behind the newest frame the interpolator deliberately plays. A healthy session sits at
#: about this sequence gap; only a larger one means frames are genuinely piling up.
INTERP_LAG_FRAMES = 2

#: Keys `frame_to_draw` and the renderer rely on. A frame missing any of them is rejected at the
#: door rather than thrown from inside a paint handler.
REQUIRED_FRAME_KEYS = ("t", "n", "seq", "x", "y", "yaw")


@dataclass
class Freshness:
    """Everything the UI needs to say honestly how current the picture is."""
    arrival_age: float          # [s] since the newest frame was received by the GUI
    creation_age: float         # [s] since the worker built it (its clock, mapped onto ours)
    clock_uncertainty: float    # [s] half-width of the worker/GUI clock offset estimate
    seq_gap: int                # newest sequence the worker announced minus the one being drawn
    have_data: bool

    def is_stale(self, limit: float = STALE_AFTER_S) -> bool:
        return self.have_data and (self.arrival_age > limit or self.creation_age > limit)


class ClockLink:
    """Diagnostics for the worker's clock. Deliberately not a correction.

    The worker runs on this machine, spawned by this process, and on Linux `time.monotonic()` is
    `CLOCK_MONOTONIC` -- system-wide, not per-process. Measured with both fork and spawn, a child's
    reading falls strictly inside the parent's surrounding window. So `created_monotonic` from the
    worker is directly comparable to ours, and a frame's age is simply `now - created_monotonic`.

    An earlier version estimated an offset from the round trip by assuming the worker read its
    clock half-way through. That assumption is wrong for the one exchange we have: the worker's
    first reply comes after it has imported torch, which takes seconds, so its timestamp sits next
    to the *reply*, not in the middle. Send at 0, worker stamps 2.0, reply at 2.01 gives a midpoint
    estimate of -0.995 s -- and every frame then reads a second older than it is. A freshness
    display that inflates itself by a second is worse than none.

    What the round trip does tell us is how long the worker took to answer, which is worth showing
    as a diagnostic. `skew` is what the midpoint estimator *would* have concluded; a large value
    here means a slow reply, not a clock difference, and it is reported that way.

    A worker on another host would need a real time-sync protocol, and is not in scope.
    """

    def __init__(self, same_host: bool = True):
        self.same_host = same_host
        self.offset = 0.0            # always 0 for a local worker; see the docstring
        self.rtt = math.inf          # [s] best (lowest) observed round trip
        self.skew = 0.0              # [s] apparent midpoint skew, diagnostic only
        self.samples = 0

    def observe(self, t_send: float, t_worker: float, t_recv: float) -> None:
        """One round trip: we sent at `t_send`, the worker stamped `t_worker`, we got it at `t_recv`."""
        rtt = max(0.0, t_recv - t_send)
        self.samples += 1
        if rtt < self.rtt:
            self.rtt = rtt
            self.skew = (t_send + rtt / 2.0) - t_worker

    def to_local(self, t_worker: float) -> float:
        """The worker's monotonic reading, in our terms. Identity on a shared clock."""
        return t_worker + self.offset

    @property
    def uncertainty(self) -> float:
        """How wrong a frame age could be. On a shared clock this is the timestamping granularity,
        not half the round trip -- the round trip is the worker's response time, not clock error."""
        if not self.samples:
            return math.inf
        return 0.0 if self.same_host else self.rtt / 2.0

    @property
    def ready(self) -> bool:
        return self.samples > 0


#: A car that moved further than this multiple of what its own speed allows in the elapsed time did
#: not drive there. Generous on purpose: the point is to catch a reset, not to police the physics.
TELEPORT_FACTOR = 4.0
#: Plus a floor, so a car that was stationary at the instant before a reset is still caught.
TELEPORT_FLOOR_M = 0.5


def _teleported(f0: dict, f1: dict) -> bool:
    """Did any car jump further between these two frames than driving can explain?"""
    if "x" not in f0 or "x" not in f1 or "y" not in f0 or "y" not in f1:
        return False
    dt = float(f1.get("t", 0.0)) - float(f0.get("t", 0.0))
    if dt <= 0.0:
        return False
    try:
        step = np.hypot(np.asarray(f1["x"]) - np.asarray(f0["x"]),
                        np.asarray(f1["y"]) - np.asarray(f0["y"]))
        v = np.abs(np.asarray(f0.get("vx", 0.0)))
        allowed = TELEPORT_FACTOR * (v + 1.0) * dt + TELEPORT_FLOOR_M
        return bool(np.any(step > allowed))
    except Exception:
        return False


class FrameBuffer:
    """Bounded latest-snapshot store with a playback clock.

    The playback clock is inherited from the previous viewer and is worth keeping: it advances in
    *simulation* time at the simulation's own measured rate and deliberately trails the newest frame
    by a frame or two, so irregular arrivals still play back smoothly instead of stuttering
    forward. What it must never do is run faster than the simulation actually went -- the clock is
    driven by the measured sim rate, never by the render rate, so a fast display cannot make the
    physics look quick.
    """

    def __init__(self, capacity: int = DEFAULT_CAPACITY, clock: Optional[ClockLink] = None):
        self.capacity = capacity
        self._frames: Deque[dict] = deque(maxlen=capacity)
        self.clock = clock or ClockLink()
        # Two very different events, kept apart. A frame leaving the history *after* it has been
        # drawn is ordinary retirement -- the buffer only keeps a few for interpolation. A frame
        # leaving *before* it was ever drawn is the GUI genuinely falling behind. Counting both as
        # "dropped" made a perfectly healthy session report 577 drops out of 581 frames.
        self.retired = 0                 # evicted after being drawn: normal
        self.dropped = 0                 # evicted without ever being drawn: the GUI fell behind
        self.malformed = 0               # refused at the door: missing keys
        self.received = 0
        self.worker_dropped = 0          # what the worker reported dropping on its side
        self.receiver_dropped = 0        # the GUI's own inbound slot -- NOT the worker's
        self.newest_seq = -1             # newest sequence the worker says exists
        self._arrival = 0.0
        self._play_t: Optional[float] = None
        self._play_wall = 0.0
        self._sim_rate = 1.0
        #: What the last `frame_to_draw` actually did. Three scalars, overwritten each call -- no
        #: history, no allocation. Without them "the picture is stepping" cannot be attributed to
        #: the playback clock, the bracket, or the camera, and the answer would be a guess.
        self.play_t: Optional[float] = None
        self.last_alpha: Optional[float] = None
        self.last_bracket: Optional[tuple] = None
        self.last_mode: str = "none"
        self._drawn_seq = -1
        self._generation = -1

    # ---------------------------------------------------------------- ingest
    def set_generation(self, gen: int, reset_clock: bool = False) -> None:
        """Adopt a new configuration generation and forget everything from the previous one.

        Without this, a quick map switch would keep drawing the old map's cars for a few frames
        after the new geometry is up -- the classic "my last request got overwritten by the one
        before it" bug. Frames tagged with any other generation are refused outright.

        Generations must be allocated by the console and never reused, so a new session cannot
        collide with a stale frame still in flight; `SessionController` owns that counter and
        `assert` guards it here. `reset_clock` is for a *new worker process* -- the same worker's
        clock link stays valid across generations, and throwing away a good measurement would only
        make the freshness display worse.
        """
        gen = int(gen)
        if gen < self._generation:
            raise ValueError(f"generation went backwards: {gen} after {self._generation}")
        if gen == self._generation:
            return
        self._generation = gen
        self._frames.clear()
        self._play_t = None
        self._drawn_seq = -1
        self.newest_seq = -1
        # A new session runs at its own pace; inheriting the previous one's measured rate makes the
        # first second of playback run at a speed nothing has measured.
        self._sim_rate = 1.0
        self.worker_dropped = 0
        self.retired = 0
        self.dropped = 0
        self.receiver_dropped = 0
        if reset_clock:
            self.clock = ClockLink()

    @staticmethod
    def valid(frame) -> bool:
        """Enough checking that nothing downstream raises; no more than that.

        Key presence is not sufficient -- a `seq` that is a string or an `x` shorter than `n` gets
        past that and then throws somewhere far less convenient, like a paint handler. This checks
        the few things the renderer indexes with, and nothing else: the point is an exception
        boundary, not a schema.
        """
        if not isinstance(frame, dict):
            return False
        if any(k not in frame for k in REQUIRED_FRAME_KEYS):
            return False
        try:
            n = int(frame["n"])
            int(frame["seq"])
            int(frame.get("gen", -1))
            float(frame["t"])
        except (TypeError, ValueError):
            return False
        if n < 0:
            return False
        for k in ("x", "y", "yaw"):
            arr = frame[k]
            try:
                if len(arr) < n:
                    return False
            except TypeError:
                return False
        return True

    def push(self, frame: dict) -> bool:
        """Accept a decoded frame.

        False means it was refused: a generation we no longer show, one we have not adopted yet, or
        a frame the renderer could not index safely. Refusing here is what keeps a malformed frame
        from becoming an exception inside a paint handler.
        """
        if self._generation < 0:
            return False                       # nothing adopted yet: no frame can be for "now"
        if not self.valid(frame):
            self.malformed += 1
            return False
        try:
            if int(frame["gen"]) != self._generation:
                return False
            seq = int(frame["seq"])
            dropped = frame.get("worker_dropped")
            dropped = int(dropped) if dropped is not None else None
            rate = frame.get("sim_rate")
            rate = float(rate) if rate else None
        except (KeyError, TypeError, ValueError):
            self.malformed += 1
            return False
        if len(self._frames) == self._frames.maxlen:
            evicted = self._frames[0]
            try:
                seen = int(evicted.get("seq", -1)) <= self._drawn_seq
            except (TypeError, ValueError):
                seen = False
            if seen:
                self.retired += 1
            else:
                self.dropped += 1
        self._frames.append(frame)
        self.received += 1
        self._arrival = time.monotonic()
        self.newest_seq = max(self.newest_seq, seq)
        if dropped is not None:
            self.worker_dropped = dropped
        if rate:
            self._sim_rate = rate
        return True

    def note_receiver_drops(self, n: int) -> None:
        """Frames the GUI's own inbound slot discarded.

        Kept separate from `worker_dropped`, which the worker reports about its own outbound slot.
        Folding one into the other blamed the producer for the receiver's queue.
        """
        self.receiver_dropped = int(n)

    def clear(self) -> None:
        self._frames.clear()
        self._play_t = None
        self._drawn_seq = -1

    # ---------------------------------------------------------------- read
    @property
    def latest(self) -> Optional[dict]:
        return self._frames[-1] if self._frames else None

    def freshness(self) -> Freshness:
        newest = self.latest
        if newest is None:
            return Freshness(math.inf, math.inf, self.clock.uncertainty, 0, False)
        now = time.monotonic()
        created = newest.get("created_monotonic")
        if created is not None:
            # Direct subtraction: the worker is a local process on the same CLOCK_MONOTONIC, so its
            # stamp is already in our terms. No round-trip correction -- see ClockLink.
            creation_age = max(0.0, now - self.clock.to_local(float(created)))
        else:
            creation_age = max(0.0, now - self._arrival)     # no stamp: arrival is all we know
        # Raw gap includes the interpolator's deliberate lag; the caller compares against
        # INTERP_LAG_FRAMES rather than against zero.
        gap = max(0, self.newest_seq - self._drawn_seq) if self._drawn_seq >= 0 else 0
        return Freshness(arrival_age=max(0.0, now - self._arrival), creation_age=creation_age,
                         clock_uncertainty=self.clock.uncertainty, seq_gap=gap, have_data=True)

    def stats(self) -> Dict[str, int]:
        return {"received": self.received,
                "retired_after_draw": self.retired,      # normal history turnover
                "dropped_undrawn_gui": self.dropped,     # the GUI could not keep up
                "dropped_worker": self.worker_dropped,   # producer side, reported by the worker
                "dropped_receiver_queue": self.receiver_dropped,   # the GUI's inbound slot
                "malformed": self.malformed,
                "buffered": len(self._frames)}

    # ---------------------------------------------------------------- playback
    def frame_to_draw(self, interpolate: bool = True) -> Optional[dict]:
        """The frame the renderer should draw now, interpolated between two real ones when it can."""
        frames: List[dict] = list(self._frames)
        if not frames:
            return None
        if not interpolate or len(frames) < 3:
            self._drawn_seq = int(frames[-1].get("seq", -1))
            self.last_mode = "raw:disabled" if not interpolate else "raw:too_few_frames"
            self.last_alpha, self.last_bracket = None, None
            self.play_t = frames[-1].get("t")
            return frames[-1]
        now = time.monotonic()
        t_new, t_old = frames[-1]["t"], frames[0]["t"]
        span = max(1e-3, (t_new - t_old) / max(1, len(frames) - 1))
        if self._play_t is None:
            self._play_t, self._play_wall = t_new - 2 * span, now
            self._drawn_seq = int(frames[-1].get("seq", -1))
            self.last_mode = "raw:clock_start"
            self.last_alpha, self.last_bracket = None, None
            self.play_t = self._play_t
            return frames[-1]
        lag = t_new - self._play_t
        # advance in sim seconds, at the sim's measured pace, nudged to hold ~2 frames of buffer
        rate = max(0.05, self._sim_rate) * float(np.clip(1.0 + 0.5 * (lag - 2 * span) / (2 * span), 0.5, 1.5))
        self._play_t = min(self._play_t + (now - self._play_wall) * rate, t_new)
        self._play_wall = now
        j = len(frames) - 1
        while j > 0 and frames[j - 1]["t"] > self._play_t:
            j -= 1
        self.play_t = self._play_t
        if j == 0:
            self._drawn_seq = int(frames[0].get("seq", -1))
            self.last_mode = "raw:behind_buffer"
            self.last_alpha, self.last_bracket = None, None
            return frames[0]
        f0, f1 = frames[j - 1], frames[j]
        self._drawn_seq = int(f1.get("seq", -1))
        if f0["n"] != f1["n"]:
            self.last_mode = "raw:car_count_changed"
            self.last_alpha, self.last_bracket = None, None
            return f1
        if _teleported(f0, f1):
            # An episode reset moves a car to the start line between one frame and the next.
            # Interpolating that draws the car sliding across the map at a speed it never had, and
            # the chase camera then follows the slide. Cut to the new pose instead.
            self.last_mode = "raw:teleport"
            self.last_alpha, self.last_bracket = None, None
            return f1
        a = float(np.clip((self._play_t - f0["t"]) / max(1e-6, f1["t"] - f0["t"]), 0.0, 1.0))
        self.last_mode = "interp"
        self.last_alpha = a
        self.last_bracket = (int(f0.get("seq", -1)), int(f1.get("seq", -1)))
        out = dict(f1)
        for k in ("x", "y", "vx", "steer", "roll", "pitch"):
            if k in f0 and k in f1:
                out[k] = f0[k] + (f1[k] - f0[k]) * a
        if "yaw" in f0 and "yaw" in f1:
            dyaw = (f1["yaw"] - f0["yaw"] + math.pi) % (2 * math.pi) - math.pi
            out["yaw"] = f0["yaw"] + dyaw * a
        out["t"] = self._play_t
        return out

    def freeze(self) -> None:
        """Drop the playback clock, for a pause the worker has confirmed.

        Called on the pause *ack*, never on the click. While a pause is still pending the
        simulation really is still stepping, so the picture must keep following it -- freezing
        early would be the same lie as a button that reads "일시정지" before anything paused.

        After the ack the worker stops producing, and the frames still in flight are real states it
        simulated before stopping; drawing forward to the newest of them is the coherent thing to
        do, and that is what `frame_to_draw` does with interpolation off. Clearing the clock here
        means resuming starts a fresh playback rather than trying to catch up a stale one.
        """
        self._play_t = None


class RateMeter:
    """Frame pacing, kept raw, with the difference between "throughput" and "typical" made explicit.

    Two numbers that are easy to confuse and are not the same:

    * **throughput** -- frames actually completed divided by the time they took. This is the frame
      rate. Reported over the whole run and over the retained window.
    * **percentiles of the interval** -- what a typical gap looked like, and what the worst was.
      These describe the *distribution*, and 1000/median is emphatically not a frame rate: intervals
      of 1, 1 and 98 ms have a median of 1 ms, and calling that 1000 fps describes three frames in a
      tenth of a second as if it were a thousand.

    Deliberately no exponential smoothing. Smoothing an fps counter is how a viewer that hitches
    every second reports a healthy average, and finding the hitch was the point.
    """

    def __init__(self, capacity: int = 120):
        self.capacity = capacity
        self._dt: Deque[float] = deque(maxlen=capacity)
        self._last: Optional[float] = None
        self.total = 0                    # every tick since reset, not just the retained ones
        self._t_first: Optional[float] = None
        self._t_last: Optional[float] = None

    def tick(self, now: Optional[float] = None) -> Optional[float]:
        """Record one completed frame; returns its duration in ms, or None for the first."""
        now = time.monotonic() if now is None else now
        dt_ms = None
        if self._last is not None:
            dt_ms = (now - self._last) * 1e3
            self._dt.append(dt_ms)
            self.total += 1
        else:
            self._t_first = now
        self._last = now
        self._t_last = now
        return dt_ms

    def reset(self) -> None:
        self._dt.clear()
        self._last = None
        self.total = 0
        self._t_first = self._t_last = None

    @property
    def count(self) -> int:
        """Intervals currently retained -- at most `capacity`. NOT the number of frames drawn."""
        return len(self._dt)

    @property
    def elapsed_s(self) -> float:
        if self._t_first is None or self._t_last is None:
            return 0.0
        return self._t_last - self._t_first

    def fps(self) -> Optional[float]:
        """Measured throughput over the retained window: frames / time those frames took.

        Not 1000/median. See the class docstring.
        """
        if not self._dt:
            return None
        total_ms = sum(self._dt)
        return (len(self._dt) / total_ms * 1e3) if total_ms > 0 else None

    def fps_run(self) -> Optional[float]:
        """Measured throughput over the whole run since reset."""
        if self.total < 1 or self.elapsed_s <= 0:
            return None
        return self.total / self.elapsed_s

    def percentiles(self):
        """(p50, p95, max) ms of the retained intervals -- the most recent `count` of them."""
        if not self._dt:
            return None
        s = sorted(self._dt)
        return (s[len(s) // 2],
                s[min(len(s) - 1, int(0.95 * len(s)))],
                s[-1])
