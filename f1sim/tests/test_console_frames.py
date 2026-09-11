"""The bounded latest-snapshot channel and the freshness accounting.

No Qt, no GL, no torch: this is the part of the console whose correctness decides whether the
numbers on screen are true, so it is tested on its own.
"""
import math
import time

import numpy as np
import pytest

from f1sim.viewer.console.frames import (ClockLink, FrameBuffer, INTERP_LAG_FRAMES, RateMeter,
                                         STALE_AFTER_S)
from f1sim.viewer.console.protocol import LatestSlot


def frame(seq, t, gen=0, n=1, **extra):
    f = {"gen": gen, "seq": seq, "t": t, "n": n,
         "x": np.zeros(n, np.float32), "y": np.zeros(n, np.float32),
         "yaw": np.zeros(n, np.float32)}
    f.update(extra)
    return f


# ---------------------------------------------------------------- LatestSlot
def test_latest_slot_drops_oldest_and_never_blocks():
    slot = LatestSlot(2)
    assert slot.put("a") is True
    assert slot.put("b") is True
    assert slot.put("c") is False            # had to discard to make room
    assert slot.dropped == 1
    assert slot.get(0.0) == "b"              # the oldest survivor, not "a"
    assert slot.get(0.0) == "c"
    assert slot.get(0.0) is None


def test_latest_slot_bounded_under_flood():
    slot = LatestSlot(3)
    for i in range(1000):
        slot.put(i)
    assert len(slot) == 3
    assert slot.dropped == 997
    assert [slot.get(0.0) for _ in range(3)] == [997, 998, 999]   # always the newest


# ---------------------------------------------------------------- generations
def test_push_refused_before_a_generation_is_adopted():
    b = FrameBuffer()
    assert b.push(frame(0, 0.0)) is False
    b.set_generation(0)
    assert b.push(frame(0, 0.0)) is True


def test_frames_from_another_generation_are_refused():
    b = FrameBuffer()
    b.set_generation(4)
    assert b.push(frame(0, 0.0, gen=4)) is True
    assert b.push(frame(1, 0.1, gen=3)) is False       # the previous session, arriving late
    assert b.push(frame(2, 0.2, gen=5)) is False       # one we have not adopted yet
    assert b.stats()["received"] == 1


def test_adopting_a_generation_forgets_the_previous_one():
    b = FrameBuffer()
    b.set_generation(1)
    for i in range(3):
        b.push(frame(i, i * 0.1, gen=1, sim_rate=2.0))
    assert b.latest is not None
    b.set_generation(2)
    assert b.latest is None
    assert b.newest_seq == -1
    # the previous run's measured pace must not drive the new one's playback
    assert b._sim_rate == 1.0


def test_generations_cannot_go_backwards():
    b = FrameBuffer()
    b.set_generation(3)
    with pytest.raises(ValueError):
        b.set_generation(2)
    b.set_generation(3)          # same value is a no-op, not an error
    b.set_generation(4)


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("missing", ["t", "n", "seq", "x", "y", "yaw"])
def test_malformed_frames_are_refused_not_raised(missing):
    b = FrameBuffer()
    b.set_generation(0)
    f = frame(0, 0.0)
    del f[missing]
    assert b.push(f) is False
    assert b.stats()["malformed"] == 1
    assert b.frame_to_draw() is None          # and drawing does not raise


def test_non_dict_frame_is_refused():
    b = FrameBuffer()
    b.set_generation(0)
    assert b.push(None) is False
    assert b.push([1, 2, 3]) is False


# ---------------------------------------------------------------- freshness
def test_freshness_without_data():
    b = FrameBuffer()
    f = b.freshness()
    assert not f.have_data
    assert not f.is_stale()


def test_creation_age_exposes_a_backlog_that_arrival_age_hides():
    """Frames arriving on time can still all be old. Arrival age alone would call that healthy."""
    b = FrameBuffer()
    b.set_generation(0)
    b.clock.observe(t_send=100.0, t_worker=100.0, t_recv=100.0)     # perfect link, offset 0
    old = time.monotonic() - 2.0
    b.push(frame(0, 0.0, created_monotonic=old))
    f = b.freshness()
    assert f.arrival_age < 0.1                 # it *arrived* a moment ago
    assert f.creation_age > 1.5                # but it was built two seconds ago
    assert f.is_stale(STALE_AFTER_S)


def test_seq_gap_counts_worker_side_drops():
    b = FrameBuffer()
    b.set_generation(0)
    for seq in (0, 1, 2):
        b.push(frame(seq, seq * 0.025, created_monotonic=time.monotonic()))
    b.frame_to_draw(interpolate=False)         # draws seq 2
    b.push(frame(40, 1.0, created_monotonic=time.monotonic()))   # worker dropped 37 of them
    assert b.freshness().seq_gap == 38


def test_healthy_interpolated_playback_sits_near_the_expected_lag():
    b = FrameBuffer()
    b.set_generation(0)
    now = time.monotonic()
    for seq in range(4):
        b.push(frame(seq, seq * 0.025, created_monotonic=now, sim_rate=1.0))
    b.frame_to_draw(interpolate=True)
    gap = b.freshness().seq_gap
    assert 0 <= gap <= INTERP_LAG_FRAMES + 1, gap


def test_worker_drop_count_is_reported_not_inferred():
    b = FrameBuffer()
    b.set_generation(0)
    b.push(frame(0, 0.0, worker_dropped=17))
    assert b.stats()["dropped_worker"] == 17


# ---------------------------------------------------------------- playback
def test_retiring_a_drawn_frame_is_not_a_drop():
    """A small history turns over constantly; that is not the GUI failing to keep up.

    Counting every eviction as a drop made a healthy session report 577 drops out of 581 frames --
    a number that would send someone looking for a performance problem that was not there.
    """
    b = FrameBuffer(capacity=4)
    b.set_generation(0)
    for i in range(20):
        b.push(frame(i, i * 0.025))
        b.frame_to_draw(interpolate=False)          # drawn between pushes, as a live viewer does
    st = b.stats()
    assert st["dropped_undrawn_gui"] == 0, st
    assert st["retired_after_draw"] > 0


def test_frames_never_drawn_are_counted_as_dropped():
    b = FrameBuffer(capacity=4)
    b.set_generation(0)
    for i in range(20):
        b.push(frame(i, i * 0.025))                 # renderer withheld: nothing is ever drawn
    st = b.stats()
    assert st["dropped_undrawn_gui"] == 16, st
    assert st["retired_after_draw"] == 0
    assert st["buffered"] == 4


def test_receiver_queue_drops_are_not_blamed_on_the_worker():
    b = FrameBuffer()
    b.set_generation(0)
    b.push(frame(0, 0.0, worker_dropped=5))
    b.note_receiver_drops(9)
    st = b.stats()
    assert st["dropped_worker"] == 5
    assert st["dropped_receiver_queue"] == 9


def test_interpolated_frame_lies_between_its_neighbours():
    b = FrameBuffer(capacity=8)
    b.set_generation(0)
    now = time.monotonic()
    for seq in range(5):
        f = frame(seq, seq * 0.1, n=1, sim_rate=1.0, created_monotonic=now)
        f["x"] = np.array([float(seq)], np.float32)
        f["vx"] = np.array([1.0], np.float32)
        b.push(f)
    b.frame_to_draw(interpolate=True)        # first call seeds the clock
    time.sleep(0.05)
    out = b.frame_to_draw(interpolate=True)
    assert out is not None
    assert 0.0 <= float(out["x"][0]) <= 4.0


def test_playback_never_runs_faster_than_the_measured_sim_rate():
    """A fast display must not make the physics look quick."""
    b = FrameBuffer(capacity=8)
    b.set_generation(0)
    now = time.monotonic()
    for seq in range(6):
        b.push(frame(seq, seq * 0.1, sim_rate=0.25, created_monotonic=now))
    b.frame_to_draw(interpolate=True)
    t0 = b._play_t
    time.sleep(0.1)
    b.frame_to_draw(interpolate=True)
    advanced = b._play_t - t0
    # 0.1 s of wall time at a measured 0.25x, with at most the interpolator's 1.5x nudge
    assert advanced <= 0.1 * 0.25 * 1.5 + 1e-6, advanced


def test_yaw_interpolation_takes_the_short_way_round():
    b = FrameBuffer(capacity=8)
    b.set_generation(0)
    now = time.monotonic()
    for seq, yaw in enumerate([math.pi - 0.05, math.pi - 0.05, -math.pi + 0.05, -math.pi + 0.05]):
        f = frame(seq, seq * 0.1, sim_rate=1.0, created_monotonic=now)
        f["yaw"] = np.array([yaw], np.float32)
        b.push(f)
    b.frame_to_draw(interpolate=True)
    time.sleep(0.02)
    out = b.frame_to_draw(interpolate=True)
    # crossing +pi/-pi must not sweep the long way through zero
    assert abs(float(out["yaw"][0])) > math.pi - 0.2


def test_freeze_resets_the_playback_clock():
    b = FrameBuffer(capacity=8)
    b.set_generation(0)
    for seq in range(4):
        b.push(frame(seq, seq * 0.1, sim_rate=1.0))
    b.frame_to_draw(interpolate=True)
    assert b._play_t is not None
    b.freeze()
    assert b._play_t is None


# ---------------------------------------------------------------- clock link
def test_clock_link_keeps_the_lowest_round_trip():
    c = ClockLink()
    assert not c.ready
    c.observe(t_send=0.0, t_worker=0.5, t_recv=1.0)          # rtt 1.0
    assert c.rtt == pytest.approx(1.0)
    c.observe(t_send=10.0, t_worker=12.0, t_recv=14.0)       # rtt 4.0, worse -> not taken
    assert c.rtt == pytest.approx(1.0)
    c.observe(t_send=20.0, t_worker=20.05, t_recv=20.1)      # rtt 0.1, better
    assert c.rtt == pytest.approx(0.1)
    assert c.ready


def test_clock_link_never_corrects_a_shared_clock():
    """A local worker shares CLOCK_MONOTONIC, so the mapping is identity, whatever the round trip."""
    c = ClockLink(same_host=True)
    c.observe(t_send=0.0, t_worker=2.0, t_recv=2.01)         # slow reply, not a clock difference
    assert c.offset == 0.0
    assert c.to_local(123.5) == 123.5
    assert c.uncertainty == 0.0
    assert c.skew == pytest.approx(-0.995, abs=1e-3)         # kept, but only as a diagnostic


def test_a_slow_worker_startup_does_not_inflate_frame_age():
    """PM's case: the worker stamps its hello after importing torch, seconds after we asked.

    A midpoint offset estimate would conclude the worker's clock runs ~1 s ahead and then report
    every fresh frame as a second old. It must not.
    """
    b = FrameBuffer()
    b.set_generation(0)
    b.clock.observe(t_send=0.0, t_worker=2.0, t_recv=2.01)   # 2 s of `import torch`
    now = time.monotonic()
    b.push(frame(0, 0.0, created_monotonic=now - 0.1))       # genuinely 100 ms old
    age = b.freshness().creation_age
    assert 0.05 < age < 0.3, f"reported {age:.3f}s for a 0.1s-old frame"
    assert not b.freshness().is_stale()


# ---------------------------------------------------------------- rate meter
def test_fps_is_throughput_not_inverse_median():
    """1, 1 and 98 ms is three frames in a tenth of a second: 30 fps, not 1000."""
    m = RateMeter()
    t = 0.0
    for dt in (0.001, 0.001, 0.098):
        m.tick(t)
        t += dt
    m.tick(t)
    assert m.fps() == pytest.approx(30.0, rel=0.02)
    assert m.fps_run() == pytest.approx(30.0, rel=0.02)
    p50, p95, mx = m.percentiles()
    assert p50 == pytest.approx(1.0, abs=0.1)       # the distribution still shows the hitch
    assert mx == pytest.approx(98.0, abs=0.1)


def test_retained_interval_count_is_not_the_frame_count():
    """`count` is how many intervals are retained (capped); `total` is how many frames happened."""
    m = RateMeter(capacity=10)
    t = 0.0
    for _ in range(50):
        m.tick(t)
        t += 0.016
    assert m.count == 10
    assert m.total == 49
    assert m.elapsed_s == pytest.approx(49 * 0.016, rel=1e-6)


def test_rate_meter_reports_percentiles_not_a_smoothed_average():
    m = RateMeter(capacity=100)
    base = 1000.0
    times = [base]
    for dt in [0.016] * 99 + [0.25]:            # 99 good frames and one long hitch
        times.append(times[-1] + dt)
    for t in times:
        m.tick(t)
    p50, p95, mx = m.percentiles()
    assert p50 == pytest.approx(16.0, abs=1.0)
    assert mx == pytest.approx(250.0, abs=1.0)   # the hitch is visible, not averaged away
    assert m.fps() == pytest.approx(1000.0 * 100 / (99 * 16 + 250), rel=0.05)


def test_rate_meter_is_empty_before_two_frames():
    m = RateMeter()
    assert m.percentiles() is None
    assert m.fps() is None
    assert m.tick(0.0) is None
    assert m.tick(0.016) == pytest.approx(16.0, abs=0.1)


# ---------------------------------------------------------------- type safety
@pytest.mark.parametrize("bad", [
    {"seq": "not-an-int"},
    {"n": "two"},
    {"t": None},
    {"gen": object()},
    {"n": 5},                              # more cars claimed than the arrays hold
    {"x": 3.0},                            # not indexable
    {"n": -1},
])
def test_wrong_types_are_refused_not_raised(bad):
    """`valid` checks types and shapes, not just key presence: a bad value must not reach paint."""
    b = FrameBuffer()
    b.set_generation(0)
    f = frame(0, 0.0)
    f.update(bad)
    assert b.push(f) is False
    assert b.stats()["malformed"] >= 1
    assert b.frame_to_draw() is None


def test_bad_optional_fields_are_refused_without_raising():
    b = FrameBuffer()
    b.set_generation(0)
    f = frame(0, 0.0, worker_dropped="lots")
    assert b.push(f) is False
    f = frame(1, 0.1, sim_rate="fast")
    assert b.push(f) is False
    assert b.stats()["received"] == 0


def test_a_valid_frame_still_gets_through_after_bad_ones():
    b = FrameBuffer()
    b.set_generation(0)
    bad = frame(0, 0.0)
    bad["n"] = "x"
    b.push(bad)
    assert b.push(frame(1, 0.1, worker_dropped=3, sim_rate=1.0)) is True
    assert b.stats()["received"] == 1
    assert b.stats()["dropped_worker"] == 3
