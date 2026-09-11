"""Pass detection: slow crossings, wrap, respawn, contact, re-pass, lead start, binary success."""
from __future__ import annotations
import importlib
import pytest

LENGTH = 50.0          # track arc length, m
HOLD = 40              # 1.0 s at 40 Hz


@pytest.fixture
def mk(bench):
    ot = importlib.import_module("f1sim.learn.benchmark.overtake")
    gm = importlib.import_module("f1sim.learn.benchmark.geom")

    def _mk(vehicle_length=0.58):
        overlap, clear = gm.thresholds(vehicle_length)
        return ot, ot.PassDetector(length=LENGTH, overlap=overlap, clear=clear, hold_steps=HOLD)
    return _mk


def drive(det, gaps, **kw):
    for g in gaps:
        det.update(g, **kw)
    return det


def ramp(start, end, step):
    """A crossing that takes many frames, which is what a real pass looks like."""
    g, out = start, []
    n = int(abs(end - start) / step)
    for _ in range(n):
        g += step if end > start else -step
        out.append(g)
    return out


def test_thresholds_include_the_rear_box(bench, vehicle_length):
    gm = importlib.import_module("f1sim.learn.benchmark.geom")
    overlap, clear = gm.thresholds(vehicle_length)
    assert overlap == pytest.approx(vehicle_length + 0.11)
    assert clear == pytest.approx(vehicle_length + 0.11 + 0.20)
    # the thing R1 was about: clear must sit outside the worst-case contact envelope
    assert clear > gm.contact_extent_m(vehicle_length, 0.11)


def test_slow_multi_frame_pass_succeeds(mk):
    ot, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -3.0, 0.02))          # 300 frames: ~7.5 s, a realistic pass
    drive(det, [-3.0] * HOLD)
    assert det.succeeded
    assert ot.outcome(det)["success"] is True


def test_jump_skipping_the_alongside_band_is_not_counted(mk):
    """A pass must be driven, not teleported.

    +3 m to -3 m in one 25 ms step is 240 m/s of relative motion. The machine requires the
    manoeuvre to pass through the alongside band (|G| <= overlap), which any real crossing samples
    many times over: at 3 m/s relative, dG is 0.075 m per step against a 1.38 m band.
    """
    ot, det = mk()
    det.start(3.0)
    det.update(-3.0)
    drive(det, [-3.0] * HOLD * 2)
    assert not det.succeeded
    assert det.state is ot.State.ARMED_BEHIND


def test_hold_is_exactly_one_second_of_qualifying_steps(mk):
    """The crossing step counts as hold step 1, so HOLD qualifying steps complete the pass."""
    _, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -0.90, 0.02))          # arrive just past `clear`
    assert not det.succeeded
    drive(det, [-0.90] * (HOLD - 2))
    assert not det.succeeded
    det.update(-0.90)
    assert det.succeeded


def test_stall_alongside_is_not_a_pass(mk):
    _, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, 0.0, 0.02))
    drive(det, [0.0] * 400)
    assert not det.succeeded


def test_abort_back_to_behind_is_not_a_pass(mk):
    ot, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, 0.0, 0.02))
    drive(det, ramp(0.0, 3.0, 0.02))
    assert det.state is ot.State.ARMED_BEHIND
    assert not det.succeeded


def test_repass_before_hold_completes_is_a_failure(mk):
    """Stop the crossing just past `clear` so the hold is still running when it is taken back."""
    _, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -0.95, 0.02))
    assert not det.succeeded                    # only a few qualifying steps so far
    drive(det, ramp(-0.95, 3.0, 0.02))          # opponent takes it back mid-hold
    assert not det.succeeded
    assert det.reason == "repassed"


def test_lap_wrap_is_not_a_pass(mk):
    """g jumps by ~L at the wrap; integrated G must not move."""
    _, det = mk()
    det.start(3.0)
    before = det.G
    det.update(3.0 - LENGTH)                    # same position, other side of the wrap
    assert det.G == pytest.approx(before, abs=1e-9)
    assert not det.succeeded


def test_wrap_during_a_real_pass_still_counts(mk):
    _, det = mk()
    det.start(3.0)
    for g in ramp(3.0, -3.0, 0.02):
        det.update(g - LENGTH if g < 0 else g)  # wrap representation flips mid-manoeuvre
    drive(det, [-3.0 - LENGTH] * HOLD)
    assert det.succeeded


def test_opponent_respawn_invalidates(mk):
    ot, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -0.95, 0.02))          # mid-hold, not yet a success
    assert not det.succeeded
    det.update(-0.95, opponent_reset=True)
    assert det.state is ot.State.INVALID
    assert det.reason == "opponent_respawn"
    assert not det.succeeded


def test_contact_during_pass_invalidates(mk):
    ot, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -0.95, 0.02))
    det.update(-0.95, contact=True)
    assert det.state is ot.State.INVALID and det.reason == "contact"


def test_initial_lead_never_arms(mk):
    """G starts at the real g0: a car already ahead has not passed anyone."""
    ot, det = mk()
    det.start(-3.0)
    drive(det, [-3.0] * (HOLD * 3))
    assert not det.succeeded
    assert det.state is ot.State.IDLE
    assert ot.outcome(det)["armed"] is False


def test_G_starts_at_g0_not_zero(mk):
    _, det = mk()
    det.start(7.25)
    assert det.G == pytest.approx(7.25)


def test_success_is_binary_and_terminal(mk):
    ot, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -3.0, 0.02))
    drive(det, [-3.0] * HOLD)
    assert det.succeeded
    det.update(5.0, opponent_reset=True)        # later events cannot revoke it
    det.update(5.0, contact=True)
    assert det.succeeded and det.state is ot.State.PASS_HELD


def test_done_stops_measurement(mk):
    _, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -3.0, 0.02))
    drive(det, [-3.0] * HOLD)
    assert det.done


# -- margin dip vs a real re-pass: the distinction PM flagged ---------------------------------

def test_margin_dip_while_still_ahead_is_not_a_repass(mk):
    """Ego ahead by 0.95 m drifts to 0.80 m and recovers.

    The opponent never got ahead. That interrupts the *continuous* hold and nothing else: the race
    stays live, no failure reason is attached, and a later clean hold still succeeds.
    """
    ot, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -0.95, 0.02))
    drive(det, ramp(-0.95, -0.80, 0.02))        # dip inside the margin, still ahead
    assert det.state is ot.State.ALONGSIDE
    assert det.reason is None                    # NOT repassed
    assert det.interruptions == 1
    drive(det, ramp(-0.80, -1.20, 0.02))        # recover the margin
    drive(det, [-1.20] * HOLD)
    assert det.succeeded
    assert ot.outcome(det)["reason"] is None     # a success carries no failure reason


def test_real_repass_ends_the_race_as_failure(mk):
    """The opponent gets genuinely ahead (G > +clear) before the hold completes."""
    ot, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -0.95, 0.02))
    drive(det, ramp(-0.95, 2.0, 0.02))          # opponent retakes the lead outright
    assert det.state is ot.State.INVALID
    assert det.reason == "repassed"
    assert det.done                              # terminal: the race is over
    assert ot.outcome(det)["reason"] == "repassed"


def test_repass_is_terminal_and_cannot_later_succeed(mk):
    _, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -0.95, 0.02))
    drive(det, ramp(-0.95, 2.0, 0.02))
    drive(det, ramp(2.0, -3.0, 0.02))           # would otherwise be a clean pass
    drive(det, [-3.0] * HOLD)
    assert not det.succeeded


def test_interruptions_are_counted_not_fatal(mk):
    _, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -0.95, 0.02))
    for _ in range(3):
        drive(det, ramp(-0.95, -0.80, 0.02))
        drive(det, ramp(-0.80, -0.95, 0.02))
    assert det.interruptions == 3
    assert det.reason is None
    drive(det, [-0.95] * HOLD)
    assert det.succeeded


def test_no_pass_reason_when_race_simply_ends(mk):
    ot, det = mk()
    det.start(3.0)
    drive(det, [3.0] * 100)
    assert ot.outcome(det)["reason"] == "no_pass"


def test_start_fully_clears_prior_state(mk):
    ot, det = mk()
    det.start(3.0)
    drive(det, ramp(3.0, -0.95, 0.02))
    drive(det, ramp(-0.95, 2.0, 0.02))
    assert det.reason == "repassed"
    det.start(4.0)                               # reused for the next race
    assert det.reason is None and det.interruptions == 0
    assert det.state is ot.State.ARMED_BEHIND and det.G == pytest.approx(4.0)
    assert len(det.history) == 1
