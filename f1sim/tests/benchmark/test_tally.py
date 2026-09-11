"""Denominators cannot silently shrink, and N/A is never a zero."""
from __future__ import annotations
import importlib
import pytest


@pytest.fixture
def tal(bench):
    return importlib.import_module("f1sim.learn.benchmark.tally")


def test_approach_failure_stays_in_the_denominator(tal):
    """A car that crashed before reaching the obstacle did not avoid it. It is a failure."""
    t = tal.Tally()
    t.record(True)
    t.record(False, "approach_collision")
    t.record(False, "approach_timeout")
    assert t.denominator == 3
    assert t.rate()["value"] == pytest.approx(1 / 3)


def test_respawn_invalidated_race_stays_in_denominator_without_credit(tal):
    t = tal.Tally()
    t.record(True)
    t.record(False, "opponent_respawn")
    assert t.denominator == 2
    assert t.successes == 1
    assert t.rate()["value"] == pytest.approx(0.5)


def test_only_setup_failures_leave_the_denominator(tal):
    t = tal.Tally()
    t.record(True)
    t.record(False, "lead_start")
    t.record(False, "invalid_geometry")
    assert t.denominator == 1
    assert t.excluded == 2
    assert t.rate()["value"] == pytest.approx(1.0)


def test_empty_is_na_not_zero(tal):
    t = tal.Tally()
    r = t.rate()
    assert r["value"] is None and r["reason"]


def test_all_failed_is_zero_not_na(tal):
    """Zero and unmeasured are different: everything failed here, and that is a real 0."""
    t = tal.Tally()
    t.record(False, "hit")
    assert t.rate()["value"] == 0.0


def test_conditional_rate_is_separate_from_primary(tal):
    t = tal.Tally()
    t.record(True)
    t.record(False, "approach_collision")
    assert t.rate()["value"] == pytest.approx(0.5)          # primary: all pre-validated
    assert t.conditional_rate(encountered=1)["value"] == pytest.approx(1.0)   # secondary


def test_conditional_rate_without_encounters_is_na(tal):
    assert tal.Tally().conditional_rate(0)["value"] is None


def test_reasons_are_preserved_for_rebuilding(tal):
    t = tal.Tally()
    for _ in range(3):
        t.record(False, "approach_collision")
    t.record(False, "lead_start")
    d = t.as_dict()
    assert d["failures"]["approach_collision"] == 3
    assert d["setup_failures"]["lead_start"] == 1
    assert d["denominator"] == 3 and d["excluded_from_denominator"] == 1


# -- post-freeze integrity ---------------------------------------------------------------------

def test_setup_failure_is_refused_after_freeze(tal):
    """Geometry proves scenarios valid before the freeze; afterwards a lead start is a refusal."""
    t = tal.Tally(expected_n=8, frozen=True)
    with pytest.raises(ValueError, match="after suite freeze"):
        t.record(False, "lead_start")


def test_setup_failure_still_diagnoses_before_freeze(tal):
    t = tal.Tally(frozen=False)
    t.record(False, "invalid_geometry")
    assert t.setup_failures["invalid_geometry"] == 1


def test_short_denominator_cannot_be_ranked(tal):
    """6 of 8 trials is a different measurement, not a worse score."""
    t = tal.Tally(expected_n=8)
    for _ in range(4):
        t.record(True)
    for _ in range(2):
        t.record(False, "contact")
    r = t.rate()
    assert r["value"] is None
    assert "6 of 8" in r["reason"]
    assert t.as_dict()["complete"] is False


def test_full_denominator_ranks(tal):
    t = tal.Tally(expected_n=8)
    for _ in range(5):
        t.record(True)
    for _ in range(3):
        t.record(False, "no_pass")
    assert t.rate()["value"] == pytest.approx(5 / 8)
    assert t.as_dict()["complete"] is True


def test_conditional_rate_rejects_impossible_orderings(tal):
    t = tal.Tally()
    for _ in range(3):
        t.record(True)
    t.record(False, "hit")
    with pytest.raises(ValueError, match="only 2 encounters"):
        t.conditional_rate(2)                    # 3 successes, 2 encounters
    with pytest.raises(ValueError, match="outside"):
        t.conditional_rate(9)                    # more encounters than trials
    assert t.conditional_rate(4)["value"] == pytest.approx(0.75)
