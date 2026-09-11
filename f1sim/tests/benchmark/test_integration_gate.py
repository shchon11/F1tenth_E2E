"""Real-env opponent independence. Slow (builds a simulator), CPU only, no checkpoint."""
from __future__ import annotations
import importlib
import pytest

pytest.importorskip("torch")
pytest.importorskip("f1sim.gym_env")


@pytest.fixture(scope="module")
def gate():
    return importlib.import_module("f1sim.learn.benchmark.integration_gate")


EST = ("/home/shchon11/Documents/Codex/2026-09-10/new-chat/work/learning-next/"
       "continuous-learning/runner/estimator-out/estimator_seed401.pt")


@pytest.mark.slow
def test_the_estimated_arm_needs_an_explicit_pinned_estimator(gate):
    """There is no default: a benchmark reaching into one experiment's work directory is not
    portable, and a silent default would let the wrong student be scored."""
    with pytest.raises(ValueError, match="explicit pinned estimator|no default"):
        gate.run_gate(routed=True, steps=2, estimator_path=None)


@pytest.mark.slow
def test_routed_opponent_is_bit_identical_across_candidates(gate):
    r = gate.run_gate(routed=True, steps=30, estimator_path=EST)
    assert r["opponent_commands_identical"], f"opponent moved by {r['max_abs_delta']}"
    assert r["max_abs_delta"] == 0.0


@pytest.mark.slow
def test_unrouted_control_actually_diverges(gate):
    """Without this, the gate above passes vacuously.

    A candidate *action* cannot move an opponent row -- PlanTracker is row-wise -- so the control
    perturbs the tracker, which is what an arm change does at this boundary. It must diverge.
    """
    c = gate.run_gate(routed=False, steps=30, estimator_path=EST)
    assert not c["opponent_commands_identical"]
    assert c["max_abs_delta"] > 0.0


@pytest.mark.slow
def test_warmth_is_read_per_case_from_learner_history_not_a_step_count(gate):
    """`steps > 40` asserts warmth it cannot know, and one warm row anywhere is not coverage.

    Every estimated case must independently start cold and reach a full history on EVERY learner.
    Counting all rows let an opponent's full window stand in for the candidate's; keeping one dict
    let the last case overwrite the others.
    """
    r = gate.run_gate(routed=True, steps=60, estimator_path=EST)
    hist = r["estimator_history"]
    assert hist, "the estimated arm must report the history it actually held"

    est_cases = [k for k in r["cases"] if k.endswith("/estimated")]
    assert len(est_cases) == 2, "both actors must be crossed with the estimated arm"
    for case in est_cases:
        h = hist[case]                              # per case, not one shared record
        assert h["learner_rows"] > 0
        assert h["min_valid_seen"] < h["warm_frames"], f"{case} never started cold"
        assert h["max_valid_seen"] >= h["warm_frames"], f"{case} never reached a full history"
        assert h["started_cold"] and h["all_learners_warm"]

    assert r["covers_cold_and_warm"] is True
    assert r["uncovered_cases"] == []


@pytest.mark.slow
def test_coverage_is_refused_when_the_history_cannot_fill(gate):
    """Too few steps to fill a 40-frame window: report uncovered rather than claim warmth."""
    r = gate.run_gate(routed=True, steps=6, estimator_path=EST)
    assert r["covers_cold_and_warm"] is False
    assert r["uncovered_cases"], "an uncovered run must name the cases that failed"
