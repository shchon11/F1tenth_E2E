"""CellRecorder: first-attempt masking, static-mu guard, avoidance windows, denominators."""
from __future__ import annotations
import importlib
import pytest


@pytest.fixture
def rn(bench):
    return importlib.import_module("f1sim.learn.benchmark.runner")


def mk(rn, suite="S", n=2, **kw):
    return rn.CellRecorder(n=n, suite=suite, track_length_m=50.0, vehicle_length=0.58,
                           vehicle_width=0.31, hold_steps=40, **kw)


def step(rec, *, progress=None, collision=None, truncated=None, **kw):
    n = rec.n
    rec.update(progress=progress or [1.0] * n, speed=[2.0] * n, dt=0.025,
               collision=collision or [False] * n, truncated=truncated or [False] * n, **kw)


# ---------------------------------------------------------------- first-attempt masking

def test_a_trial_contributes_once(rn):
    rec = mk(rn)
    step(rec, collision=[True, False])
    before = rec.progress_m[0]
    for _ in range(10):
        step(rec)                                 # the crashed trial keeps being fed
    assert rec.progress_m[0] == before            # but accumulates nothing further
    assert not rec.active[0] and rec.active[1]


def test_outcomes_are_mutually_exclusive(rn):
    rec = mk(rn)
    step(rec, collision=[True, False], truncated=[True, True])
    rec.finalize()
    assert rec.outcome[0]["reason"] == "collision"     # collision wins; not both
    assert rec.outcome[1]["reason"] == "timeout"


def test_unfinished_trials_time_out_rather_than_vanish(rn):
    rec = mk(rn)
    step(rec)
    out = rec.finalize()
    assert out["tally"]["denominator"] == 2
    assert out["tally"]["failures"]["timeout"] == 2


def test_counts_match_the_per_trial_array(rn):
    rec = mk(rn, n=4)
    step(rec, collision=[True, False, False, False])
    step(rec, truncated=[False, True, False, False])
    out = rec.finalize()
    assert len(out["outcomes"]) == 4
    assert out["tally"]["denominator"] == 4
    assert sum(out["tally"]["failures"].values()) == 4


# ---------------------------------------------------------------- static mu

def test_mu_change_without_reset_raises(rn):
    rec = mk(rn)
    rec.bind_mu([0.9, 0.9])
    with pytest.raises(rn.MuChangedError, match="fixed for the whole episode"):
        rec.check_mu([0.9, 0.8], fresh=[False, False])


def test_mu_may_be_redrawn_on_reset(rn):
    """`fresh` means the env ENTERED the step just reset -- the only sound discriminator here."""
    rec = mk(rn)
    rec.bind_mu([0.9, 0.9])
    rec.check_mu([0.9, 0.75], fresh=[False, True])     # car 1 just reset: a new draw is legal
    rec.check_mu([0.9, 0.75], fresh=[False, False])    # and is now its episode value


def test_rear_box_guard_rejects_a_widened_draw(bench):
    gm = importlib.import_module("f1sim.learn.benchmark.geom")
    import torch
    gm.assert_rear_box_within_bound(torch.tensor([0.06, 0.11]))
    with pytest.raises(ValueError, match="exceeds the declared bound"):
        gm.assert_rear_box_within_bound(torch.tensor([0.06, 0.20]))


# ---------------------------------------------------------------- avoidance

def test_cleared_requires_passing_the_window(rn):
    rec = mk(rn, suite="A", n=1, s_obs_m=20.0)
    for s in (10.0, 16.0, 18.0, 20.0, 22.0, 25.0):
        step(rec, s=[s])
    assert rec.outcome[0]["success"] is True
    assert rec.encountered[0]


def test_collision_inside_the_window_is_a_hit(rn):
    rec = mk(rn, suite="A", n=1, s_obs_m=20.0)
    step(rec, s=[19.0])
    step(rec, s=[20.0], collision=[True])
    assert rec.outcome[0] == {"success": False, "reason": "hit"}


def test_collision_before_the_window_is_an_approach_failure(rn):
    """Not an N/A exclusion: it is a failure and stays in the denominator."""
    rec = mk(rn, suite="A", n=1, s_obs_m=20.0)
    step(rec, s=[5.0], collision=[True])
    out = rec.finalize()
    assert rec.outcome[0]["reason"] == "approach_collision"
    assert out["tally"]["denominator"] == 1
    assert out["tally"]["rate"]["value"] == 0.0


def test_conditional_rate_is_reported_beside_the_primary(rn):
    rec = mk(rn, suite="A", n=2, s_obs_m=20.0)
    for s in (10.0, 19.0, 20.0, 24.0):
        step(rec, s=[s, 5.0])
    step(rec, s=[26.0, 5.0], collision=[False, True])
    out = rec.finalize()
    assert out["encountered"] == 1
    assert out["tally"]["rate"]["value"] == pytest.approx(0.5)        # 1 of 2 pre-validated
    assert out["conditional_cleared"]["value"] == pytest.approx(1.0)  # 1 of 1 encountered


# ---------------------------------------------------------------- overtaking

def test_held_pass_ends_measurement_as_success(rn):
    rec = mk(rn, suite="O", n=1)
    g = 3.0
    step(rec, gap=[g])
    while g > -1.0:
        g -= 0.02
        step(rec, gap=[g])
    for _ in range(45):
        step(rec, gap=[-1.0])
    assert rec.outcome[0]["success"] is True
    assert not rec.active[0]


def test_no_pass_times_out_with_a_reason(rn):
    rec = mk(rn, suite="O", n=1)
    for _ in range(20):
        step(rec, gap=[3.0])
    out = rec.finalize()
    assert out["tally"]["failures"]["no_pass"] == 1
    assert out["tally"]["denominator"] == 1


def test_opponent_respawn_stays_in_the_denominator(rn):
    rec = mk(rn, suite="O", n=1)
    step(rec, gap=[3.0])
    step(rec, gap=[2.0], opponent_reset=[True])
    out = rec.finalize()
    assert out["tally"]["denominator"] == 1
    assert out["tally"]["successes"] == 0
    assert "opponent_respawn" in out["tally"]["failures"]


def test_hold_interruptions_are_reported(rn):
    rec = mk(rn, suite="O", n=1)
    step(rec, gap=[3.0])
    for g in (2.0, 1.0, 0.0, -1.0, -0.8, -1.0):
        step(rec, gap=[g])
    out = rec.finalize()
    assert "hold_interruptions" in out


# ---------------------------------------------------------------- progress

def test_route_progress_fraction_is_signed(rn):
    assert rn.route_progress_fraction([171.0], [342.0])[0] == pytest.approx(0.5)
    assert rn.route_progress_fraction([-20.0], [342.0])[0] < 0      # reversing subtracts


def test_progress_accumulates_only_while_active(rn):
    rec = mk(rn, n=1)
    for _ in range(5):
        step(rec, progress=[2.0])
    step(rec, progress=[2.0], collision=[True])
    dead = rec.progress_m[0]
    for _ in range(5):
        step(rec, progress=[2.0])
    assert rec.progress_m[0] == dead


def test_short_cell_reports_na_rather_than_a_rank(rn):
    """A cell that lost trials is a different measurement, not a worse score."""
    rec = mk(rn, n=8)
    rec.n = 8
    for _ in range(2):
        step(rec)
    out = rec.finalize()
    assert out["tally"]["complete"] is True       # all 8 accounted for, as timeouts
    rec2 = mk(rn, n=8)
    rec2.outcome = [{"success": True, "reason": None}] * 6 + [None, None]
    rec2.active = [False] * 6 + [False, False]
    t = importlib.import_module("f1sim.learn.benchmark.tally").Tally(expected_n=8)
    for o in rec2.outcome[:6]:
        t.record(o["success"], o["reason"])
    assert t.rate()["value"] is None and "6 of 8" in t.rate()["reason"]
