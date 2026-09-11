"""Named fixtures for the round-2 review findings. One test per bug."""
from __future__ import annotations
import importlib
import pytest


@pytest.fixture
def rn(bench):
    return importlib.import_module("f1sim.learn.benchmark.runner")


@pytest.fixture
def rp(bench):
    return importlib.import_module("f1sim.learn.benchmark.report")


@pytest.fixture
def ro(bench):
    return importlib.import_module("f1sim.learn.benchmark.roster")


def mk(rn, suite="S", n=1, **kw):
    return rn.CellRecorder(n=n, suite=suite, track_length_m=50.0, vehicle_length=0.58,
                           vehicle_width=0.31, hold_steps=40, **kw)


def step(rec, *, progress=None, speed=None, collision=None, truncated=None, **kw):
    n = rec.n
    rec.update(progress=progress or [1.0] * n, speed=speed or [2.0] * n, dt=0.025,
               collision=collision or [False] * n, truncated=truncated or [False] * n, **kw)


# ---- 1. collision on the exit stride must not credit a clear ---------------------------------

def test_collision_on_exit_stride_is_a_hit_not_a_clear(rn):
    rec = mk(rn, suite="A", s_obs_m=20.0)
    step(rec, s=[19.0])                                   # inside the window
    step(rec, s=[25.0], collision=[True])                 # exits AND collides on the same step
    assert rec.outcome[0]["success"] is False
    assert rec.outcome[0]["reason"] == "hit"


def test_collision_on_the_completing_step_is_not_a_completion(rn):
    rec = mk(rn, suite="S")
    step(rec, progress=[60.0], collision=[True])          # past a 50 m lap, but crashed
    assert rec.outcome[0]["success"] is False


# ---- 2/3/7 moved: report and roster validation is frontend-owned ------------------------------
#
# Duplicate/missing cells, mixed SHAs, mixed source digests, the derived-summary refusal, the
# alias rule and the raw per-trial integrity checks now live with `report.py` and `roster.py` in
# the frontend lane (see report-handoff.md), together with `tests/test_result_integrity.py`.
# Keeping a second copy here would duplicate coverage and pin their API from outside their file.
# What remains below is core's: the producer and the recorder.


# ---- 4. collisions/km uses travelled distance, not net progress -------------------------------

def test_distance_is_travelled_not_net_progress(rn):
    """A car that oscillates covers ground; net progress near zero would inflate collisions/km."""
    rec = mk(rn)
    for p_ in (2.0, -2.0, 2.0, -2.0):
        step(rec, progress=[p_], speed=[4.0])
    out = rec.finalize()
    assert out["progress_m"][0] == pytest.approx(0.0)      # net
    assert out["distance_m"][0] == pytest.approx(4 * 4.0 * 0.025)   # travelled



# ---- 5. opponent friction is checked too ------------------------------------------------------

def test_opponent_mu_change_is_caught(rn):
    """Binding only the measured rows lets an opponent change friction unnoticed."""
    rec = mk(rn, suite="O", n=1)
    rec.bind_mu([0.9, 0.9])                       # 2 cars, 1 measured
    with pytest.raises(rn.MuChangedError, match="car 1"):
        rec.check_mu([0.9, 0.7], fresh=[False, False])


# ---- 6. a terminal detector failure deactivates the trial -------------------------------------

def test_detector_invalidation_deactivates_the_trial(rn):
    rec = mk(rn, suite="O", n=1)
    rec.seed_gaps([3.0])
    step(rec, gap=[2.0], opponent_reset=[True])
    assert not rec.active[0]
    out = rec.finalize()
    assert out["tally"]["failures"]["opponent_respawn"] == 1


def test_gaps_are_seeded_before_the_first_transition(rn):
    """Seeding inside the first update would swallow that step's terminal flags."""
    rec = mk(rn, suite="O", n=1)
    rec.seed_gaps([3.0])
    assert rec.detectors[0].G == pytest.approx(3.0)
    step(rec, gap=[2.9], collision=[True])
    assert rec.outcome[0]["success"] is False



# ---- car contact is labelled from the actual flag ---------------------------------------------

def test_car_contact_is_labelled_from_the_flag_not_the_suite(rn):
    rec = mk(rn, suite="O", n=1)
    rec.seed_gaps([3.0])
    step(rec, gap=[2.0], collision=[True], car_contact=[False])
    assert rec.outcome[0]["reason"] == "collision"        # a wall, in a race
    rec2 = mk(rn, suite="O", n=1)
    rec2.seed_gaps([3.0])
    step(rec2, gap=[2.0], collision=[True], car_contact=[True])
    assert rec2.outcome[0]["reason"] == "contact"


# ---- S completion branch ----------------------------------------------------------------------

def test_progress_past_the_lap_completes_inclusively(rn):
    """51 m of a 50 m lap is a completion, not a timeout."""
    rec = mk(rn, suite="S")
    step(rec, progress=[51.0])
    assert rec.outcome[0] == {"success": True, "reason": None}
    out = rec.finalize()
    assert out["completed"][0] is True and out["lap_time_s"][0] is not None


def test_exactly_the_lap_length_completes(rn):
    rec = mk(rn, suite="S")
    step(rec, progress=[50.0])
    assert rec.outcome[0]["success"] is True


def test_budget_exhaustion_times_out(rn):
    rec = mk(rn, suite="S")
    rec.update(progress=[1.0], speed=[2.0], dt=0.025, collision=[False], truncated=[False],
               budget_exhausted=True)
    assert rec.outcome[0]["reason"] == "timeout"


# ---- producer -> validate -> aggregate, on REAL recorder output --------------------------------
#
# A and O successes carry `lap_time_s = None` by construction: `completed[i] = success and
# suite == "S"`, so a task success is not a lap completion and there is no lap time to report.
# Task duration is `elapsed_s`. These drive the real recorder and push its own output through the
# shared gate and the aggregator, which is the path a scored row actually takes.

def _finish_A(rn):
    rec = rn.CellRecorder(n=1, suite="A", track_length_m=50.0, vehicle_length=0.58,
                          vehicle_width=0.31, hold_steps=40, s_obs_m=20.0)
    for s in (10.0, 19.0, 20.0, 25.0):
        rec.update(progress=[1.0], speed=[2.0], dt=0.025, collision=[False], truncated=[False],
                   s=[s])
    return rec.finalize()


def _finish_O(rn):
    rec = rn.CellRecorder(n=1, suite="O", track_length_m=50.0, vehicle_length=0.58,
                          vehicle_width=0.31, hold_steps=40)
    rec.seed_gaps([3.0])
    g = 3.0
    while g > -1.2:
        g -= 0.02
        rec.update(progress=[1.0], speed=[2.0], dt=0.025, collision=[False], truncated=[False],
                   gap=[g])
    while not rec.detectors[0].succeeded:
        rec.update(progress=[1.0], speed=[2.0], dt=0.025, collision=[False], truncated=[False],
                   gap=[-1.2])
    return rec.finalize()


def _row_for(suite, res):
    # `effective` is what the adapter reports the cell actually turned out to be; the real pipeline
    # attaches it in cmd_run. Declared and effective are both recorded so a silent difference is
    # catchable, which is why the validator refuses a row without it.
    res = dict(res)
    res["effective"] = {"plant_mu": 0.73423, "true_mu": 0.73423, "speed_cap": 9.0,
                        "budget_laps": 3.0, "sensor_noise": True, "backend": "f1sim"}
    # A row must record the initial condition it was measured from, or pairing cannot be shown.
    # `run_cell` captures this from the live env; a hand-built row has to supply it too.
    res["start_fingerprint"] = {
        "schema_version": 1, "physical_sha256": "a" * 64, "actor_input_sha256": "b" * 64,
        "calibration_sha256": "c" * 64, "physical_tensors": 111, "actor_input_tensors": 14,
        "calibration_tensors": 5, "n_envs_total": 1, "race_size": 1, "sim_t": 0.025,
        "sim_imu_phase": 1, "obs_spec_sha256": "d" * 64}
    return {"system_id": "sys", "checkpoint_sha256": "a" * 64, "controller_arm": "legacy",
            "runtime": "legacy", "suite_freeze_sha256": "f" * 64, "suite_version": "v1",
            "map_id": "m", "mu": 0.73423, "seed": 4401, "n_envs": 1, "suite": suite,
            "cell_id": f"{suite}:m:0.73423:4401", "source_digest": {"sim.py": "x"}, "result": res}


def test_finalize_emits_a_complete_row_on_its_own(rn):
    """`route_progress_fraction` used to be bolted on by `run_cell`, so a producer-side row was
    incomplete and the validator refused it for a missing array the real pipeline happened to fill."""
    for res in (_finish_A(rn), _finish_O(rn)):
        for key in ("outcomes", "progress_m", "distance_m", "route_progress_fraction",
                    "lap_time_s", "elapsed_s"):
            assert key in res, f"{key} missing from finalize() output"


@pytest.mark.parametrize("suite", ["A", "O"])
def test_a_successful_task_row_validates_and_aggregates(rn, rp, suite):
    res = _finish_A(rn) if suite == "A" else _finish_O(rn)
    assert res["tally"]["successes"] == 1
    assert res["lap_time_s"] == [None], "a task success is not a lap completion"
    assert res["elapsed_s"][0] > 0, "task duration is recorded as elapsed_s"

    row = _row_for(suite, res)
    su = importlib.import_module("f1sim.learn.benchmark.suite")
    cell = su.Cell(suite, "m", 0.73423, 4401, 1)
    # the FULL shared gate, not just validate_raw: it stamps the derived values aggregation needs,
    # so a row that skipped it cannot be aggregated
    rp.validate_cell(row, cell)
    agg = rp.aggregate([row])
    key = "avoidance" if suite == "A" else "overtaking"
    assert key in agg and agg[key], f"{suite} produced no {key} block"


def test_an_S_success_without_a_lap_time_is_refused(rn, rp):
    """The negative: on S a success IS a lap completion, so a missing lap time is a real defect."""
    rec = rn.CellRecorder(n=1, suite="S", track_length_m=50.0, vehicle_length=0.58,
                          vehicle_width=0.31, hold_steps=40)
    rec.update(progress=[51.0], speed=[2.0], dt=0.025, collision=[False], truncated=[False])
    res = rec.finalize()
    assert res["lap_time_s"][0] is not None                # the producer does emit it
    res["lap_time_s"] = [None]                             # simulate a producer that lost it
    with pytest.raises(rp.ReportError, match="no lap time"):
        rp.validate_raw(_row_for("S", res))
