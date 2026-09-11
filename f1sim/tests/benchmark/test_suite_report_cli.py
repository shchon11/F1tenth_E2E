"""Suite freeze, roster pinning, renderer strictness, CLI smoke."""
from __future__ import annotations
import hashlib
import importlib
import json
import os
import subprocess
import sys
import pathlib

import pytest



@pytest.fixture
def su(bench):
    return importlib.import_module("f1sim.learn.benchmark.suite")


@pytest.fixture
def rp(bench):
    return importlib.import_module("f1sim.learn.benchmark.report")


@pytest.fixture
def ro(bench):
    return importlib.import_module("f1sim.learn.benchmark.roster")


# ---------------------------------------------------------------- suite freeze

def test_matrix_matches_the_declared_counts(su):
    s = su.Suite()
    assert len(s.cells()) == 34
    assert s.expected_trials() == {"S": 144, "A": 64, "O": 64}
    assert sum(s.expected_trials().values()) == 272


def test_mu_mid_is_labelled_midpoint_not_nominal(su):
    assert su.MU_MID == pytest.approx((su.MU_LOW + su.MU_HIGH) / 2)
    assert "midpoint" in su.MU_LABELS[su.MU_MID]
    assert su.MU_NOMINAL_VEHICLE == 1.0489 != su.MU_MID


def test_seeds_are_distinct_from_sgr(su):
    assert su.SEEDS == (4401, 4402)


def test_freeze_hash_changes_with_scenario(su):
    a = su.Suite()
    b = su.Suite(seeds=(4401, 9999))
    assert a.freeze_hash() != b.freeze_hash()


def test_freeze_hash_ignores_calibration_timing(su):
    """Cell timing is a property of the machine, not the scenario: it must not break a freeze."""
    a = su.Suite()
    b = su.Suite(calibration={"seconds_per_cell": 11.4})
    assert a.freeze_hash() == b.freeze_hash()


def test_edited_suite_file_is_refused(su, tmp_path):
    p = tmp_path / "suite.json"
    su.Suite().save(str(p))
    raw = json.loads(p.read_text())
    raw["suite"]["seeds"] = [1, 2]               # tamper after freeze
    p.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="edited after freeze"):
        su.load(str(p))


def test_roundtrip_preserves_hash(su, tmp_path):
    p = tmp_path / "suite.json"
    h = su.Suite().save(str(p))
    s2, h2 = su.load(str(p))
    assert h == h2 == s2.freeze_hash()


# ---------------------------------------------------------------- roster pinning

def _ckpt(tmp_path, name="w.pt", body=b"weights"):
    f = tmp_path / name
    f.write_bytes(body)
    return str(f), hashlib.sha256(body).hexdigest()


def test_roster_rejects_latest(ro, tmp_path):
    path, sha = _ckpt(tmp_path)
    e = ro.Entry("s", str(tmp_path / "latest" / "w.pt"), sha, "legacy")
    with pytest.raises(ValueError, match="unresolved 'latest' path segment"):
        e.verify()


def test_roster_rejects_sha_drift(ro, tmp_path):
    path, sha = _ckpt(tmp_path)
    e = ro.Entry("s", path, "0" * 64, "legacy")
    with pytest.raises(ValueError, match="sha mismatch"):
        e.verify()


def test_roster_requires_estimator_pin_for_estimated_arm(ro, tmp_path):
    path, sha = _ckpt(tmp_path)
    with pytest.raises(ValueError, match="estimator pin disagree"):
        ro.Entry("s", path, sha, "estimated").verify()


def test_same_weights_two_arms_are_two_systems(ro, tmp_path):
    path, sha = _ckpt(tmp_path)
    a = ro.Entry("orig@legacy", path, sha, "legacy")
    b = ro.Entry("orig@fixed_low", path, sha, "fixed_low")
    assert a.key() != b.key()
    a.verify(); b.verify()


def test_roster_rejects_duplicate_identity(ro, tmp_path):
    path, sha = _ckpt(tmp_path)
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"systems": [
        {"system_id": "a", "path": path, "checkpoint_sha256": sha, "controller_arm": "legacy"},
        {"system_id": "b", "path": path, "checkpoint_sha256": sha, "controller_arm": "legacy"}]}))
    with pytest.raises(ValueError, match="duplicate system identity"):
        ro.load(str(p))


# ---------------------------------------------------------------- renderer strictness

def _row(**kw):
    r = {"system_id": "sys", "checkpoint_sha256": "a" * 64, "controller_arm": "legacy",
         "suite_freeze_sha256": "f" * 64, "suite_version": "v1", "map_id": "m",
         "mu": 0.73423, "seed": 4401, "n_envs": 8, "suite": "S",
         "source_digest": {"sim.py": "abc"},
         "runtime": "legacy", "cell_id": "S:m:0.73423:4401",
         # rows must carry their raw per-trial data: the report validates arrays against counts
         "result": {"n": 8, "tally": {"successes": 8, "denominator": 8, "failures": {}},
                    "progress_m": [1.0] * 8, "distance_m": [1.0] * 8,
                    "route_progress_fraction": [1.0] * 8, "lap_time_s": [10.0] * 8,
                    "outcomes": [{"success": True, "reason": None}] * 8}}
    r.update(kw)
    return r


def _validated(*rows):
    """Rows as `aggregate` requires them: derived counts rebuilt from the outcome arrays.

    Aggregation refuses an unvalidated row, so a test that aggregates must validate first -- the
    same order the real report takes. Mirrors `report._derive` rather than duplicating its rules.
    """
    import f1sim.learn.benchmark.report as _rp
    for r in rows:
        r["_derived"] = r["result"]["_derived"] = _rp._derive(r["result"])
    return list(rows)


def test_render_refuses_missing_pins(rp):
    with pytest.raises(rp.ReportError, match="missing required pins"):
        rp.validate_row(_row(checkpoint_sha256=None))


def test_render_refuses_estimated_without_estimator(rp):
    with pytest.raises(rp.ReportError, match="without an estimator pin"):
        rp.validate_row(_row(controller_arm="estimated"))


def test_render_refuses_mixed_suite_hashes(rp):
    rows = [_row(), _row(seed=4402, suite_freeze_sha256="e" * 64)]
    with pytest.raises(rp.ReportError, match="different protocols cannot share one table"):
        rp.validate_results(rows, suite_freeze="f" * 64, expected_systems={"sys"},
                            expected_trials={"S": 16})


def test_render_refuses_incomplete_roster(rp):
    with pytest.raises(rp.ReportError, match="roster mismatch"):
        rp.validate_results([_row()], suite_freeze="f" * 64,
                            expected_systems={"sys", "other"}, expected_trials={"S": 8})


def test_render_refuses_short_trial_count(rp):
    with pytest.raises(rp.ReportError, match="short denominator is a different measurement"):
        rp.validate_results([_row()], suite_freeze="f" * 64, expected_systems={"sys"},
                            expected_trials={"S": 144})


def test_render_accepts_a_complete_set(rp):
    rows = [_row(), _row(seed=4402)]
    info = rp.validate_results(rows, suite_freeze="f" * 64, expected_systems={"sys"},
                               expected_trials={"S": 16})
    assert info["n_rows"] == 2 and info["n_systems"] == 1


def test_na_never_renders_as_zero(rp):
    assert rp.fmt({"value": None, "reason": "no encounters"}) == "N/A (no encounters)"
    assert rp.fmt({"value": 0.0, "reason": None}) == "0.000"
    assert rp.fmt(None).startswith("N/A")


def test_unevaluated_category_is_na_not_absent(rp, su):
    md = rp.render_markdown({"suite_freeze_sha256": "f" * 64, "n_rows": 1, "n_systems": 1},
                            suite=su.Suite())
    assert "N/A (category not evaluated)" in md
    assert "Overtaking" in md


def test_markdown_states_scope_and_mu_labels(rp, su):
    md = rp.render_markdown({"suite_freeze_sha256": "f" * 64, "n_rows": 1, "n_systems": 1},
                            suite=su.Suite())
    assert "No unseen-map or generalisation claim" in md
    assert "range midpoint" in md and "1.0489" in md
    assert "No composite score" in md


# ---------------------------------------------------------------- CLI smoke

def _cli(*args, cwd):
    # The package is importable as part of f1sim here, so the child needs the repo on the path,
    # not a work directory.
    import f1sim
    repo = os.path.dirname(os.path.dirname(os.path.abspath(f1sim.__file__)))
    env = {"PYTHONPATH": repo, "PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, "-m", "f1sim.learn.benchmark", *args],
                          capture_output=True, text=True, cwd=cwd, env=env, timeout=120)


def test_cli_plan_runs_without_a_suite_file(tmp_path):
    r = _cli("plan", "--suite", str(tmp_path / "none.json"), cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "trials/system 272" in r.stdout
    assert "not frozen" in r.stdout


def _real_ckpt(tmp_path, name="ck.pt", arm="legacy"):
    """A genuine torch checkpoint, so the arm probe has something real to read."""
    torch = pytest.importorskip("torch")
    p = tmp_path / name
    # The production contract: the arm lives at extra.experiment.controller.arm, which is what
    # `model.controller_arm_of` reads and what the roster probe now reads.
    torch.save({"state_dict": {"w": torch.zeros(2)},
                "extra": {"experiment": {"controller": {"arm": arm}}}}, str(p))
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def test_cli_run_refuses_without_gpu_lease(su, ro, tmp_path):
    sp = tmp_path / "suite.json"
    s = su.Suite()
    s.placements = {"gen:control:1400": {"placement": {}, "proofs": {}}}
    s.save(str(sp))
    path, sha = _real_ckpt(tmp_path)
    rp_ = tmp_path / "roster.json"
    rp_.write_text(json.dumps({"systems": [
        {"system_id": "sys", "path": path, "checkpoint_sha256": sha, "controller_arm": "legacy"}]}))
    r = _cli("run", "--suite", str(sp), "--roster", str(rp_), "--system", "sys", cwd=str(tmp_path))
    assert r.returncode == 3
    assert "--lease" in r.stderr


def test_cli_run_refuses_before_geometry_freeze(su, tmp_path):
    sp = tmp_path / "suite.json"
    su.Suite().save(str(sp))                     # no placements
    path, sha = _ckpt(tmp_path)
    rp_ = tmp_path / "roster.json"
    rp_.write_text(json.dumps({"systems": [
        {"system_id": "sys", "path": path, "checkpoint_sha256": sha, "controller_arm": "legacy"}]}))
    r = _cli("run", "--suite", str(sp), "--roster", str(rp_), "--system", "sys", cwd=str(tmp_path))
    assert r.returncode == 1
    assert "geometry --freeze" in r.stderr


def test_cli_end_to_end_freeze_then_report(su, tmp_path):
    """geometry-free freeze -> synthetic rows -> rendered leaderboard. No candidate is loaded."""
    sp = tmp_path / "suite.json"
    s = su.Suite(solo_maps=("m1",), obstacle_maps=(), race_maps=(), solo_mus=(su.MU_LOW,),
                 seeds=(4401,), envs=8)
    s.placements = {"m1": {"placement": {}, "proofs": {}}}
    freeze = s.save(str(sp))
    assert s.expected_trials()["S"] == 8

    path, sha = _ckpt(tmp_path)
    rj = tmp_path / "roster.json"
    rj.write_text(json.dumps({"systems": [
        {"system_id": "sys", "path": path, "checkpoint_sha256": sha, "controller_arm": "legacy"}]}))

    rows = [{"system_id": "sys", "checkpoint_sha256": sha, "controller_arm": "legacy",
             "suite_freeze_sha256": freeze, "suite_version": s.version, "map_id": "m1",
             "mu": su.MU_LOW, "seed": 4401, "n_envs": 8, "suite": "S",
             # the declared protocol the runner stamps on every row (suite.py:176-178); a fixture
             # without it is not a row the runner could have written
             "speed_cap": s.speed_cap, "budget_laps": s.budget_laps,
             "sensor_noise": s.sensor_noise, "backend": s.backend,
             "source_digest": {"sim.py": "abc"}}]
    res = tmp_path / "results.json"
    # The full producer schema: lap time present exactly on the completions, the cell this row
    # claims, and the effective configuration the run reported. A fixture missing any of these is
    # not a row the runner could have written, and the validator now says so.
    cells = [dict(rows[0], runtime="legacy", cell_id=f"S:m1:{su.MU_LOW}:4401", result={
        "n": 8, "tally": {"successes": 4, "denominator": 8, "failures": {"collision": 4}},
        "progress_m": [10.0] * 8, "distance_m": [10.0] * 8,
        "route_progress_fraction": [0.2] * 8,
        "lap_time_s": [10.0] * 4 + [None] * 4,
        "effective": {"true_mu": su.MU_LOW, "plant_mu": su.MU_LOW},
        # the start this cell was measured from; the runner records it after the final seeded
        # reset and before the controller begins, so a row without it is not one it could write
        "start_fingerprint": {
            "schema_version": 1,
            "physical_sha256": "a" * 64, "physical_tensors": 20,
            "actor_input_sha256": "b" * 64, "actor_input_tensors": 6,
            "calibration_sha256": "c" * 64, "calibration_tensors": 8,
            "obs_spec_sha256": "d" * 64,
            # This is a SOLO S cell, so race_size is 1 -- suite.race_size (2) is the OVERTAKING
            # size only. Stamping the suite value here is exactly the bug that refused every solo
            # cell, and a fixture that copies it would hide the fix.
            "sim_t": 0.3, "sim_imu_phase": 1,
            "n_envs_total": 8, "race_size": 1},
        "outcomes": [{"success": True, "reason": None}] * 4 +
                    [{"success": False, "reason": "collision"}] * 4})]
    res.write_text(json.dumps({"cells": cells}))

    out = tmp_path / "lb.md"
    r = _cli("report", "--suite", str(sp), "--roster", str(rj), "--results", str(res),
             "--out", str(out), cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    md = out.read_text()
    assert "0.500" in md                                  # derived from raw cells, not handed in
    assert "N/A (category not evaluated)" in md           # overtaking was never run
    assert "No composite score" in md


# -- reviewer findings HIGH2 / MEDIUM3 ----------------------------------------------------------

def test_immutable_baseline_named_latest_frozen_is_accepted(ro, tmp_path):
    """HIGH2 regression: the ban was on the substring, which rejected the required baseline.

    `ppo_latest_frozen.pt` is an immutable copy. Only unresolved aliases are refused; the bytes
    settle the rest, and they are hashed.
    """
    path, sha = _ckpt(tmp_path, name="ppo_latest_frozen.pt")
    ro.Entry("frozen_original@legacy", path, sha, "legacy").verify()


def test_moving_pointer_basename_is_still_refused(ro, tmp_path):
    path, sha = _ckpt(tmp_path, name="ppo_latest.pt")
    with pytest.raises(ValueError, match="rewritten in place by training"):
        ro.Entry("s", path, sha, "legacy").verify()


def test_latest_path_segment_is_still_refused(ro, tmp_path):
    d = tmp_path / "latest"
    d.mkdir()
    path, sha = _ckpt(d, name="w.pt")
    with pytest.raises(ValueError, match="unresolved 'latest' path segment"):
        ro.Entry("s", path, sha, "legacy").verify()


def test_estimated_entry_with_bare_sha_is_refused(ro, tmp_path):
    """MEDIUM3 regression: a declared sha with no file is not a pin; the runtime cannot load it."""
    path, sha = _ckpt(tmp_path)
    e = ro.Entry("s", path, sha, "estimated", estimator_path=None, estimator_sha256="f" * 64)
    with pytest.raises(ValueError, match="needs an estimator_path"):
        e.verify()


def test_estimated_entry_with_missing_estimator_file_is_refused(ro, tmp_path):
    path, sha = _ckpt(tmp_path)
    e = ro.Entry("s", path, sha, "estimated",
                 estimator_path=str(tmp_path / "nope.pt"), estimator_sha256="f" * 64)
    with pytest.raises(FileNotFoundError):
        e.verify()


def test_estimated_entry_with_drifted_estimator_sha_is_refused(ro, tmp_path):
    path, sha = _ckpt(tmp_path)
    est, _ = _ckpt(tmp_path, name="est.pt", body=b"estimator")
    e = ro.Entry("s", path, sha, "estimated", estimator_path=est, estimator_sha256="f" * 64)
    with pytest.raises(ValueError, match="estimator sha mismatch"):
        e.verify()


def test_estimated_entry_fully_pinned_is_accepted(ro, tmp_path):
    path, sha = _ckpt(tmp_path)
    est, esha = _ckpt(tmp_path, name="est.pt", body=b"estimator")
    ro.Entry("s", path, sha, "estimated", estimator_path=est, estimator_sha256=esha).verify()


def test_relative_estimator_and_checkpoint_paths_resolve(ro, tmp_path, monkeypatch):
    """A relative path names a real file once canonicalised; absoluteness was never the point."""
    path, sha = _ckpt(tmp_path)
    est, esha = _ckpt(tmp_path, name="est.pt", body=b"estimator")
    monkeypatch.chdir(tmp_path)
    e = ro.Entry("s", "./w.pt", sha, "estimated", estimator_path="./est.pt", estimator_sha256=esha)
    e.verify()
    assert e.resolved().endswith("w.pt")


def test_probe_uses_weights_only_and_reads_metadata(ro, tmp_path):
    """The probe must not need to execute anything in the file."""
    torch = pytest.importorskip("torch")
    p = tmp_path / "ck.pt"
    torch.save({"state_dict": {"w": torch.zeros(3, 4)},
                "meta": {"act_dim": 8},
                "extra": {"experiment": {"controller": {"arm": "estimated"}},
                          "spec": {"n_beams": 1081}}}, str(p))
    info = ro.probe(str(p))
    assert info["recorded_arm"] == "estimated"
    assert info["n_params"] == 12 and info["n_beams"] == 1081


def test_declared_arm_must_match_the_recorded_arm(ro, tmp_path):
    torch = pytest.importorskip("torch")
    p = tmp_path / "ck.pt"
    torch.save({"state_dict": {},
                "extra": {"experiment": {"controller": {"arm": "estimated"}}}}, str(p))
    body = p.read_bytes()
    e = ro.Entry("s", str(p), hashlib.sha256(body).hexdigest(), "legacy")
    with pytest.raises(ValueError, match="trained under 'estimated'"):
        ro.check_arm_matches_record(e)


def test_aggregate_pools_counts_not_rates(rp):
    """Cells differ in size; a mean of rates would weight a short cell like a full one."""
    def cell(succ, den, n):
        return {"system_id": "s", "runtime": "legacy", "suite": "S", "mu": 0.73423,
                "result": {"n": n, "tally": {"successes": succ, "denominator": den, "failures": {}},
                           "progress_m": [1.0] * n, "route_progress_fraction": [0.1] * n,
                           # the producer always emits travelled distance; there is no fallback
                           "distance_m": [1.0] * n,
                           "lap_time_s": [10.0] * succ + [None] * (n - succ),
                           "outcomes": ([{"success": True, "reason": None}] * succ
                                        + [{"success": False, "reason": "collision"}]
                                        * (n - succ))}}
    # The fixture now MEANS what its tally says. Before, it declared 0 successes while every
    # outcome said success, and only passed because aggregation trusted the tally over the
    # measurement -- which is the defect this change closes.
    agg = rp.aggregate(_validated(cell(1, 1, 1), cell(0, 7, 7)))
    # pooled 1/8, not mean(1.0, 0.0) = 0.5
    assert agg["driving"][0]["values"]["completion ↑"]["value"] == pytest.approx(1 / 8)


def test_aggregate_reports_real_exposure_counters(rp):
    """Spins and slip now come from TrialAccumulator, threaded through the live loop."""
    agg = rp.aggregate(_validated({"system_id": "s", "runtime": "legacy", "suite": "S", "mu": 0.73423,
                         "result": {"n": 1, "tally": {"successes": 0, "denominator": 1,
                                                      "failures": {"collision": 1}},
                                    "progress_m": [10.0], "distance_m": [10.0],
                                    "spin_events": 2, "large_slip_seconds": 1.0,
                                    "max_abs_yaw_rate": 3.5,
                                    "route_progress_fraction": [0.1],
                                    "lap_time_s": [None],
                                    "outcomes": [{"success": False, "reason": "collision"}]}}))
    vals = agg["stability"][0]["values"]
    assert vals["spins ↓"]["value"] == 2
    # diagnostic, not lower-is-better: a high yaw rate on a tight corner is the corner
    assert vals["max yaw rate rad/s (diagnostic)"]["value"] == pytest.approx(3.5)
    assert vals["collisions/km ↓"]["value"] == pytest.approx(100.0)


def test_report_reads_raw_cells_jsonl(tmp_path, su, rp):
    """cells.jsonl is the raw artefact; the summary is derived, never hand-written.

    `_read_results` no longer aggregates -- aggregation happens after validation in `cmd_report` --
    and it no longer synthesises `n_envs` from `result.n`, so a row carries the cardinality its
    producer recorded and the validator can catch a disagreement.
    """
    from f1sim.learn.benchmark.__main__ import _read_results
    p = tmp_path / "cells.jsonl"
    # One success and one collision: the tally MEANS what the outcomes say. The earlier fixture
    # declared 1 success while both outcomes said success, and only agreed with the reported 0.5
    # because aggregation trusted the tally over the measurement.
    p.write_text(json.dumps({
        "system_id": "s", "runtime": "legacy", "suite": "S", "map_id": "m", "mu": 0.73423,
        "seed": 4401, "n_envs": 2, "cell_id": "S:m:0.73423:4401",
        "result": {"n": 2, "tally": {"successes": 1, "denominator": 2,
                                     "failures": {"collision": 1}},
                   "progress_m": [1.0, 2.0], "distance_m": [1.0, 2.0],
                   "route_progress_fraction": [0.1, 0.2], "lap_time_s": [10.0, None],
                   "outcomes": [{"success": True, "reason": None},
                                {"success": False, "reason": "collision"}]}}) + "\n")
    got = _read_results(str(p))
    assert got["rows"][0]["n_envs"] == 2
    assert "summary" not in got, "aggregation must not happen before validation"
    summary = rp.aggregate(_validated(*got["cells"]))
    assert summary["driving"][0]["values"]["completion ↑"]["value"] == pytest.approx(0.5)
