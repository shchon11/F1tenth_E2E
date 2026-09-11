"""Start fingerprints: captured at the right moment, over the right tensors, persisted per cell."""
from __future__ import annotations
import importlib
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("f1sim.gym_env")


from .test_model_adapter import make_checkpoint, TEST_MAP, MU_LOW          # noqa: E402

SEED = 4401

def _good():
    return {"schema_version": 1, "physical_sha256": "a" * 64, "actor_input_sha256": "b" * 64,
            "calibration_sha256": "c" * 64, "physical_tensors": 3, "actor_input_tensors": 2,
            "calibration_tensors": 1, "n_envs_total": 8, "race_size": 2, "sim_t": 0.0,
            "sim_imu_phase": 0, "obs_spec_sha256": "d" * 64}




@pytest.fixture
def fp(bench):
    return importlib.import_module("f1sim.learn.benchmark.fingerprint")


@pytest.fixture
def rn(bench):
    return importlib.import_module("f1sim.learn.benchmark.runner")


@pytest.fixture
def ma(bench):
    return importlib.import_module("f1sim.learn.benchmark.model_adapter")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    p, spec = make_checkpoint(tmp_path_factory.mktemp("fp"), "fp_legacy")
    return {"path": p, "arm": "legacy"}, {"spec": spec, "cap": 9.0}


def _prep(ma, ckpt, seed=SEED, race_size=1, envs=2):
    entry, extra = ckpt
    suite = {"envs": envs, "speed_cap": 9.0, "sensor_noise": True, "budget_laps": 0.2,
             "race_size": race_size, "opponent": "teacher", "opp_speed_range": [0.6, 0.8]}
    return ma.prepare_cell(entry, extra, {"map": TEST_MAP, "true_mu": MU_LOW, "seed": seed,
                                          "envs": envs, "race_size": race_size}, suite, "cpu")


def _capture(fp, rn, ma, ckpt, *, seed=SEED, rng_noise=0, race_size=1, spec=None):
    prep = _prep(ma, ckpt, seed=seed, race_size=race_size)
    try:
        torch.manual_seed(77)
        for _ in range(rng_noise):
            torch.rand(13)
        obs = rn._reset_obs(prep.env, seed)
        return fp.start_fingerprint(prep.env, obs, obs_spec=spec)
    finally:
        prep.close()


def test_a_real_capture_covers_all_three_layers(fp, rn, ma, ckpt):
    got = _capture(fp, rn, ma, ckpt)
    for k in ("physical_sha256", "actor_input_sha256", "calibration_sha256"):
        assert got[k] and len(got[k]) == 64
    assert got["physical_tensors"] > 0 and got["actor_input_tensors"] > 0
    assert got["schema_version"] == fp.SCHEMA_VERSION
    assert "sim_t" in got and "sim_imu_phase" in got


def test_a_digest_over_nothing_is_refused(fp):
    """A hash of zero tensors is perfectly stable and would read as agreement."""
    class Empty:
        class sim:
            P = {}
        B, M = 0, 1
    with pytest.raises(ValueError, match="digest over none of them"):
        fp.start_fingerprint(Empty(), {})


def test_the_same_seed_pairs_despite_a_disturbed_global_rng(fp, rn, ma, ckpt):
    spec = {"scan_stack": 6, "hist_len": 20}
    a = _capture(fp, rn, ma, ckpt, rng_noise=0, spec=spec)
    b = _capture(fp, rn, ma, ckpt, rng_noise=31, spec=spec)
    got = fp.compare(a, b)
    assert got["paired"], got["differs"]


def test_a_different_seed_does_not_pair(fp, rn, ma, ckpt):
    """Otherwise the pairing assertion would pass on a constant and prove nothing."""
    spec = {"scan_stack": 6, "hist_len": 20}
    a = _capture(fp, rn, ma, ckpt, seed=SEED, spec=spec)
    b = _capture(fp, rn, ma, ckpt, seed=SEED + 1, spec=spec)
    got = fp.compare(a, b)
    assert not got["paired"]
    assert "physical" in got["differs"]


def test_a_race_cell_fingerprints_every_car_not_only_the_learner(fp, rn, ma, ckpt):
    solo = _capture(fp, rn, ma, ckpt, race_size=1)
    race = _capture(fp, rn, ma, ckpt, race_size=2)
    assert race["race_size"] == 2 and race["n_envs_total"] > solo["n_envs_total"]
    assert race["physical_sha256"] != solo["physical_sha256"]


def test_obs_spec_disagreement_breaks_pairing(fp, rn, ma, ckpt):
    """Two systems can hash identical observations while expecting different layouts."""
    a = _capture(fp, rn, ma, ckpt, spec={"scan_stack": 6, "hist_len": 20})
    b = _capture(fp, rn, ma, ckpt, spec={"scan_stack": 3, "hist_len": 20})
    got = fp.compare(a, b)
    assert not got["paired"] and "obs_spec" in got["differs"]


def test_schema_mismatch_is_refused_rather_than_compared(fp):
    """A different schema is a different definition; comparing them is not a comparison."""
    a = _good()
    b = dict(a, schema_version=a["schema_version"] + 1)
    with pytest.raises(ValueError, match="schema"):
        fp.compare(a, b)


def test_run_cell_persists_the_fingerprint_in_its_row(fp, rn, ma, ckpt):
    """The whole point: it must survive into the record, not only exist in-process."""
    prep = _prep(ma, ckpt)
    try:
        res = rn.run_cell(prep.env, lambda o: torch.zeros(prep.env.B, prep.env.act_dim),
                          suite="S", n_steps=3, frozen=False, controller=prep, seed=SEED,
                          obs_spec={"scan_stack": 6})
    finally:
        prep.close()
    got = res["start_fingerprint"]
    assert got["physical_sha256"] and got["obs_spec_sha256"]
    assert got["schema_version"] == fp.SCHEMA_VERSION


def test_assert_paired_refuses_rows_without_a_fingerprint(fp):
    rows = [{"system_id": "a", "result": {"start_fingerprint": _good()}},
            {"system_id": "b", "result": {}}]
    with pytest.raises(ValueError, match="no start fingerprint"):
        fp.assert_paired(rows, cell_id="S:m:0.7:1")


def test_assert_paired_names_the_layer_that_differs(fp):
    base = _good()
    other = dict(base)
    other["actor_input_sha256"] = "e" * 64
    rows = [{"system_id": "a", "result": {"start_fingerprint": base}},
            {"system_id": "b", "result": {"start_fingerprint": other}}]
    with pytest.raises(ValueError, match="actor_input"):
        fp.assert_paired(rows, cell_id="S:m:0.7:1")


def test_assert_paired_validates_a_lone_row_rather_than_calling_it_paired(fp):
    """The one row that never reaches `compare`, and so was never validated.

    A single row has nothing to be compared against, so the loop over later rows never runs. A row
    carrying only a well-formed obs_spec_sha256 — no physical digest, no tensor counts, no sim_t —
    used to come back `paired: True`, which reads as "this start was verified" when nothing about it
    was ever looked at.
    """
    rows = [{"system_id": "a", "result": {"start_fingerprint": {"obs_spec_sha256": "f" * 64}}}]
    with pytest.raises(ValueError):
        fp.assert_paired(rows, cell_id="S:m:0.7:1")


def test_assert_paired_validates_the_base_row_not_only_its_neighbours(fp):
    """The same hole with more rows: base is the one side `compare` never checks on its own."""
    good = _good()
    rows = [{"system_id": "a", "result": {"start_fingerprint": {"obs_spec_sha256": "f" * 64}}},
            {"system_id": "b", "result": {"start_fingerprint": good}}]
    with pytest.raises(ValueError):
        fp.assert_paired(rows, cell_id="S:m:0.7:1")


def test_assert_paired_accepts_a_genuinely_paired_set(fp):
    one = _good()
    rows = [{"system_id": s, "result": {"start_fingerprint": dict(one)}} for s in ("a", "b", "c")]
    assert fp.assert_paired(rows, cell_id="S:m:0.7:1")["paired"] is True


def test_rows_spanning_two_obs_specs_are_grouped_not_pooled(fp):
    """Different input layouts are a legitimate benchmark shape, but not one population."""
    def row(sid, spec):
        return {"system_id": sid, "result": {"start_fingerprint": dict(_good(),
                                                                       obs_spec_sha256=spec)}}
    rows = [row("a", "spec1"), row("b", "spec1"), row("c", "spec2")]
    groups = fp.group_by_obs_spec(rows)
    assert len(groups) == 2 and sorted(groups["spec1"]) == ["a", "b"]
    with pytest.raises(ValueError, match="incompatible observation layouts"):
        fp.assert_single_obs_spec(rows, cell_id="S:m:0.7:1")


def test_a_row_without_an_obs_spec_is_refused_not_assumed_to_match(fp):
    """Absent must not pass the test that present has to pass."""
    def row(sid, spec):
        fpd = _good()
        if spec:
            fpd["obs_spec_sha256"] = spec
        else:
            fpd.pop("obs_spec_sha256")
        return {"system_id": sid, "result": {"start_fingerprint": fpd}}
    with pytest.raises(ValueError, match="recorded no ObsSpec"):
        fp.assert_single_obs_spec([row("a", "s"), row("b", None)], cell_id="S:m:0.7:1")


def test_one_shared_obs_spec_passes(fp):
    """The state today: all systems share one layout, so this returns a single group."""
    rows = [{"system_id": s, "result": {"start_fingerprint": dict(_good(),
                                                                  obs_spec_sha256="f" * 64)}}
            for s in ("a", "b", "c")]
    assert fp.assert_single_obs_spec(rows, cell_id="S:m:0.7:1") == "f" * 64


# ---- digest fidelity: three ways two different starts used to hash the same -------------------

def test_a_tiny_float64_difference_is_not_erased_by_a_cast(fp):
    """Casting to float32 first made values 1e-10 apart identical, so a plant state that genuinely
    differed read as paired."""
    a, _ = fp._digest([("x", torch.tensor([1.0], dtype=torch.float64))])
    b, _ = fp._digest([("x", torch.tensor([1.0 + 1e-10], dtype=torch.float64))])
    assert a != b


def test_shape_is_part_of_the_digest(fp):
    """(2, 2) and (4,) holding the same numbers are different states."""
    a, _ = fp._digest([("x", torch.ones(2, 2))])
    b, _ = fp._digest([("x", torch.ones(4))])
    assert a != b


def test_dtype_is_part_of_the_digest(fp):
    a, _ = fp._digest([("x", torch.ones(3, dtype=torch.int64))])
    b, _ = fp._digest([("x", torch.ones(3, dtype=torch.float64))])
    assert a != b


def test_the_field_name_is_part_of_the_digest(fp):
    """Otherwise a value moving between fields cancels out."""
    v = torch.ones(3)
    a, _ = fp._digest([("alpha", v)])
    b, _ = fp._digest([("beta", v)])
    assert a != b


def test_identical_tensors_still_agree(fp):
    """The fidelity checks must not make everything differ."""
    a, na = fp._digest([("x", torch.arange(6, dtype=torch.float64).reshape(2, 3))])
    b, nb = fp._digest([("x", torch.arange(6, dtype=torch.float64).reshape(2, 3))])
    assert a == b and na == nb == 1


# ---- validation before comparison: None == None must not pass --------------------------------

def test_two_empty_fingerprints_are_refused_not_paired(fp):
    """The strongest possible false positive: it claims two systems started identically when
    neither recorded where it started."""
    with pytest.raises(ValueError, match="empty or missing"):
        fp.compare({}, {})


@pytest.mark.parametrize("mutate,match", [
    ({"schema_version": 99}, "schema"),
    ({"physical_sha256": "short"}, "not a 64-character hex"),
    ({"actor_input_sha256": "z" * 64}, "not a 64-character hex"),
    ({"physical_tensors": 0}, "digest over none"),
    ({"actor_input_tensors": 0}, "digest over none"),
    ({"calibration_tensors": 0}, "digest over none"),
    ({"n_envs_total": None}, "n_envs_total"),
])
def test_a_malformed_fingerprint_is_refused(fp, mutate, match):
    bad = dict(_good()); bad.update(mutate)
    with pytest.raises(ValueError, match=match):
        fp.validate_fingerprint(bad)


@pytest.mark.parametrize("drop", ["sim_t", "sim_imu_phase", "obs_spec_sha256"])
def test_an_absent_recorded_value_is_refused(fp, drop):
    """An unrecorded value cannot be shown to match one that was recorded."""
    bad = dict(_good()); bad.pop(drop)
    with pytest.raises(ValueError, match=drop):
        fp.validate_fingerprint(bad)


def test_a_well_formed_fingerprint_passes(fp):
    fp.validate_fingerprint(_good())
    assert fp.compare(_good(), _good())["paired"] is True


def test_a_real_capture_is_well_formed(fp, rn, ma, ckpt):
    """The producer's own output must satisfy the validator, not just the synthetic fixtures."""
    got = _capture(fp, rn, ma, ckpt, spec={"scan_stack": 6, "hist_len": 20})
    fp.validate_fingerprint(got)
    assert got["calibration_tensors"] >= 1


# ---- type and shape strictness: four ways a malformed start used to validate ------------------

def test_a_bool_is_not_a_tensor_count(fp):
    """`isinstance(True, int)` is True in Python, so a plain int check read True as one tensor."""
    for k in ("physical_tensors", "actor_input_tensors", "calibration_tensors"):
        with pytest.raises(ValueError, match="bool is not a count|covered True"):
            fp.validate_fingerprint(dict(_good(), **{k: True}))


def test_a_bool_is_not_an_env_count(fp):
    for k in ("n_envs_total", "race_size"):
        with pytest.raises(ValueError, match="positive integer"):
            fp.validate_fingerprint(dict(_good(), **{k: True}))


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_clock_is_refused(fp, bad):
    """inf compares equal to itself and nan never does; neither is evidence two runs started
    together."""
    with pytest.raises(ValueError, match="not a finite clock reading"):
        fp.validate_fingerprint(dict(_good(), sim_t=bad))


def test_the_fingerprint_is_bound_to_the_declared_cell(fp, bench):
    """A fingerprint recording 16 cars in a race of 4 is well formed on its own terms and still
    describes a different experiment from the 8-learner solo cell it claims to be."""
    from f1sim.learn.benchmark.suite import Cell, Suite
    cell, suite = Cell("S", "m", 0.73423, 4401, 8), Suite()
    fp.validate_fingerprint(dict(_good(), n_envs_total=8, race_size=1),
                            expected_cell=cell, suite=suite)          # the honest one passes
    with pytest.raises(ValueError, match="race_size 4 but the declared cell"):
        fp.validate_fingerprint(dict(_good(), n_envs_total=16, race_size=4),
                                expected_cell=cell, suite=suite)
    with pytest.raises(ValueError, match="n_envs_total 99"):
        fp.validate_fingerprint(dict(_good(), n_envs_total=99, race_size=1),
                                expected_cell=cell, suite=suite)


def test_a_race_cell_expects_learners_times_race_size(fp, bench):
    from f1sim.learn.benchmark.suite import Cell, Suite
    cell, suite = Cell("O", "m", 0.73423, 4401, 8), Suite()
    fp.validate_fingerprint(dict(_good(), n_envs_total=16, race_size=2),
                            expected_cell=cell, suite=suite)
    with pytest.raises(ValueError, match="n_envs_total 8 "):
        fp.validate_fingerprint(dict(_good(), n_envs_total=8, race_size=2),
                                expected_cell=cell, suite=suite)


def test_two_differently_shaped_runs_do_not_pair(fp):
    """Even when every digest matches: a solo cell and a 4-car race are different measurements."""
    got = fp.compare(_good(), dict(_good(), n_envs_total=16, race_size=4))
    assert not got["paired"]
    assert set(got["differs"]) >= {"n_envs_total", "race_size"}


# ---- one validator, one definition -----------------------------------------------------------
#
# These went through the shared helper AND, where the report exposes it, through the report's own
# entry point. Two copies of a rule drift; the point of consolidating is that a regression here
# covers both callers.

@pytest.mark.parametrize("bad,why", [
    (True, "a bool: True == 1 in Python, so a bare equality accepts it"),
    (1.0, "a float that happens to equal the version"),
    (1.9, "a fractional version an int() cast elsewhere would round into agreement"),
    ("1", "a string"),
    (None, "absent"),
])
def test_schema_version_must_be_the_exact_integer(fp, bad, why):
    with pytest.raises(ValueError, match="schema_version"):
        fp.validate_fingerprint(dict(_good(), schema_version=bad))


@pytest.mark.parametrize("bad", ["short", "z" * 64, "D" * 64, "", None, 12345])
def test_the_obs_spec_hash_must_be_a_real_digest(fp, bad):
    """Presence alone let a row record a placeholder and still claim a matching layout."""
    with pytest.raises(ValueError, match="obs_spec_sha256 is not a 64-character hex digest"):
        fp.validate_fingerprint(dict(_good(), obs_spec_sha256=bad))


def test_an_absent_obs_spec_hash_is_refused(fp):
    bad = dict(_good()); bad.pop("obs_spec_sha256")
    with pytest.raises(ValueError, match="obs_spec_sha256"):
        fp.validate_fingerprint(bad)


@pytest.mark.parametrize("field", ["physical", "actor_input", "calibration"])
def test_every_digest_must_be_real_hex(fp, field):
    with pytest.raises(ValueError, match=f"{field}_sha256 is not a 64-character hex"):
        fp.validate_fingerprint(dict(_good(), **{f"{field}_sha256": "Q" * 64}))


def test_the_obs_spec_hash_needs_no_tensor_count(fp):
    """It hashes a spec, not tensors; demanding a count for it would refuse every real row."""
    good = _good()
    assert "obs_spec_tensors" not in good
    fp.validate_fingerprint(good)


def test_the_report_and_the_helper_agree_on_every_malformed_case(fp, bench):
    """Behavioural consolidation check: whatever the report does internally, the two must not
    disagree about what is malformed. Asserting that a function NAME appears in the report's source
    would only test a spelling -- the thing that matters is that no row is accepted by one and
    refused by the other."""
    from f1sim.learn.benchmark import report
    from f1sim.learn.benchmark.suite import Cell, Suite
    cell, suite = Cell("S", "m", 0.73423, 4401, 1), Suite()
    cases = [dict(_good(), schema_version=True), dict(_good(), schema_version=1.9),
             dict(_good(), obs_spec_sha256="short"), dict(_good(), physical_tensors=0),
             dict(_good(), sim_t=float("inf")), dict(_good(), calibration_sha256="Q" * 64)]
    for bad in cases:
        helper_refused = True
        try:
            fp.validate_fingerprint(bad)
            helper_refused = False
        except ValueError:
            pass
        report_refused = True
        try:
            report.validate_cell(_report_row(bad), cell, suite=suite)
            report_refused = False
        except Exception:
            pass
        assert helper_refused and report_refused, (
            f"disagreement on {bad.get('schema_version')!r}/{bad.get('obs_spec_sha256','')[:8]}: "
            f"helper refused={helper_refused}, report refused={report_refused}")


def _report_row(fingerprint):
    return {"system_id": "sys", "checkpoint_sha256": "a" * 64, "controller_arm": "legacy",
            "suite_freeze_sha256": "f" * 64, "suite_version": "v1", "map_id": "m",
            "mu": 0.73423, "seed": 4401, "n_envs": 1, "suite": "S",
            "cell_id": "S:m:0.73423:4401", "source_digest": {"sim.py": "x"},
            "result": {"n": 1, "tally": {"successes": 1, "denominator": 1, "failures": {}},
                       "outcomes": [{"success": True, "reason": None}],
                       "progress_m": [1.0], "distance_m": [1.0],
                       "route_progress_fraction": [0.02], "lap_time_s": [0.5],
                       "effective": {"plant_mu": 0.73423, "true_mu": 0.73423, "speed_cap": 9.0,
                                     "budget_laps": 3.0, "sensor_noise": True,
                                     "backend": "f1sim"},
                       "start_fingerprint": fingerprint}}


@pytest.mark.parametrize("bad", [True, 1.9, "1"])
def test_a_bad_schema_is_refused_through_the_report_too(bench, bad):
    """The same malformed row, through the report's public entry point."""
    from f1sim.learn.benchmark import report
    from f1sim.learn.benchmark.suite import Cell, Suite
    with pytest.raises(report.ReportError):
        report.validate_cell(_report_row(dict(_good(), schema_version=bad)),
                             Cell("S", "m", 0.73423, 4401, 1), suite=Suite())
