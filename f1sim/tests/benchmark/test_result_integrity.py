"""Regressions for the report/roster defects an independent review reproduced.

One test per defect, each built from a CONTROL row that passes, so a refusal is the mutation's own
and not a shared artefact. Synthetic raw only; no checkpoint scoring, no GPU, no live evidence.
"""
import copy
import importlib

import pytest


@pytest.fixture
def rp():
    return importlib.import_module("f1sim.learn.benchmark.report")


@pytest.fixture
def ro():
    return importlib.import_module("f1sim.learn.benchmark.roster")


class Cell:
    """Minimal stand-in for suite.Cell: only what validate_cell reads."""

    def __init__(self, envs=8, mu=0.94401, suite="S", map_id="m", seed=4401):  # noqa: D401
        self.envs, self.mu, self.true_mu = envs, mu, mu
        self.suite, self.map_id, self.seed = suite, map_id, seed


class Suite:
    """Minimal stand-in for the declared suite: the protocol validate_cell compares against."""

    speed_cap, budget_laps, sensor_noise, backend = 9.0, 3.0, True, "f1sim"
    race_size = 1


N = 8


def fingerprint(n=None, **over):
    """A start fingerprint shaped as `fingerprint.start_fingerprint` writes it.

    Required metadata now, so a fixture without it is not a row the runner could have produced.
    The tensor counts are real values rather than placeholders: a digest over zero tensors is
    perfectly stable and would read as agreement for every system.
    """
    from f1sim.learn.benchmark import fingerprint as _fp
    fp = {"schema_version": _fp.SCHEMA_VERSION,
          "physical_sha256": "a" * 64, "physical_tensors": 20,
          "actor_input_sha256": "b" * 64, "actor_input_tensors": 6,
          "calibration_sha256": "c" * 64, "calibration_tensors": 8,
          "obs_spec_sha256": "d" * 64,
          "sim_t": 0.3, "sim_imu_phase": 1,
          # scales with the cell: the population must BE the declared one, not a constant
          "n_envs_total": (N if n is None else n), "race_size": 1}
    fp.update(over)
    return fp


def control(n=N):
    return {"system_id": "sys", "cell_id": "S:m:0.94401:4401", "runtime": "legacy",
            "controller_arm": "legacy", "suite": "S", "map_id": "m", "mu": 0.94401,
            "seed": 4401, "n_envs": n,
            # the declared protocol the runner writes on every row (suite.py:176-178)
            "speed_cap": 9.0, "budget_laps": 3.0, "sensor_noise": True, "backend": "f1sim",
            "result": {"n": n,
                       "outcomes": [{"success": True, "reason": None}] * n,
                       "lap_time_s": [10.0] * n, "progress_m": [50.0] * n,
                       "distance_m": [50.0] * n, "route_progress_fraction": [1.0] * n,
                       "cross_track_rms_m": [0.2] * n, "cross_track_samples": [100] * n,
                       "spin_events": 0, "large_slip_seconds": 0.0, "max_abs_yaw_rate": 1.0,
                       "effective": {"true_mu": 0.94401, "plant_mu": 0.94401},
                       "start_fingerprint": fingerprint(n),
                       "tally": {"successes": n, "denominator": n, "failures": {}}}}


def mutated(**over):
    r = copy.deepcopy(control())
    r["result"].update(over)
    return r


def test_control_passes(rp):
    """Every test below is this row with one change; if the control itself failed they would
    all pass for the wrong reason and prove nothing."""
    rp.validate_cell(control(), Cell())


# ---- D1: schema, domains, contradictions
@pytest.mark.parametrize("over, match", [
    ({"large_slip_seconds": -12.0}, "negative"),
    ({"spin_events": -4}, "negative"),
    ({"lap_time_s": [-5.0] * N}, "negative"),
    ({"route_progress_fraction": [float("nan")] * N}, "not finite"),
    ({"max_abs_yaw_rate": float("inf")}, "not finite"),
])
def test_invalid_domains_refused(rp, over, match):
    with pytest.raises(rp.ReportError, match=match):
        rp.validate_cell(mutated(**over), Cell())


def test_required_array_absent_refused(rp):
    r = control()
    del r["result"]["route_progress_fraction"]
    with pytest.raises(rp.ReportError, match="required per-trial array"):
        rp.validate_cell(r, Cell())


def test_every_array_is_width_checked(rp):
    """Not just progress_m/distance_m: lap_time_s at length 3 against n=8 used to be accepted."""
    with pytest.raises(rp.ReportError, match="entries for n=8"):
        rp.validate_cell(mutated(lap_time_s=[10.0] * 3), Cell())


def test_signed_route_progress_survives(rp):
    """Reversing must subtract. An abs() here would silently reward driving backwards."""
    rp.validate_cell(mutated(route_progress_fraction=[-0.4] * N), Cell())


def test_lap_time_required_on_completions_and_absent_on_failures(rp):
    fail_all = {"outcomes": [{"success": False, "reason": "collision"}] * N,
                "tally": {"successes": 0, "denominator": N, "failures": {"collision": N}}}
    with pytest.raises(rp.ReportError, match="did not complete but carries lap time"):
        rp.validate_cell(mutated(**fail_all), Cell())
    ok = dict(fail_all, lap_time_s=[None] * N)
    rp.validate_cell(mutated(**ok), Cell())
    with pytest.raises(rp.ReportError, match="completed a lap but carries no lap time"):
        rp.validate_cell(mutated(lap_time_s=[None] * N), Cell())


def test_failure_counts_are_rebuilt_not_trusted(rp):
    """A tally and its outcomes can be wrong the same way; only one of them is the measurement."""
    r = mutated(outcomes=[{"success": False, "reason": "timeout"}] * N,
                lap_time_s=[None] * N,
                tally={"successes": 0, "denominator": N, "failures": {"contact": 999}})
    with pytest.raises(rp.ReportError, match="do not match the outcomes"):
        rp.validate_cell(r, Cell())


# ---- D2: cardinality against the declared cell
def test_per_cell_denominator_checked_against_declared_envs(rp):
    with pytest.raises(rp.ReportError, match="!= declared"):
        rp.validate_cell(control(), Cell(envs=16))


def test_recorded_n_envs_may_not_disagree(rp):
    r = control()
    r["n_envs"] = 16
    with pytest.raises(rp.ReportError, match="recorded n_envs"):
        rp.validate_cell(r, Cell())


# ---- D3: effective config and runtime identity
def test_effective_mu_must_match_the_declared_cell(rp):
    r = control()
    r["result"]["effective"] = {"true_mu": 1.15379, "plant_mu": 1.15379}
    with pytest.raises(rp.ReportError, match="effective"):
        rp.validate_cell(r, Cell())


def test_effective_block_is_required(rp):
    r = control()
    del r["result"]["effective"]
    with pytest.raises(rp.ReportError, match="no effective configuration"):
        rp.validate_cell(r, Cell())


def test_runtime_may_not_contradict_the_controller_arm(rp):
    r = control()
    r["runtime"] = "estimated"          # arm stays legacy
    with pytest.raises(rp.ReportError, match="runtime .* contradicts"):
        rp.validate_cell(r, Cell())


def test_cell_id_must_match_its_own_metadata(rp):
    r = control()
    r["cell_id"] = "S:elsewhere:0.94401:4401"
    with pytest.raises(rp.ReportError, match="does not match"):
        rp.validate_row_identity(r, {})


# ---- D1 rendering: absent optionals are N/A, never 0
def test_absent_optional_metrics_render_na_not_zero(rp):
    r = control()
    for k in ("spin_events", "large_slip_seconds", "max_abs_yaw_rate"):
        del r["result"][k]
    rp.validate_cell(r, Cell())
    vals = rp.aggregate([r])["stability"][0]["values"]
    for k in ("spins ↓", "large-slip s/km ↓", "max yaw rate rad/s (diagnostic)"):
        assert vals[k]["value"] is None, f"{k} rendered {vals[k]} for an unmeasured quantity"
        assert vals[k]["reason"]


def test_distance_has_no_net_progress_fallback(rp):
    """|net progress| would report a car oscillating in place as having covered no ground."""
    r = control()
    del r["result"]["distance_m"]
    with pytest.raises(rp.ReportError, match="distance_m"):
        rp.validate_cell(r, Cell())


def test_aggregate_refuses_an_unvalidated_row(rp):
    with pytest.raises(rp.ReportError, match="aggregated before validation"):
        rp.aggregate([control()])


# ---- D7: a real RMS, pooled over samples
def test_centreline_rms_is_sample_weighted(rp):
    """Two trials, 1 sample at 0 m and 1 at 2 m -> RMS sqrt(2), not the mean of per-trial RMSs."""
    r = control(2)
    r["result"].update({"cross_track_rms_m": [0.0, 2.0], "cross_track_samples": [1, 1]})
    rp.validate_cell(r, Cell(envs=2))
    for block in rp.aggregate([r]).values():
        for row in block:
            v = row["values"].get("centreline offset RMS m")
            if v and v.get("value") is not None:
                assert v["value"] == pytest.approx(2.0 ** 0.5)
                return
    pytest.skip("centreline block not emitted for this row shape")


# ---- D5/D6: roster
def test_duplicate_system_id_refused(ro, tmp_path):
    import json
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"systems": [
        {"system_id": "sys", "path": "a.pt", "checkpoint_sha256": "a" * 64,
         "controller_arm": "legacy"},
        {"system_id": "sys", "path": "b.pt", "checkpoint_sha256": "b" * 64,
         "controller_arm": "legacy"}]}))
    with pytest.raises(ValueError, match="duplicate system_id"):
        ro.load(str(p))


def test_probe_reads_the_production_controller_contract(ro, tmp_path):
    """extra.experiment.controller.arm with the legacy default, matching
    model.controller_arm_of."""
    torch = pytest.importorskip("torch")
    p = tmp_path / "ck.pt"
    torch.save({"state_dict": {"w": torch.zeros(2)},
                "extra": {"experiment": {"controller": {"arm": "estimated"}}}}, str(p))
    assert ro.probe(str(p))["recorded_arm"] == "estimated"
    q = tmp_path / "legacy.pt"
    torch.save({"state_dict": {"w": torch.zeros(2)}}, str(q))
    assert ro.probe(str(q))["recorded_arm"] == "legacy"


# ---- effective config survives a float32 round-trip, but not a real difference
def _f32(x):
    """The value the plant actually gives back: float64 -> float32 -> float64."""
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]


@pytest.mark.parametrize("mu", [0.73423, 0.94401, 1.15379])
def test_effective_mu_accepts_the_float32_round_trip(rp, mu):
    """The plant stores mu as float32, so a declared level never returns bit-identical.

    An exact comparison refused every real cell at every friction level while passing synthetic
    rows whose `effective` values were float64 literals -- it failed only on the measurements it
    exists to protect. These fixtures use the round-tripped value on purpose.
    """
    r = control()
    r["result"]["effective"] = {"true_mu": _f32(mu), "plant_mu": _f32(mu)}
    assert _f32(mu) != mu, "fixture must not be a float64 literal or it proves nothing"
    rp.validate_cell(r, Cell(mu=mu))


@pytest.mark.parametrize("wrong", [0.94401, 1.15379])
def test_effective_mu_still_refuses_a_real_difference(rp, wrong):
    """Tolerance absorbs 2e-08, not a different friction level."""
    r = control()
    r["result"]["effective"] = {"true_mu": _f32(wrong), "plant_mu": _f32(wrong)}
    with pytest.raises(rp.ReportError, match="effective"):
        rp.validate_cell(r, Cell(mu=0.73423))


def test_tolerance_is_tighter_than_any_meaningful_difference(rp):
    """1e-6 relative: a 0.1% error in mu is still refused."""
    r = control()
    r["result"]["effective"] = {"true_mu": 0.94401 * 1.001, "plant_mu": 0.94401 * 1.001}
    with pytest.raises(rp.ReportError, match="effective"):
        rp.validate_cell(r, Cell(mu=0.94401))


# ---- task success is not lap completion: A/O rows carry no lap times
def ao_row(suite_id, n=4):
    """A row shaped as `runner.py` writes it for the A and O suites.

    runner.py:196 sets `completed[i] = success and self.suite == "S"`, so on A and O `completed` is
    always False and `lap_time_s` is all None even when the task succeeded. Task duration lives in
    `elapsed_s`, which the recorder accumulates for every suite.
    """
    r = control(n)
    r["suite"] = suite_id
    r["cell_id"] = f"{suite_id}:m:0.94401:4401"
    r["result"].update({
        "n": n,
        "outcomes": [{"success": True, "reason": None}] * n,   # task success
        "lap_time_s": [None] * n,                              # no laps on A/O, by construction
        "elapsed_s": [7.5] * n,                                # the duration that IS recorded
        "progress_m": [30.0] * n, "distance_m": [30.0] * n,
        "route_progress_fraction": [0.6] * n,
        "cross_track_rms_m": [0.2] * n, "cross_track_samples": [50] * n,
        "tally": {"successes": n, "denominator": n, "failures": {}}})
    return r


@pytest.mark.parametrize("suite_id", ["A", "O"])
def test_ao_success_without_lap_time_is_accepted(rp, suite_id):
    """The real producer's A/O rows must validate; demanding a lap time refused every one."""
    r = ao_row(suite_id)
    rp.validate_cell(r, Cell(envs=4, suite=suite_id))
    assert r["result"]["_derived"]["successes"] == 4


@pytest.mark.parametrize("suite_id", ["A", "O"])
def test_ao_row_validates_then_aggregates(rp, suite_id):
    """End to end on the real shape: producer row -> validate -> aggregate, no fabricated lap."""
    r = ao_row(suite_id)
    rp.validate_cell(r, Cell(envs=4, suite=suite_id))
    agg = rp.aggregate([r])
    assert agg, "aggregation produced nothing for a valid A/O row"


@pytest.mark.parametrize("suite_id", ["A", "O"])
def test_ao_row_with_a_lap_time_is_refused(rp, suite_id):
    """A lap number on a suite that runs no laps is fabricated, not merely surprising."""
    r = ao_row(suite_id)
    r["result"]["lap_time_s"] = [9.9] * 4
    with pytest.raises(rp.ReportError, match="has no laps"):
        rp.validate_cell(r, Cell(envs=4, suite=suite_id))


def test_s_success_missing_lap_time_is_still_refused(rp):
    """The S rule is unchanged: a completed lap must carry its time."""
    r = control()
    r["result"]["lap_time_s"] = [None] * N
    with pytest.raises(rp.ReportError, match="completed a lap but carries no lap time"):
        rp.validate_cell(r, Cell())


# ---- the Suite object is required on the authoritative path
def test_validate_results_refuses_without_the_suite(rp):
    """Without it, speed_cap/budget_laps/sensor_noise/backend are silently never compared."""
    with pytest.raises(rp.ReportError, match="needs the Suite object"):
        rp.validate_results([control()], suite_freeze="f" * 64, expected_systems={"sys"},
                            expected_trials={"S": 8}, expected_cells=[Cell()], roster=None)


# ---- presence, not only equality: a field that is merely absent used to pass every comparison
def _proto_row():
    """The control row; it already carries the declared protocol the producer writes."""
    return control()


def test_declared_protocol_control_passes(rp):
    rp.validate_cell(_proto_row(), Cell(), suite=Suite())


@pytest.mark.parametrize("field", ["speed_cap", "budget_laps", "sensor_noise", "backend"])
def test_missing_top_level_protocol_field_refused(rp, field):
    """Absent is not agreement. Each of these was accepted before presence was required."""
    r = _proto_row()
    r.pop(field)
    with pytest.raises(rp.ReportError, match="records no"):
        rp.validate_cell(r, Cell(), suite=Suite())


@pytest.mark.parametrize("bad", [
    {"speed_cap": 2}, {"budget_laps": 100}, {"sensor_noise": False}, {"backend": "other"}])
def test_effective_protocol_must_match_the_declared_suite(rp, bad):
    """These were accepted because validate_results called validate_cell without the Suite."""
    r = _proto_row()
    r["result"]["effective"].update(bad)
    with pytest.raises(rp.ReportError, match="effective"):
        rp.validate_cell(r, Cell(), suite=Suite())


@pytest.mark.parametrize("drop", ["true_mu", "plant_mu"])
def test_effective_present_but_missing_mu_refused(rp, drop):
    """A non-empty `effective` without the friction it ran at is not a checkable record."""
    r = _proto_row()
    r["result"]["effective"] = {k: v for k, v in r["result"]["effective"].items() if k != drop}
    r["result"]["effective"]["speed_cap"] = 9.0          # non-empty, so truthiness will not catch it
    with pytest.raises(rp.ReportError, match="records no"):
        rp.validate_cell(r, Cell(), suite=Suite())


# ---- start fingerprint: required reproducibility metadata, not an optional extra
def test_missing_start_fingerprint_is_refused(rp):
    """Without it, "these systems ran the same cell" is an assumption nothing can check."""
    r = control()
    del r["result"]["start_fingerprint"]
    with pytest.raises(rp.ReportError, match="empty or missing|no start_fingerprint"):
        rp.validate_cell(r, Cell(), suite=Suite())


@pytest.mark.parametrize("part", ["physical", "actor_input", "calibration"])
def test_missing_digest_part_is_refused(rp, part):
    r = control()
    r["result"]["start_fingerprint"][f"{part}_sha256"] = None
    with pytest.raises(rp.ReportError, match=f"{part}_sha256 is not a 64-character hex"):
        rp.validate_cell(r, Cell(), suite=Suite())


@pytest.mark.parametrize("part", ["physical", "actor_input"])
def test_zero_tensor_digest_is_refused(rp, part):
    """A sha256 over no tensors is stable and reads as agreement for every system.

    That is the rename failure mode: an attribute disappears, every digest becomes the hash of an
    empty sequence, and the whole matrix suddenly "pairs".
    """
    r = control()
    r["result"]["start_fingerprint"][f"{part}_tensors"] = 0
    with pytest.raises(rp.ReportError, match="digest over none|positive integer"):
        rp.validate_cell(r, Cell(), suite=Suite())


def test_foreign_schema_version_is_refused(rp):
    """Comparing two different definitions is not a comparison."""
    r = control()
    r["result"]["start_fingerprint"]["schema_version"] = 99
    with pytest.raises(rp.ReportError, match="expected the integer"):
        rp.validate_cell(r, Cell(), suite=Suite())


def _two_systems(fp_b=None):
    a, b = control(), control()
    b["system_id"] = "sys2"
    if fp_b is not None:
        b["result"]["start_fingerprint"] = fp_b
    for r in (a, b):
        r.update({"checkpoint_sha256": "a" * 64, "suite_freeze_sha256": "f" * 64,
                  "suite_version": "v1", "source_digest": {"sim.py": "abc"}})
    return [a, b]


def _validate_pair(rp, rows):
    return rp.validate_results(rows, suite_freeze="f" * 64,
                               expected_systems={"sys", "sys2"},
                               expected_trials={"S": 8}, expected_cells=[Cell()],
                               roster=None, suite=Suite())


def test_paired_starts_accepted_across_systems(rp):
    _validate_pair(rp, _two_systems())


@pytest.mark.parametrize("layer", ["physical", "actor_input", "calibration"])
def test_unpaired_start_is_refused_and_names_the_layer(rp, layer):
    """"actor_input differs" is a five-minute diagnosis; "not paired" is a re-run."""
    bad = fingerprint(**{f"{layer}_sha256": "e" * 64})
    with pytest.raises(rp.ReportError, match=layer):
        _validate_pair(rp, _two_systems(bad))


def test_mixed_obs_spec_is_refused_not_pooled(rp):
    """Two systems can agree on every observation tensor and still expect different stacking."""
    bad = fingerprint(obs_spec_sha256="9" * 64)
    with pytest.raises(rp.ReportError):
        _validate_pair(rp, _two_systems(bad))


# ---- a structurally empty or malformed fingerprint must never reach compare()
def test_empty_fingerprints_do_not_pair(rp):
    """Two absent records must not agree with each other.

    `compare({}, {})` once returned paired=True, so an unfingerprinted matrix read as perfectly
    paired. It now raises at the producer, and the validator refuses an empty record before any
    comparison runs -- both ends, because either alone leaves the other reachable.
    """
    from f1sim.learn.benchmark import fingerprint as _fp
    with pytest.raises(ValueError, match="not evidence"):
        _fp.compare({}, {})
    r = control()
    r["result"]["start_fingerprint"] = {}
    with pytest.raises(rp.ReportError, match="empty or missing|no start_fingerprint"):
        rp.validate_cell(r, Cell(), suite=Suite())


@pytest.mark.parametrize("bad", ["", "x" * 64, "a" * 63, "A" * 64, None, 12345])
def test_digest_must_be_64_lowercase_hex(rp, bad):
    r = control()
    r["result"]["start_fingerprint"]["physical_sha256"] = bad
    with pytest.raises(rp.ReportError, match="not a 64-character hex"):
        rp.validate_cell(r, Cell(), suite=Suite())


def test_calibration_tensor_count_must_be_nonzero(rp):
    """Calibration counts too: it is the command path, and an empty one pairs just as silently."""
    r = control()
    r["result"]["start_fingerprint"]["calibration_tensors"] = 0
    with pytest.raises(rp.ReportError, match="digest over none|positive integer"):
        rp.validate_cell(r, Cell(), suite=Suite())


def test_obs_spec_hash_is_required(rp):
    r = control()
    r["result"]["start_fingerprint"]["obs_spec_sha256"] = None
    with pytest.raises(rp.ReportError, match="obs_spec_sha256|obs_spec"):
        rp.validate_cell(r, Cell(), suite=Suite())


@pytest.mark.parametrize("key,bad", [
    ("sim_t", None), ("sim_imu_phase", "1"), ("n_envs_total", None), ("race_size", 0),
    ("n_envs_total", 0)])
def test_scalar_metadata_required_and_sane(rp, key, bad):
    r = control()
    r["result"]["start_fingerprint"][key] = bad
    with pytest.raises(rp.ReportError, match="."):
        rp.validate_cell(r, Cell(), suite=Suite())


# ---- the fingerprint's population must BE the declared cell's, not merely self-consistent
def test_fingerprint_population_bound_to_cell_and_suite(rp):
    """An 8-learner solo cell whose fingerprint claims 16 envs in races of 4 is describing a
    different measurement, and two such rows can each be internally consistent while having run
    different-sized races."""
    r = control()
    r["result"]["start_fingerprint"].update(n_envs_total=16, race_size=4)
    with pytest.raises(rp.ReportError, match="race_size|n_envs_total"):
        rp.validate_cell(r, Cell(), suite=Suite())


def test_fingerprint_env_total_must_equal_learners_times_race(rp):
    r = control()
    r["result"]["start_fingerprint"]["n_envs_total"] = 9      # 8 learners x race_size 1
    with pytest.raises(rp.ReportError, match="n_envs_total"):
        rp.validate_cell(r, Cell(), suite=Suite())


@pytest.mark.parametrize("part", ["physical", "actor_input", "calibration"])
def test_tensor_count_of_True_is_refused(rp, part):
    """`isinstance(True, int)` is True and `True >= 1`, so a bool sails through an int check."""
    r = control()
    r["result"]["start_fingerprint"][f"{part}_tensors"] = True
    with pytest.raises(rp.ReportError, match="digest over none|positive integer"):
        rp.validate_cell(r, Cell(), suite=Suite())


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_non_finite_sim_t_is_refused(rp, bad):
    """Infinity compares equal to Infinity, so two such records would pair on the clock."""
    r = control()
    r["result"]["start_fingerprint"]["sim_t"] = bad
    with pytest.raises(rp.ReportError, match="finite clock reading"):
        rp.validate_cell(r, Cell(), suite=Suite())


@pytest.mark.parametrize("key", ["sim_imu_phase", "n_envs_total", "race_size"])
def test_bool_scalars_are_refused(rp, key):
    r = control()
    r["result"]["start_fingerprint"][key] = True
    with pytest.raises(rp.ReportError, match="."):
        rp.validate_cell(r, Cell(), suite=Suite())


# ---- race_size binds PER CELL, not per suite
class RaceSuite(Suite):
    """The real suite's race_size is 2 -- but that is the OVERTAKING size only."""

    race_size = 2


def test_solo_cell_under_a_race_suite_is_accepted(rp):
    """S and A cells are solo and legitimately fingerprint race_size 1.

    Comparing every cell against `suite.race_size` refused every solo cell -- 208 of 272 trials per
    system -- and passed on every hand-built fixture, because a fixture picks its own race_size on
    both sides. Only a row from a real run could expose it.
    """
    rp.validate_cell(control(), Cell(suite="S"), suite=RaceSuite())
    rp.validate_cell(ao_row("A"), Cell(envs=4, suite="A"), suite=RaceSuite())


def test_o_cell_expects_learners_times_race_size(rp):
    """An O cell of 8 learners at race_size 2 must fingerprint 16 envs, and 8 is refused."""
    r = ao_row("O")
    r["result"]["start_fingerprint"].update(race_size=2, n_envs_total=4 * 2)
    rp.validate_cell(r, Cell(envs=4, suite="O"), suite=RaceSuite())

    bad = ao_row("O")
    bad["result"]["start_fingerprint"].update(race_size=2, n_envs_total=4)
    with pytest.raises(rp.ReportError, match="n_envs_total"):
        rp.validate_cell(bad, Cell(envs=4, suite="O"), suite=RaceSuite())


def test_solo_cell_claiming_a_race_is_still_refused(rp):
    r = control()
    r["result"]["start_fingerprint"].update(race_size=2, n_envs_total=16)
    with pytest.raises(rp.ReportError, match="race_size"):
        rp.validate_cell(r, Cell(suite="S"), suite=RaceSuite())


# ---- one validator: the same malformed fingerprint must be refused through BOTH public paths
@pytest.mark.parametrize("key,bad,label", [
    ("schema_version", 1.9, "fractional schema (int() truncated it to 1)"),
    ("schema_version", True, "bool schema (bool is an int subclass)"),
    ("obs_spec_sha256", "not-hex", "obs_spec that is not hex"),
    ("obs_spec_sha256", "d" * 63, "obs_spec one char short"),
    ("obs_spec_sha256", None, "obs_spec absent"),
])
def test_malformed_fingerprint_refused_through_both_entry_points(rp, key, bad, label):
    """report.validate_cell and fingerprint.validate_fingerprint must agree exactly.

    They were two copies of one contract, and the copy in report.py had drifted: it cast
    schema_version with int(), so 1.9 and True passed there while the shared helper refused them.
    Asserting both entry points is what keeps a single rule single.
    """
    from f1sim.learn.benchmark import fingerprint as _fp
    r = control()
    r["result"]["start_fingerprint"][key] = bad

    with pytest.raises(rp.ReportError):
        rp.validate_cell(r, Cell(), suite=Suite())
    with pytest.raises(ValueError):
        _fp.validate_fingerprint(r["result"]["start_fingerprint"], expected_cell=Cell(),
                                 suite=Suite())


def test_report_converts_helper_errors_to_report_errors(rp):
    """A ValueError escaping the report API would bypass every `except ReportError` handler."""
    r = control()
    r["result"]["start_fingerprint"]["schema_version"] = 1.9
    with pytest.raises(rp.ReportError):
        rp.validate_cell(r, Cell(), suite=Suite())
