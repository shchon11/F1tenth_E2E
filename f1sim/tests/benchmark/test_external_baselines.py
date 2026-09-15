"""The `external` roster kind: what it refuses, and that a cell of it actually drives.

Two things worth pinning:

* an external baseline has **no controller arm**. It publishes (steer, speed), so there is no plan
  for a tracker to follow and no solver for an arm to wrap. A roster entry that names one is a
  category error and is refused at verification time, not at GPU time.
* its driver options are part of its **identity**. End2Race with the quarter of its scan this car
  cannot see filled at 30 m, and the same weights filled at 0 m, are two evaluated systems -- the
  same relationship the same checkpoint under two controller arms has.

The last test builds real simulators on CPU (`slow`): two cells per model, which is CONTRACT.md's
smoke of each baseline on the benchmark path.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

MODELS = os.path.join(os.path.expanduser("~"), "Documents", "Codex", "2026-09-10", "new-chat",
                      "work", "baselines", "models")
TLN_ONNX = os.path.join(MODELS, "tinylidarnet_L_1081.onnx")
E2R_WEIGHTS = os.path.join(os.path.expanduser("~"), "F1tenth", "F1tenth_E2E", "external",
                           "baselines", "End2Race", "pretrained", "end2race.pth")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def need(path, what):
    if not os.path.exists(path):
        pytest.skip(f"{what} is not available at {path}")
    return path


def roster_file(tmp_path, systems):
    p = tmp_path / "roster.json"
    p.write_text(json.dumps({"systems": systems}))
    return str(p)


def external_entry(**kw):
    need(TLN_ONNX, "the converted TinyLidarNet ONNX")
    e = {"system_id": "tinylidarnet_L", "kind": "tinylidarnet", "weights": TLN_ONNX,
         "checkpoint_sha256": sha(TLN_ONNX), "controller_arm": "none"}
    e.update(kw)
    return e


# --------------------------------------------------------------------------- roster
def test_roster_accepts_an_external_entry_with_arm_none(tmp_path, bench):
    from f1sim.learn.benchmark import roster
    entries = roster.load(roster_file(tmp_path, [external_entry()]))
    info = roster.verify_all(entries)
    assert info["n_systems"] == 1
    assert entries[0].kind == "tinylidarnet"
    assert entries[0].path == TLN_ONNX          # `weights` is mapped onto `path`, so pins apply


@pytest.mark.parametrize("arm", ["legacy", "fixed_low", "fixed_low+clearance", "estimated"])
def test_roster_refuses_an_external_entry_with_a_controller_arm(tmp_path, arm, bench):
    from f1sim.learn.benchmark import roster
    entries = roster.load(roster_file(tmp_path, [external_entry(controller_arm=arm)]))
    with pytest.raises(ValueError, match="external baseline declares arm"):
        roster.verify_all(entries)


def test_roster_refuses_arm_none_on_an_ordinary_checkpoint(tmp_path, bench):
    from f1sim.learn.benchmark import roster
    e = external_entry()
    e.pop("kind")
    entries = roster.load(roster_file(tmp_path, [e]))
    with pytest.raises(ValueError, match="absence of a plan tracker"):
        roster.verify_all(entries)


def test_roster_refuses_an_estimator_on_an_external_entry(tmp_path, bench):
    from f1sim.learn.benchmark import roster
    entries = roster.load(roster_file(tmp_path, [
        external_entry(estimator_path=TLN_ONNX, estimator_sha256=sha(TLN_ONNX))]))
    with pytest.raises(ValueError, match="no controller arm"):
        roster.verify_all(entries)


def test_roster_refuses_an_unknown_kind_and_an_unknown_field(tmp_path, bench):
    from f1sim.learn.benchmark import roster
    entries = roster.load(roster_file(tmp_path, [external_entry(kind="not_a_baseline")]))
    with pytest.raises(ValueError, match="unknown external kind"):
        roster.verify_all(entries)
    with pytest.raises(ValueError, match="unknown roster field"):
        roster.load(roster_file(tmp_path, [dict(external_entry(), speed_map="sim")]))


def test_driver_options_are_part_of_the_row_identity(tmp_path, bench):
    """Same weights, two fills -> two systems. Same weights, same options -> a duplicate."""
    from f1sim.learn.benchmark import roster
    need(E2R_WEIGHTS, "End2Race")
    base = {"kind": "end2race", "weights": E2R_WEIGHTS, "checkpoint_sha256": sha(E2R_WEIGHTS),
            "controller_arm": "none"}
    two = roster.load(roster_file(tmp_path, [
        dict(base, system_id="e2r_fill30", options={"scan_fill": 30.0}),
        dict(base, system_id="e2r_fill0", options={"scan_fill": 0.0})]))
    assert two[0].key() != two[1].key()
    roster.verify_all(two)
    with pytest.raises(ValueError, match="duplicate system identity"):
        roster.load(roster_file(tmp_path, [
            dict(base, system_id="a", options={"scan_fill": 30.0}),
            dict(base, system_id="b", options={"scan_fill": 30.0})]))


def test_check_arm_matches_record_does_not_try_to_open_an_onnx_as_a_checkpoint(tmp_path, bench):
    from f1sim.learn.benchmark import roster
    e = roster.load(roster_file(tmp_path, [external_entry()]))[0]
    info = roster.check_arm_matches_record(e)
    assert info["external"] is True and info["recorded_arm"] == "none"


def test_a_report_row_may_not_carry_an_estimator_pin_under_arm_none(bench):
    from f1sim.learn.benchmark import report
    row = {"system_id": "x", "checkpoint_sha256": "a" * 64, "controller_arm": "none",
           "suite_freeze_sha256": "b" * 64, "suite_version": "v2.1", "map_id": "m", "mu": 0.9,
           "seed": 1, "n_envs": 8, "source_digest": "c" * 16}
    report.validate_row(row)                                  # accepted: `none` is not splittable
    with pytest.raises(report.ReportError, match="no controller"):
        report.validate_row(dict(row, estimator_sha256="d" * 64))


# --------------------------------------------------------------------------- adapter
def test_the_adapter_refuses_an_external_entry_that_names_an_arm(bench):
    from f1sim.learn.benchmark import model_adapter as ma
    with pytest.raises(ma.AdapterError, match="only honest declaration"):
        ma.load_external({"kind": "tinylidarnet", "weights": need(TLN_ONNX, "the ONNX"),
                          "arm": "fixed_low"})


def test_external_spec_is_this_cars_scanner_not_the_models_own(bench):
    from f1sim.learn import baselines
    from f1sim.learn.benchmark import model_adapter as ma
    from f1sim.params import Config
    driver, extra = ma.load_external({"kind": "end2race", "weights": need(E2R_WEIGHTS, "End2Race"),
                                      "arm": "none"})
    lid = Config().lidar
    assert extra["spec"]["n_beams"] == int(lid.n_beams)
    assert extra["spec"]["range_max"] == pytest.approx(float(lid.range_max))
    assert driver.scan.n_beams == 1440 and driver.source["n_beams"] == int(lid.n_beams)
    assert driver.source["unseen_fraction"] == pytest.approx(0.25, abs=0.005)
    assert isinstance(driver, baselines.BaselineDriver)


def test_policy_for_refuses_a_driver_and_points_at_external_policy(bench):
    from f1sim.learn.benchmark import model_adapter as ma
    driver, _extra = ma.load_external({"kind": "tinylidarnet", "weights": need(TLN_ONNX, "the ONNX"),
                                       "arm": "none"})
    with pytest.raises(ma.AdapterError, match="external_policy"):
        ma.policy_for(driver)


# --------------------------------------------------------------------------- the real thing
@pytest.mark.slow
@pytest.mark.parametrize("kind,weights_of", [("tinylidarnet", lambda: TLN_ONNX),
                                             ("end2race", lambda: E2R_WEIGHTS)])
def test_two_cells_of_each_baseline_drive_on_cpu(kind, weights_of, bench):
    """CONTRACT.md's 2-cell CPU smoke. A real env, a real reset, real steps, and the checks that
    the cell is what it says it is: no controller, direct actions, the declared friction."""
    import torch

    from f1sim.learn.benchmark import model_adapter as ma
    from f1sim.learn.benchmark.runner import run_cell
    from f1sim.params import VehicleParams

    weights = need(weights_of(), kind)
    entry = {"kind": kind, "weights": weights, "arm": "none", "system_id": f"{kind}_smoke"}
    driver, extra = ma.load_actor(entry, "cpu")
    policy = ma.external_policy(driver, extra["spec"], float(VehicleParams().s_max))
    suite = {"speed_cap": 9.0, "sensor_noise": True, "budget_laps": 0.25, "envs": 2}
    for mu, seed in ((0.73423, 4401), (1.15379, 4402)):
        cell = {"map": "gen:control:9100", "true_mu": mu, "seed": seed, "envs": 2, "race_size": 1}
        prepared = ma.prepare_cell(entry, extra, cell, suite, torch.device("cpu"))
        try:
            p = prepared.protocol
            assert p["arm"] == "none" and p["action_mode"] == "direct"
            assert prepared.controller is None
            assert prepared.env.tracker is None, "an external baseline must not build a tracker"
            assert p["plant_mu"] == pytest.approx(mu, abs=1e-6)   # float32 plant
            assert p["external"]["kind"] == kind
            res = run_cell(prepared.env, policy, suite="S",
                           n_steps=int(prepared.env.ecfg.max_steps), frozen=True,
                           controller=prepared, seed=seed, obs_spec=p["spec"])
        finally:
            prepared.close()
        t = res["tally"]
        assert t["denominator"] == 2
        assert len(res["outcomes"]) == 2
        assert sum(res["distance_m"]) > 0.0, "the car never moved"


@pytest.mark.slow
def test_evaluate_drives_a_published_baseline_through_the_traffic_proxy(bench):
    """`learn.evaluate --external-kind` is how the fair comparison gets "the same evaluation".

    The traffic proxy (worker 17's D2/D3 protocol) is `evaluate`'s rolling path with a race and the
    event set; running a baseline through it must reuse that code rather than a second copy, so what
    is checked here is that the same function accepts the driver, runs in `direct` mode with no arm,
    and reports the same `TrafficMeter` block every other arm reports.
    """
    from f1sim.learn.evaluate import evaluate
    from f1sim.params import Config

    cfg = Config()
    cfg.sim.compile = False
    res = evaluate("", ["gen:control:9100"], envs=6, steps=60, speed_cap=9.0, device="cpu",
                   seed=4242, cfg=cfg, protocol="rolling", race_size=3, opponent="teacher",
                   opp_speed_range=(0.6, 1.15),
                   opp_events=("brake", "stop", "shift", "defend", "yield", "line", "oblivious"),
                   opp_event_rate=1.0, budget_laps=None,
                   external={"kind": "tinylidarnet", "weights": need(TLN_ONNX, "the ONNX")})
    md = res["metadata"]
    assert md["action_mode"] == "direct"
    assert md["controller_arm"] == "legacy"           # i.e. nothing installed
    assert md["external"]["kind"] == "tinylidarnet"
    assert md["external"]["backend"]["backend"] == "onnxruntime"
    t = res["traffic"]
    assert t["learners"] == 2 and t["opponents_per_learner"] == 2
    assert t["learner_minutes"] > 0 and t["ego_progress_m"] > 0
    with pytest.raises(ValueError, match="no plan tracker"):
        evaluate("", ["gen:control:9100"], envs=2, steps=5, speed_cap=9.0, device="cpu", cfg=cfg,
                 controller="fixed_low",
                 external={"kind": "tinylidarnet", "weights": TLN_ONNX})
