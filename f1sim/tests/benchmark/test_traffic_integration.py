"""The T family through the real adapter and the real loop, on a procedural map outside every suite.

Not mocked and not a benchmark score: a synthetic checkpoint, a map no suite declares, a handful of
steps. What is under test is that the traffic path actually reaches the simulator -- the scenario's
opponent count and speed range, the scripted events, the per-opponent detectors and the event-in-
window measurement -- rather than being a correct implementation nothing calls.
"""
from __future__ import annotations
import importlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("f1sim.gym_env")

from .test_model_adapter import make_checkpoint, TEST_MAP, MU_LOW          # noqa: E402


@pytest.fixture
def ma(bench):
    return importlib.import_module("f1sim.learn.benchmark.model_adapter")


@pytest.fixture
def su(bench):
    return importlib.import_module("f1sim.learn.benchmark.suite")


@pytest.fixture
def rn(bench):
    return importlib.import_module("f1sim.learn.benchmark.runner")


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    p, spec = make_checkpoint(tmp_path_factory.mktemp("traffic"), "legacy_traffic")
    return {"path": p, "arm": "legacy"}, {"spec": spec, "cap": 9.0}


def prepare(ma, su, legacy, variant, learners=2, budget_laps=1.0):
    entry, extra = legacy
    s = su.of("v2.1")
    s.envs, s.budget_laps = learners, budget_laps
    cell = su.Cell("T", TEST_MAP, MU_LOW, 4401, learners, variant=variant,
                   race_size=int(s.traffic_scenario(variant)["race_size"]))
    sd = s.adapter_suite(cell)
    prep = ma.prepare_cell(entry, extra,
                           {"map": TEST_MAP, "true_mu": MU_LOW, "seed": 4401,
                            "envs": learners, "race_size": sd["race_size"]}, sd, "cpu")
    return s, cell, prep


def drive(prep, rn, suite_obj, cell, steps=30):
    env = prep.env
    return rn.run_cell(env, lambda obs: torch.zeros(env.B, env.act_dim), suite="T",
                       n_steps=steps, hold_steps=40, frozen=False, controller=prep,
                       contention_range_m=suite_obj.traffic["contention_range_m"],
                       attack_range_m=suite_obj.traffic["attack_range_m"])


def test_the_two_opponent_scenario_really_builds_three_cars_per_race(ma, su, legacy):
    s, cell, prep = prepare(ma, su, legacy, "pair", learners=2)
    try:
        env = prep.env
        assert int(env.M) == 3 and int(env.B) == 6
        assert int(env.on_policy.sum()) == 2
        assert env.sim.other_idx.shape[1] == 2
        assert prep.protocol["race_size"] == 3
    finally:
        prep.close()


def test_the_scenario_speed_range_reaches_the_env(ma, su, legacy):
    for variant, want in (("slow", (0.5, 0.7)), ("pace", (0.8, 0.95))):
        s, cell, prep = prepare(ma, su, legacy, variant)
        try:
            assert tuple(prep.env.ecfg.opp_speed_range) == want
            assert prep.protocol["opp_speed_range"] == list(want)
        finally:
            prep.close()


def test_the_event_scenario_arms_the_event_machine_and_nothing_else_does(ma, su, legacy):
    s, cell, prep = prepare(ma, su, legacy, "event")
    try:
        env = prep.env
        assert env.events is not None and env.events.enabled
        assert sorted(env.ecfg.opp_events) == ["brake", "shift", "stop"]
        assert env.ecfg.opp_event_rate > 0
        assert sorted(prep.protocol["opp_events"]) == ["brake", "shift", "stop"]
    finally:
        prep.close()
    s, cell, prep = prepare(ma, su, legacy, "slow")
    try:
        assert prep.env.events is not None and not prep.env.events.enabled
        assert prep.protocol["opp_events"] == []
    finally:
        prep.close()


def test_events_declared_without_a_car_to_script_are_refused(ma, su, legacy):
    """Silently running the unflagged cell under the event cell's name is the failure here."""
    entry, extra = legacy
    s = su.of("v2.1")
    sd = dict(s.adapter_suite(), opp_events=["brake"], opp_event_rate=3.0, race_size=1)
    with pytest.raises(ma.AdapterError, match="race_size"):
        ma.prepare_cell(entry, extra,
                        {"map": TEST_MAP, "true_mu": MU_LOW, "seed": 4401,
                         "envs": 2, "race_size": 1}, sd, "cpu")


def test_events_at_a_zero_rate_are_refused(ma, su, legacy):
    entry, extra = legacy
    s = su.of("v2.1")
    sd = dict(s.adapter_suite(), opp_events=["brake"], opp_event_rate=0.0, race_size=2)
    with pytest.raises(ma.AdapterError, match="rate"):
        ma.prepare_cell(entry, extra,
                        {"map": TEST_MAP, "true_mu": MU_LOW, "seed": 4401,
                         "envs": 2, "race_size": 2}, sd, "cpu")


def test_a_traffic_cell_produces_a_complete_row(ma, su, legacy, rn):
    from f1sim.learn.benchmark import report
    s, cell, prep = prepare(ma, su, legacy, "slow")
    try:
        res = drive(prep, rn, s, cell, steps=25)
    finally:
        prep.close()
    assert res["n"] == 2 and res["n_opponents"] == 1
    for key in report.TRAFFIC_TRIAL_ARRAYS:
        assert len(res[key]) == 2, key
    # the continuous side is real, not a placeholder: two cars spawn 2.5-6 m apart
    assert res["contended"] >= 1
    assert sum(res["contention_s"]) > 0.0
    assert res["contention_range_m"] == s.traffic["contention_range_m"]
    assert res["attack_range_m"] == s.traffic["attack_range_m"]


def test_a_traffic_cell_with_two_opponents_tracks_both(ma, su, legacy, rn):
    s, cell, prep = prepare(ma, su, legacy, "pair")
    try:
        res = drive(prep, rn, s, cell, steps=25)
    finally:
        prep.close()
    assert res["n_opponents"] == 2
    assert res["n"] == 2


def test_the_event_cell_measures_whether_an_event_landed_in_the_window(ma, su, legacy, rn):
    """Reported from the machine's own state, per opponent, and only while the learner is close
    enough for it to be something to react to."""
    s, cell, prep = prepare(ma, su, legacy, "event", budget_laps=3.0)
    try:
        res = drive(prep, rn, s, cell, steps=400)
    finally:
        prep.close()
    assert "event_in_window_trials" in res
    assert sum(res["event_seen_s"]) > 0.0, "no event fired at all in 10 s of a 3/10 s schedule"
    assert res["event_in_window_trials"] <= res["n"]
    for a, b in zip(res["event_in_window_s"], res["event_seen_s"]):
        assert a <= b + 1e-9, "in-window event time cannot exceed total event time"


def test_a_t_cell_without_an_opponent_is_refused_rather_than_measured_empty(ma, su, legacy, rn):
    entry, extra = legacy
    s = su.of("v2.1")
    prep = ma.prepare_cell(entry, extra,
                           {"map": TEST_MAP, "true_mu": MU_LOW, "seed": 4401,
                            "envs": 2, "race_size": 1},
                           dict(s.adapter_suite(), race_size=1), "cpu")
    try:
        with pytest.raises(ValueError, match="no opponent"):
            rn.run_cell(prep.env, lambda obs: torch.zeros(prep.env.B, prep.env.act_dim),
                        suite="T", n_steps=5, frozen=False, controller=prep)
    finally:
        prep.close()


def test_the_traffic_expert_drives_the_scenario_it_was_given(ma, su, legacy, rn):
    """The feasibility reference, on the scoring path. Not a score: a synthetic checkpoint's env
    driven by a scripted teacher-based driver on a map no suite declares."""
    from f1sim.learn.benchmark.experts import TrafficExpert
    s, cell, prep = prepare(ma, su, legacy, "slow", budget_laps=1.0)
    try:
        env = prep.env
        driver = TrafficExpert(env)
        assert driver.limit.shape[:1] == env.sim.track.cl.shape[:1]
        res = rn.run_cell(env, driver, suite="T", n_steps=int(env.ecfg.max_steps),
                          hold_steps=40, frozen=False, controller=prep,
                          contention_range_m=12.0, attack_range_m=3.0)
    finally:
        prep.close()
    assert res["n"] == 2
    assert res["tally"]["denominator"] == 2


# ------------------------------------------------------------------ the whole CLI, end to end

def test_the_report_cli_renders_a_traffic_suite(tmp_path, su):
    """A frozen T suite, synthetic rows for every declared cell, and `report` through the CLI.

    The unit tests exercise `aggregate` and `validate_cell` directly; this is the path an actual
    run takes -- `suite.load`, the roster, `validate_results` with the whole strict argument set,
    then the renderer. It is where a T cell would be refused for a reason no unit test poses, and a
    144-cell GPU run finding that out afterwards is the expensive way to learn it.
    """
    import hashlib
    import json
    import os
    import subprocess
    import sys

    # A three-cell traffic suite: one map, one friction, one seed, all four scenarios.
    s = su.Suite(version="v2.1-test", solo_maps=(), obstacle_maps=(), race_maps=(),
                 solo_mus=(su.MU_LOW,), paired_mus=(su.MU_LOW,), seeds=(4401,), envs=4)
    s.traffic = {"contention_range_m": 12.0, "attack_range_m": 3.0, "maps": ["m1"],
                 "mus": [su.MU_LOW],
                 "scenarios": [dict(sc) for sc in su.V21_TRAFFIC_SCENARIOS]}
    s.placements = {"m1": {"placement": {}, "proofs": {}}}
    sp = tmp_path / "suite.json"
    freeze = s.save(str(sp))
    assert s.expected_trials() == {"S": 0, "A": 0, "O": 0, "T": 16}

    body = b"weights"
    ck = tmp_path / "w.pt"
    ck.write_bytes(body)
    sha = hashlib.sha256(body).hexdigest()
    rj = tmp_path / "roster.json"
    rj.write_text(json.dumps({"systems": [
        {"system_id": "sys", "path": str(ck), "checkpoint_sha256": sha,
         "controller_arm": "legacy"}]}))

    n = 4
    cells = []
    for cell in s.cells():
        sc = s.traffic_scenario(cell.variant)
        rs = int(sc["race_size"])
        cells.append({
            "system_id": "sys", "checkpoint_sha256": sha, "controller_arm": "legacy",
            "suite_freeze_sha256": freeze, "suite_version": s.version,
            "map_id": cell.map_id, "mu": cell.mu, "seed": cell.seed, "n_envs": n,
            "suite": "T", "variant": cell.variant, "runtime": "legacy",
            "cell_id": cell.cell_id(), "source_digest": {"sim.py": "abc"},
            "speed_cap": s.speed_cap, "budget_laps": s.budget_laps,
            "sensor_noise": s.sensor_noise, "backend": s.backend,
            "result": {
                "n": n,
                "tally": {"successes": 3, "denominator": n, "failures": {"contact": 1}},
                "outcomes": [{"success": True, "reason": None}] * 3 +
                            [{"success": False, "reason": "contact"}],
                "progress_m": [12.0] * n, "distance_m": [12.0] * n,
                "route_progress_fraction": [0.3] * n, "lap_time_s": [None] * n,
                "passes": [1, 0, 0, 0], "leads_lost": [0] * n, "opponent_respawns": [0] * n,
                "contention_s": [9.0] * n, "following_s": [5.0] * n, "attack_s": [2.0] * n,
                "defending_s": [1.0] * n, "opponent_progress_m": [10.0] * n,
                "pace_ratio": [{"value": 1.2, "reason": None}] * n,
                "closest_arc_gap_m": [0.8] * n,
                "event_in_window_s": [1.0] * n, "event_seen_s": [1.5] * n,
                "contended": n, "car_contacts": 1, "wall_collisions": 0,
                "event_in_window_trials": n if sc["opp_events"] else 0,
                "n_opponents": rs - 1, "contention_range_m": 12.0, "attack_range_m": 3.0,
                "effective": {"true_mu": cell.mu, "plant_mu": cell.mu, "envs": n * rs,
                              "race_size": rs,
                              "opp_speed_range": list(sc["opp_speed_range"]),
                              "opp_events": list(sc["opp_events"]),
                              "opp_event_rate": float(sc["opp_event_rate"])},
                "start_fingerprint": {
                    "schema_version": 1,
                    "physical_sha256": "a" * 64, "physical_tensors": 20,
                    "actor_input_sha256": "b" * 64, "actor_input_tensors": 6,
                    "calibration_sha256": "c" * 64, "calibration_tensors": 8,
                    "obs_spec_sha256": "d" * 64, "sim_t": 0.3, "sim_imu_phase": 1,
                    "n_envs_total": n * rs, "race_size": rs},
            }})
    res = tmp_path / "results.json"
    res.write_text(json.dumps({"cells": cells}))

    out = tmp_path / "lb.md"
    env = dict(os.environ)
    r = subprocess.run([sys.executable, "-m", "f1sim.learn.benchmark", "report",
                        "--suite", str(sp), "--roster", str(rj), "--results", str(res),
                        "--out", str(out)], capture_output=True, text=True, cwd=str(tmp_path),
                       env=env)
    assert r.returncode == 0, r.stderr
    md = out.read_text()
    assert "## Traffic" in md
    for v in ("slow", "pace", "event", "pair"):
        assert f"· {v} " in md or f"· {v} |" in md, f"{v} scenario missing from the table"
    assert "all scenarios" in md
    assert "0.750" in md                      # 3/4 clean, derived from the raw outcomes
    assert "N/A (category not evaluated)" in md      # S, A and O were never run
