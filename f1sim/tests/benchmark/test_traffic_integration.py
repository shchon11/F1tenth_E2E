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
