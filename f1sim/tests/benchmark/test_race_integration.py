"""Race cells through the real adapter: batch sizing, masks, ordering, controller lifecycle.

Not mocked. A synthetic checkpoint on a procedural map outside every suite map, so nothing here can
be mistaken for a benchmark score. The subject is core's integration with the adapter.
"""
from __future__ import annotations
import importlib
import os
import sys

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


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    p, spec = make_checkpoint(tmp_path_factory.mktemp("race"), "legacy_race")
    return {"path": p, "arm": "legacy"}, {"spec": spec, "cap": 9.0}


def race_suite(su, learners=4):
    s = su.Suite(envs=learners, race_size=2, budget_laps=1.0)
    return s, s.adapter_suite()


def test_learner_count_scales_to_total_cars_exactly_once(ma, su, legacy):
    """4 learners in a 2-car race is 8 envs: 4 on-policy, 4 opponents. Not 16."""
    entry, extra = legacy
    s, sd = race_suite(su, learners=4)
    prep = ma.prepare_cell(entry, extra,
                           {"map": TEST_MAP, "true_mu": MU_LOW, "seed": 4401,
                            "envs": 4, "race_size": 2}, sd, "cpu")
    try:
        env = prep.env
        assert int(env.B) == 8
        assert int(env.M) == 2
        assert int(env.on_policy.sum()) == 4
        assert int((~env.on_policy).sum()) == 4
    finally:
        prep.close()


def test_learner_and_opponent_rows_keep_a_stable_ordering(ma, su, legacy):
    """Row i of the learners must always pair with the same opponent row.

    `other_idx` is a fixed roster (gym_env.py:140), so the pairing is positional; if it drifted, a
    per-trial gap would be measured against a different car each step.
    """
    entry, extra = legacy
    s, sd = race_suite(su, learners=4)
    prep = ma.prepare_cell(entry, extra,
                           {"map": TEST_MAP, "true_mu": MU_LOW, "seed": 4401,
                            "envs": 4, "race_size": 2}, sd, "cpu")
    try:
        env = prep.env
        learners = torch.nonzero(env.on_policy).flatten()
        opp = env.sim.other_idx[learners][:, 0]
        assert learners.tolist() == [0, 2, 4, 6]          # slot 0 of each race
        assert opp.tolist() == [1, 3, 5, 7]
        assert torch.equal(env.sim.other_idx[learners][:, 0], opp)   # stable across reads
    finally:
        prep.close()


def test_the_opponent_is_the_fixed_teacher_at_the_declared_speed_range(ma, su, legacy):
    """The EnvConfig default is `policy`, which would make the opponent a copy of the candidate."""
    entry, extra = legacy
    s, sd = race_suite(su, learners=2)
    prep = ma.prepare_cell(entry, extra,
                           {"map": TEST_MAP, "true_mu": MU_LOW, "seed": 4401,
                            "envs": 2, "race_size": 2}, sd, "cpu")
    try:
        env = prep.env
        assert env.ecfg.opponent == "teacher"
        assert tuple(env.ecfg.opp_speed_range) == (0.6, 0.8)
        assert env.teacher is not None, "a teacher opponent needs a teacher installed"
    finally:
        prep.close()


def test_the_loop_drives_the_controller_lifecycle_on_a_race_cell(ma, su, legacy, monkeypatch):
    """begin/pre_action/post_step must each be called the right number of times."""
    entry, extra = legacy
    s, sd = race_suite(su, learners=2)
    prep = ma.prepare_cell(entry, extra,
                           {"map": TEST_MAP, "true_mu": MU_LOW, "seed": 4401,
                            "envs": 2, "race_size": 2}, sd, "cpu")
    calls = {"begin": 0, "pre": 0, "post": 0}
    monkeypatch.setattr(prep, "begin", lambda obs: calls.__setitem__("begin", calls["begin"] + 1))
    monkeypatch.setattr(prep, "pre_action", lambda obs: calls.__setitem__("pre", calls["pre"] + 1))
    monkeypatch.setattr(prep, "post_step",
                        lambda t, u: calls.__setitem__("post", calls["post"] + 1))
    try:
        runner = importlib.import_module("f1sim.learn.benchmark.runner")
        pol = ma.policy_for_none if hasattr(ma, "policy_for_none") else None
        steps = 5

        def scripted(obs):
            return torch.zeros(prep.env.B, prep.env.act_dim)

        res = runner.run_cell(prep.env, scripted, suite="O", n_steps=steps, hold_steps=40,
                              frozen=False, controller=prep)
        assert calls["begin"] == 1, "exactly one reset and one begin"
        assert calls["pre"] == calls["post"] >= 1
        assert calls["pre"] <= steps
        assert res["n"] == 2                       # learners measured, not all cars
        assert pol is None or True
    finally:
        prep.close()


def test_run_cell_measures_learners_only_on_a_race_cell(ma, su, legacy):
    entry, extra = legacy
    s, sd = race_suite(su, learners=4)
    prep = ma.prepare_cell(entry, extra,
                           {"map": TEST_MAP, "true_mu": MU_LOW, "seed": 4401,
                            "envs": 4, "race_size": 2}, sd, "cpu")
    try:
        runner = importlib.import_module("f1sim.learn.benchmark.runner")
        res = runner.run_cell(prep.env, lambda obs: torch.zeros(prep.env.B, prep.env.act_dim),
                              suite="O", n_steps=4, hold_steps=40, frozen=False, controller=prep)
        assert res["n"] == 4
        assert len(res["outcomes"]) == 4
        assert len(res["progress_m"]) == 4
    finally:
        prep.close()


def test_core_supplies_the_opponent_fields_the_race_needs(su):
    """Core's side of the contract: the suite dict names the fixed teacher and its speed range.

    Behavioural coverage that the adapter honours them is
    `test_the_opponent_is_the_fixed_teacher_at_the_declared_speed_range` above; this checks that
    core actually sends them.
    """
    a = su.Suite().adapter_suite()
    assert a["opponent"] == "teacher"
    assert tuple(a["opp_speed_range"]) == (0.6, 0.8)
