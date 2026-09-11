"""Paired starts: the same cell must begin identically for every system.

Graph warm-up and controller construction run before the reset and consume the global RNG by
different amounts per arm. Without a seeded FINAL reset each system would start from a different
physical state, and every paired comparison in the benchmark would be between different initial
conditions rather than between policies.

These perturb the global RNG before preparation, exactly as a differently-sized controller would.
"""
from __future__ import annotations
import importlib
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("f1sim.gym_env")


from .test_model_adapter import make_checkpoint, TEST_MAP, MU_LOW          # noqa: E402

SEED = 4401


@pytest.fixture
def ma(bench):
    return importlib.import_module("f1sim.learn.benchmark.model_adapter")


@pytest.fixture
def rn(bench):
    return importlib.import_module("f1sim.learn.benchmark.runner")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    p, spec = make_checkpoint(tmp_path_factory.mktemp("paired"), "paired_legacy")
    return {"path": p, "arm": "legacy"}, {"spec": spec, "cap": 9.0}


def _start_state(ma, rn, ckpt, *, rng_noise: int, seed=SEED):
    """Prepare a cell with the global RNG in a different place, then take the seeded start."""
    entry, extra = ckpt
    torch.manual_seed(99)
    for _ in range(rng_noise):
        torch.rand(17)                              # stand-in for a differently-sized controller
    suite = {"envs": 2, "speed_cap": 9.0, "sensor_noise": True, "budget_laps": 0.2,
             "race_size": 1}
    prep = ma.prepare_cell(entry, extra, {"map": TEST_MAP, "true_mu": MU_LOW, "seed": seed,
                                          "envs": 2}, suite, "cpu")
    try:
        obs = rn._reset_obs(prep.env, seed)
        return (prep.env.sim.state.clone(), prep.env.sim.s.clone(),
                {k: v.clone() for k, v in obs.items() if torch.is_tensor(v)},
                {k: v.clone() for k, v in prep.env.sim.P.items()})
    finally:
        prep.close()


def test_physical_start_is_identical_despite_a_disturbed_global_rng(ma, rn, ckpt):
    a_state, a_s, _a_obs, _a_p = _start_state(ma, rn, ckpt, rng_noise=0)
    b_state, b_s, _b_obs, _b_p = _start_state(ma, rn, ckpt, rng_noise=37)
    assert torch.equal(a_state, b_state), "physical start moved with the global RNG"
    assert torch.equal(a_s, b_s)


def test_actor_input_is_identical_despite_a_disturbed_global_rng(ma, rn, ckpt):
    """The observation the policy sees must pair too, not only the plant."""
    _a, _as, a_obs, _ap = _start_state(ma, rn, ckpt, rng_noise=0)
    _b, _bs, b_obs, _bp = _start_state(ma, rn, ckpt, rng_noise=11)
    assert set(a_obs) == set(b_obs) and a_obs
    for k in a_obs:
        assert torch.equal(a_obs[k], b_obs[k]), f"actor input {k!r} differs between paired starts"


def test_calibration_and_friction_pair_too(ma, rn, ckpt):
    _a, _as, _ao, a_p = _start_state(ma, rn, ckpt, rng_noise=0)
    _b, _bs, _bo, b_p = _start_state(ma, rn, ckpt, rng_noise=23)
    for k in a_p:
        assert torch.equal(a_p[k], b_p[k]), f"randomised parameter {k!r} differs; mu included"
    assert float(a_p["mu"].mean()) == pytest.approx(MU_LOW)


def test_a_different_seed_really_does_move_the_start(ma, rn, ckpt):
    """Otherwise the pairing test above would pass on a constant start and prove nothing."""
    a_state, _a, _ao, _ap = _start_state(ma, rn, ckpt, rng_noise=0, seed=SEED)
    c_state, _c, _co, _cp = _start_state(ma, rn, ckpt, rng_noise=0, seed=SEED + 1)
    assert not torch.equal(a_state, c_state)


def test_the_seeded_reset_defines_the_start_rather_than_inheriting_it(ma, rn, ckpt):
    """What is actually true, measured rather than assumed.

    The unseeded reset is *stable* against a disturbed global RNG, because `prepare_cell` reseeds
    `sim.gen`. But it yields a DIFFERENT start from the seeded one: it inherits whatever the adapter
    happened to leave in that generator. The seeded FINAL reset is the contract because the start is
    then defined by the cell's declared seed and nothing else -- not by how much RNG a particular
    arm's construction consumed, and not by the adapter's internals.
    """
    entry, extra = ckpt
    suite = {"envs": 2, "speed_cap": 9.0, "sensor_noise": True, "budget_laps": 0.2, "race_size": 1}

    def start(noise, seed):
        torch.manual_seed(99)
        for _ in range(noise):
            torch.rand(17)
        prep = ma.prepare_cell(entry, extra, {"map": TEST_MAP, "true_mu": MU_LOW, "seed": SEED,
                                              "envs": 2}, suite, "cpu")
        try:
            rn._reset_obs(prep.env, seed)
            return prep.env.sim.state.clone()
        finally:
            prep.close()

    assert torch.equal(start(0, SEED), start(37, SEED))      # the contract: seeded pairs
    assert torch.equal(start(0, None), start(37, None))      # unseeded is stable today...
    assert not torch.equal(start(0, SEED), start(0, None))   # ...but is a different start
