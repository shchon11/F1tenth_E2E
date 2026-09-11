"""Real CLI invocation and the recorder contracts the measurement depends on.

No mocks below the CLI: the gate runs the actual ControllerRuntime with the pinned frozen student,
and the recorder fixtures drive real traces. No candidate weights, no scores.
"""
from __future__ import annotations
import importlib
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("f1sim.gym_env")


from .test_model_adapter import make_checkpoint, TEST_MAP, MU_LOW          # noqa: E402

EST = ("/home/shchon11/Documents/Codex/2026-09-10/new-chat/work/learning-next/"
       "continuous-learning/runner/estimator-out/estimator_seed401.pt")


@pytest.fixture
def mm(bench):
    return importlib.import_module("f1sim.learn.benchmark.__main__")


@pytest.fixture
def rn(bench):
    return importlib.import_module("f1sim.learn.benchmark.runner")


# ------------------------------------------------------------------ CLI contracts

def test_gate_cli_parses_and_uses_the_real_signature(mm):
    """`run_gate` takes `arms`, not `arm`. The CLI called the wrong name and could never run."""
    import inspect
    from f1sim.learn.benchmark.integration_gate import run_gate
    params = inspect.signature(run_gate).parameters
    assert "arms" in params and "arm" not in params
    src = inspect.getsource(mm.cmd_gate)
    assert "arms=" in src and "estimator_path=" in src


@pytest.mark.slow
def test_gate_cli_runs_end_to_end_with_the_pinned_student(mm, capsys):
    """The published command, invoked for real on CPU with the frozen student."""
    if not os.path.exists(EST):
        pytest.skip("pinned estimator not present on this machine")
    rc = mm.main(["gate", "--estimator", EST, "--steps", "60"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "GATE PASSED" in out
    assert "cold+warm covered=True" in out


@pytest.mark.slow
def test_gate_refuses_when_cold_and_warm_are_not_covered(mm, capsys):
    """Too few steps to fill the history: the gate must refuse, not claim coverage."""
    if not os.path.exists(EST):
        pytest.skip("pinned estimator not present on this machine")
    rc = mm.main(["gate", "--estimator", EST, "--steps", "5"])
    assert rc == 3
    assert "GATE INCOMPLETE" in capsys.readouterr().err


def test_run_refuses_without_an_estimator_for_the_gate(mm, su_suite_tmp):
    """The gate crosses the estimated arm, which has no default student."""
    sp, rj = su_suite_tmp
    rc = mm.main(["run", "--suite", sp, "--roster", rj, "--system", "sys", "--lease"])
    assert rc == 2


@pytest.fixture
def su_suite_tmp(bench, tmp_path):
    import hashlib
    import json
    su = importlib.import_module("f1sim.learn.benchmark.suite")
    s = su.Suite(solo_maps=(TEST_MAP,), obstacle_maps=(), race_maps=(), solo_mus=(MU_LOW,),
                 seeds=(4401,), envs=2, budget_laps=0.2)
    s.placements = {TEST_MAP: {"placement": {"s_obs_m": 10.0, "side": 1}, "proofs": {}}}
    sp = tmp_path / "suite.json"
    s.save(str(sp))
    p, _spec = make_checkpoint(tmp_path, "cli_legacy")
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    rj = tmp_path / "roster.json"
    rj.write_text(json.dumps({"systems": [
        {"system_id": "sys", "path": p, "checkpoint_sha256": sha, "controller_arm": "legacy"}]}))
    return str(sp), str(rj)


# ------------------------------------------------------------------ recorder contracts

@pytest.mark.slow
def test_heading_error_is_real_and_nonzero_when_the_car_is_turned(rn):
    """It read a field that does not exist and a catch-all made it zero, so wrong-way and spin
    exposure were silently always zero."""
    from f1sim.learn.benchmark.integration_gate import _build, _reset
    env = _build(TEST_MAP, envs=2, race_size=1, seed=4401, force_teacher=True)
    _reset(env, 4401)
    trace = rn._PreResetTrace(env)
    with trace:
        env.step(torch.zeros(env.B, env.act_dim))
    aligned = trace._heading_error()
    assert torch.isfinite(aligned).all()

    env.sim.state[:, 2] += torch.pi                 # point the car backwards along the lane
    turned = trace._heading_error()
    assert torch.isfinite(turned).all()
    assert (turned.abs() > 1.0).all(), "a reversed car must show a large heading error"
    assert (turned.abs() <= torch.pi + 1e-6).all(), "heading error must be wrapped"
    # and the wrong-way condition TrialAccumulator uses actually fires
    assert (torch.cos(turned) < 0).all()


def test_heading_error_does_not_swallow_an_api_change(rn):
    """No catch-all: a missing field must raise rather than report clean driving."""
    import inspect
    src = inspect.getsource(rn._PreResetTrace._heading_error)
    assert "except Exception" not in src
    assert "cl_tangent" in src


def mk(rn, suite="O", n=1):
    return rn.CellRecorder(n=n, suite=suite, track_length_m=50.0, vehicle_length=0.58,
                           vehicle_width=0.31, hold_steps=40)


def step(rec, **kw):
    n = rec.n
    base = dict(progress=[1.0] * n, speed=[2.0] * n, dt=0.025,
                collision=[False] * n, truncated=[False] * n)
    base.update(kw)
    rec.update(**base)


def _drive_to_one_tick_short(rec, hold=40):
    """Crossing plus hold, stopping exactly one qualifying step before the pass completes.

    The crossing ramp itself accumulates hold steps, so a fixed count overshoots -- measured, the
    ramp alone contributes 16 of the 40.
    """
    g = 3.0
    while g > -1.2:
        g -= 0.02
        step(rec, gap=[g])
    d = rec.detectors[0]
    while d._held < hold - 1 and not d.succeeded:
        step(rec, gap=[-1.2])
    assert not d.succeeded and d._held == hold - 1
    return d


def test_opponent_crash_on_the_hold_completion_tick_is_not_a_pass(rn):
    """Same-step precedence. Keying only on next-step freshness is one frame late."""
    rec = mk(rn)
    rec.seed_gaps([3.0])
    d = _drive_to_one_tick_short(rec)
    # the opponent crashes on the very tick the hold would otherwise complete
    step(rec, gap=[-1.2], opponent_incident=[True])
    assert not d.succeeded
    out = rec.finalize()
    assert out["tally"]["successes"] == 0
    assert "opponent_respawn" in out["tally"]["failures"]


def test_a_clean_hold_still_completes_without_an_incident(rn):
    """The guard must not make every pass impossible: the same trace, one tick later, succeeds."""
    rec = mk(rn)
    rec.seed_gaps([3.0])
    _drive_to_one_tick_short(rec)
    step(rec, gap=[-1.2], opponent_incident=[False])
    assert rec.outcome[0]["success"] is True


def test_yaw_rate_comes_from_the_current_step_not_the_previous_one(rn):
    """`env.last_result` still holds the previous step's IMU when the trace reads it.

    A frame of lag can miss the peak on the terminal step — which is the step a spin or a crash
    happens on, and therefore the one max |yaw rate| is meant to catch.
    """
    class R:
        def __init__(self, v):
            self.imu = torch.full((2, 3, 6), float(v))

    current, previous = R(9.0), R(2.0)
    got = rn._yaw_rate_of(current, previous)
    assert torch.allclose(got, torch.full((2,), 9.0)), "read the current step, not the previous"

    empty = type("E", (), {"imu": torch.zeros(2, 0, 6)})()
    assert torch.allclose(rn._yaw_rate_of(empty, previous), torch.full((2,), 2.0)), \
        "an IMU that delivered no samples this step falls back rather than reporting nothing"
    assert rn._yaw_rate_of(None, None) is None
