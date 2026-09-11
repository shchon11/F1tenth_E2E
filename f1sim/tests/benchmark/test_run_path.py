"""`run` on the real adapter path, with a synthetic checkpoint.

The previous version of this file mocked `_load_policy` and `_run_one`, which hid every real defect
in the path they stood in for: a missing `seed`, the wrong observation spec, no controller install,
a hardcoded budget. Nothing here is mocked below the CLI: a real checkpoint is loaded, a real env is
built through `model_adapter`, and real cells are driven.

The map is procedural and outside every suite map, and the checkpoint is synthetic, so nothing here
is a benchmark score.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("f1sim.gym_env")


from .test_model_adapter import make_checkpoint, TEST_MAP, MU_LOW          # noqa: E402


@pytest.fixture
def mm(bench):
    return importlib.import_module("f1sim.learn.benchmark.__main__")


@pytest.fixture
def su(bench):
    return importlib.import_module("f1sim.learn.benchmark.suite")


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    p, _spec = make_checkpoint(tmp_path_factory.mktemp("runpath"), "run_legacy")
    return p, hashlib.sha256(open(p, "rb").read()).hexdigest()


def _suite(su, tmp_path, *, placements=True):
    s = su.Suite(solo_maps=(TEST_MAP,), obstacle_maps=(), race_maps=(),
                 solo_mus=(MU_LOW,), seeds=(4401,), envs=2, budget_laps=0.2)
    if placements:
        s.placements = {TEST_MAP: {"placement": {"s_obs_m": 10.0, "side": 1}, "proofs": {}}}
    p = tmp_path / "suite.json"
    s.save(str(p))
    return s, str(p)


def _roster(tmp_path, path, sha):
    p = tmp_path / "roster.json"
    p.write_text(json.dumps({"systems": [
        {"system_id": "sys", "path": path, "checkpoint_sha256": sha, "controller_arm": "legacy"}]}))
    return str(p)


#: The pinned student the independence gate crosses. `run` requires it whenever it runs the gate.
EST = ("/home/shchon11/Documents/Codex/2026-09-10/new-chat/work/learning-next/"
       "continuous-learning/runner/estimator-out/estimator_seed401.pt")


def _args(**kw):
    base = dict(suite=None, roster=None, system="sys", out=None, device="cpu", lease=False,
                estimator=EST, gate_steps=60)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _pass_gate(monkeypatch):
    """The independence gate is proven in its own module; here it must not dominate runtime.

    It still has to report cold+warm coverage, because `run` refuses without it.
    """
    monkeypatch.setattr("f1sim.learn.benchmark.integration_gate.run_gate",
                        lambda **kw: {"opponent_commands_identical": kw.get("routed", True),
                                      "max_abs_delta": 0.0 if kw.get("routed") else 1.0,
                                      "covers_cold_and_warm": True, "uncovered_cases": []})


def test_refuses_without_lease_after_verifying_the_real_checkpoint(mm, su, tmp_path, ckpt):
    path, sha = ckpt
    _, sp = _suite(su, tmp_path)
    assert mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha),
                            out=str(tmp_path / "o"))) == 3


def test_missing_freeze_is_reported_before_the_checkpoint_is_touched(mm, su, tmp_path, ckpt):
    path, sha = ckpt
    _, sp = _suite(su, tmp_path, placements=False)
    assert mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha),
                            out=str(tmp_path / "o"))) == 1


def test_the_written_row_satisfies_the_reporters_pin_requirement(mm, su, tmp_path, ckpt,
                                                                 monkeypatch):
    """The producer/reporter seam: a row `cmd_run` actually wrote must pass the real row gate.

    `n_envs` was absent from every row the producer ever wrote, while a separate fixture-based test
    asserted the reporter rejects rows missing pins. Both passed, because nothing read the
    producer's own serialized output through the reporter's own check. Only crossing that seam
    catches it, so this reads the written file rather than a dict built here.
    """
    from f1sim.learn.benchmark import report as rp
    path, sha = ckpt
    s, sp = _suite(su, tmp_path)
    _pass_gate(monkeypatch)
    out = tmp_path / "o"
    assert mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha),
                            out=str(out), lease=True)) == 0

    row = json.loads(open(out / "sys.cells.jsonl").readline())
    missing = [k for k in rp.REQUIRED_PINS if row.get(k) in (None, "")]
    assert not missing, f"producer omitted required pins: {missing}"
    # From the declared cell, not from the result: the reporter compares the two, and sourcing it
    # from `result.n` would make that check compare the result against itself.
    assert row["n_envs"] == s.cells()[0].envs


def test_a_real_cell_runs_end_to_end_and_records_its_protocol(mm, su, tmp_path, ckpt, monkeypatch):
    """The whole path: adapter load, real env, real driving, row written with identity."""
    path, sha = ckpt
    s, sp = _suite(su, tmp_path)
    _pass_gate(monkeypatch)
    out = tmp_path / "o"
    assert mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha),
                            out=str(out), lease=True)) == 0

    rows = [json.loads(l) for l in open(out / "sys.cells.jsonl")]
    assert len(rows) == len(s.cells()) == 1
    r = rows[0]
    assert r["checkpoint_sha256"] == sha
    assert r["controller_arm"] == "legacy"
    assert r["suite_freeze_sha256"] == s.freeze_hash()
    assert r["identity_sha256"]
    assert r["effective"], "the adapter's effective protocol must be recorded"
    # raw per-trial data, not just metadata
    res = r["result"]
    assert res["n"] == 2 and len(res["outcomes"]) == 2
    assert len(res["progress_m"]) == 2 and len(res["distance_m"]) == 2
    assert all(o is not None and o.get("reason") is not None or o.get("success")
               for o in res["outcomes"])


def test_the_declared_friction_is_what_the_row_reports(mm, su, tmp_path, ckpt, monkeypatch):
    path, sha = ckpt
    s, sp = _suite(su, tmp_path)
    _pass_gate(monkeypatch)
    out = tmp_path / "o"
    mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha), out=str(out), lease=True))
    r = json.loads(open(out / "sys.cells.jsonl").readline())
    assert r["mu"] == pytest.approx(MU_LOW)
    eff = r["effective"]
    if "true_mu" in eff:
        assert float(eff["true_mu"]) == pytest.approx(MU_LOW)


def test_resume_skips_only_rows_of_the_same_protocol(mm, su, tmp_path, ckpt, monkeypatch):
    path, sha = ckpt
    s, sp = _suite(su, tmp_path)
    _pass_gate(monkeypatch)
    out = tmp_path / "o"
    mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha), out=str(out), lease=True))
    mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha), out=str(out), lease=True))
    rows = [json.loads(l) for l in open(out / "sys.cells.jsonl")]
    assert len(rows) == 1, "a second run must not duplicate a completed cell"


def test_resume_refuses_a_file_from_another_protocol(mm, su, tmp_path, ckpt, monkeypatch):
    """Resuming on cell_id alone would mix two protocols in one results file."""
    path, sha = ckpt
    s, sp = _suite(su, tmp_path)
    _pass_gate(monkeypatch)
    out = tmp_path / "o"
    mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha), out=str(out), lease=True))
    f = out / "sys.cells.jsonl"
    row = json.loads(f.read_text().splitlines()[0])
    row["suite_freeze_sha256"] = "0" * 64                  # a row from a different freeze
    f.write_text(json.dumps(row) + "\n")
    with pytest.raises(SystemExit, match="different protocol"):
        mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha), out=str(out), lease=True))


def test_a_vacuous_gate_blocks_scoring(mm, su, tmp_path, ckpt, monkeypatch):
    path, sha = ckpt
    _, sp = _suite(su, tmp_path)
    monkeypatch.setattr("f1sim.learn.benchmark.integration_gate.run_gate",
                        lambda **kw: {"opponent_commands_identical": True, "max_abs_delta": 0.0,
                                      "covers_cold_and_warm": True, "uncovered_cases": []})
    assert mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha),
                            out=str(tmp_path / "o"), lease=True)) == 4


def test_run_refuses_without_an_estimator_for_the_gate(mm, su, tmp_path, ckpt):
    """The gate crosses the estimated arm, which has no default student."""
    path, sha = ckpt
    _, sp = _suite(su, tmp_path)
    assert mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha),
                            out=str(tmp_path / "o"), lease=True, estimator=None)) == 2


def test_run_refuses_a_gate_that_did_not_cover_cold_and_warm(mm, su, tmp_path, ckpt, monkeypatch):
    """A step count is not evidence of either regime; an uncovered gate must block scoring."""
    path, sha = ckpt
    _, sp = _suite(su, tmp_path)
    monkeypatch.setattr("f1sim.learn.benchmark.integration_gate.run_gate",
                        lambda **kw: {"opponent_commands_identical": kw.get("routed", True),
                                      "max_abs_delta": 0.0 if kw.get("routed") else 1.0,
                                      "covers_cold_and_warm": False,
                                      "uncovered_cases": ["A/estimated"]})
    assert mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha),
                            out=str(tmp_path / "o"), lease=True)) == 4


def test_resume_refuses_a_duplicated_cell(mm, su, tmp_path, ckpt, monkeypatch):
    """Two rows for one cell is a double measurement, not a resumable one.

    `_resume` collected into a set, so a repeat silently collapsed to a single "done" and the
    duplicate — which the report refuses outright — was never seen on the runner side.
    """
    path, sha = ckpt
    s, sp = _suite(su, tmp_path)
    _pass_gate(monkeypatch)
    out = tmp_path / "o"
    assert mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha),
                            out=str(out), lease=True)) == 0
    f = out / "sys.cells.jsonl"
    row = f.read_text().splitlines()[0]
    f.write_text(row + "\n" + row + "\n")            # the same cell, recorded twice
    with pytest.raises(SystemExit, match="appears more than once"):
        mm.cmd_run(_args(suite=sp, roster=_roster(tmp_path, path, sha),
                         out=str(out), lease=True))
