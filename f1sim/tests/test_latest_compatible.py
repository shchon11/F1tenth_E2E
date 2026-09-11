"""`latest_compatible_run` — choosing a checkpoint the default consumer can actually open.

The bug this covers: `latest_run()` returns whatever trained most recently, and once controller-arm
or conditional training is running that is a checkpoint `load_checkpoint` refuses without an opt-in
flag the default consumers do not pass. `python -m f1sim.learn.watch` and the console's unattended
"latest" both failed on the run the user is least surprised by.

`latest_run()` and the loader guards are deliberately unchanged; the selection is what moved, and it
has to stay visible -- every skipped run comes back with a reason so a caller can say which run it
opened and which newer ones it passed over.
"""
from __future__ import annotations

import dataclasses
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1sim.learn import common                                          # noqa: E402
from f1sim.learn import watch as W                                      # noqa: E402
from f1sim.learn.model import ActorCritic, save_checkpoint              # noqa: E402
from f1sim.learn.obs import ObsSpec                                     # noqa: E402
from f1sim.params import Config                                         # noqa: E402


def write_run(root, name, *, extra=None, meta_over=None, mtime=None, file="ppo_latest.pt"):
    """One run directory holding one small real checkpoint."""
    spec = ObsSpec(n_beams=Config().lidar.n_beams)
    meta = dict(n_stack=spec.scan_stack, n_beams=spec.n_beams, proprio_dim=spec.proprio_dim,
                priv_dim=17, act_dim=spec.act_dim, scan_deltas=True, temporal_encoder="cnn")
    with torch.random.fork_rng():
        torch.manual_seed(7)
        m = ActorCritic(**meta)
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, file)
    save_checkpoint(p, m, {"spec": dataclasses.asdict(spec), **(extra or {})})
    if meta_over:                       # edit the saved metadata without rebuilding the model
        ck = torch.load(p, map_location="cpu", weights_only=True)
        ck["meta"].update(meta_over)
        torch.save(ck, p)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return d


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
    return str(tmp_path)


def test_a_newer_controller_arm_run_is_skipped_for_an_older_compatible_one(runs):
    write_run(runs, "old_legacy", mtime=1000)
    write_run(runs, "new_estimated", mtime=2000,
              extra={"experiment": {"controller": {"arm": "estimated"}}})
    chosen, skipped = W.latest_compatible_run()
    assert os.path.basename(chosen) == "old_legacy"
    assert skipped == [("new_estimated", "controller arm 'estimated'")]
    # and the generic resolver still means "newest", because training depends on that
    assert os.path.basename(W.latest_run()) == "new_estimated"


def test_a_newer_conditional_run_is_skipped_too(runs):
    """A real conditional checkpoint: `cond_dim` and a matching `cond` spec, as training writes it."""
    from f1sim.learn import conditioning as C
    spec = C.spec_for("true_mu")
    write_run(runs, "old_legacy", mtime=1000)
    write_run(runs, "new_cond", mtime=2000,
              meta_over={"cond_dim": spec.dim, "cond": spec.to_meta()})
    chosen, skipped = W.latest_compatible_run()
    assert os.path.basename(chosen) == "old_legacy"
    assert len(skipped) == 1 and "conditional" in skipped[0][1], skipped


def test_conditioning_metadata_that_contradicts_itself_fails_closed(runs):
    """`cond_dim` without a matching spec is not a checkpoint this viewer should guess about --
    `load_checkpoint` raises on exactly this disagreement (`model.py:381-383`)."""
    write_run(runs, "inconsistent", mtime=2000, meta_over={"cond_dim": 3})
    chosen, skipped = W.latest_compatible_run()
    assert chosen == ""
    assert "disagrees" in skipped[0][1]


def test_an_unreadable_newest_fails_closed_and_is_reported(runs):
    write_run(runs, "old_legacy", mtime=1000)
    bad = os.path.join(runs, "half_written")
    os.makedirs(bad)
    with open(os.path.join(bad, "ppo_latest.pt"), "wb") as f:
        f.write(b"PK\x03\x04 truncated mid-write")
    os.utime(os.path.join(bad, "ppo_latest.pt"), (2000, 2000))
    chosen, skipped = W.latest_compatible_run()
    assert os.path.basename(chosen) == "old_legacy"
    assert skipped[0][0] == "half_written" and "unreadable" in skipped[0][1]


def test_nothing_compatible_returns_no_run_and_every_reason(runs):
    write_run(runs, "a_estimated", mtime=1000,
              extra={"experiment": {"controller": {"arm": "estimated"}}})
    write_run(runs, "b_fixed_low", mtime=2000,
              extra={"experiment": {"controller": {"arm": "fixed_low"}}})
    chosen, skipped = W.latest_compatible_run()
    assert chosen == ""
    assert [n for n, _ in skipped] == ["b_fixed_low", "a_estimated"]      # newest first


def test_the_probe_names_the_file_that_would_actually_be_opened(runs):
    """A run holding both checkpoints: `resolve_checkpoint` opens `ppo_latest.pt`, so that is the
    file the probe must judge. Judging `student_latest.pt` instead would accept a run whose ppo
    checkpoint is refused."""
    d = write_run(runs, "both", mtime=2000,
                  extra={"experiment": {"controller": {"arm": "estimated"}}})
    write_run(runs, "both", mtime=1500, file="student_latest.pt")        # compatible, but not opened
    assert os.path.exists(os.path.join(d, "ppo_latest.pt"))
    chosen, skipped = W.latest_compatible_run()
    assert chosen == "", "the run was judged on student_latest.pt, which is not what gets opened"
    assert skipped == [("both", "controller arm 'estimated'")]


@pytest.mark.parametrize("payload, expect", [
    ({"junk": 1}, "not a checkpoint"),
    ({"meta": 5, "state_dict": {}}, "malformed meta"),
    ({"state_dict": {}}, "not a checkpoint"),
])
def test_malformed_checkpoints_fail_closed(tmp_path, payload, expect):
    p = str(tmp_path / "x.pt")
    torch.save(payload, p)
    why = W.default_consumer_refusal(p)
    assert why is not None and expect in why


def test_a_compatible_checkpoint_is_accepted_and_actually_loads(runs):
    from f1sim.learn.model import load_checkpoint
    d = write_run(runs, "plain", mtime=1000)
    p = os.path.join(d, "ppo_latest.pt")
    assert W.default_consumer_refusal(p) is None
    load_checkpoint(p, "cpu")            # no opt-in flag: the probe's promise, checked


def test_a_run_is_ranked_by_the_file_that_gets_opened_not_by_a_newer_sibling(runs):
    """A(ppo 100, student 2000) against B(ppo 1000): B must win.

    Ranking a run by the newest file in its directory lets a fresh `student_latest.pt` promote a run
    whose stale `ppo_latest.pt` is the one `resolve_checkpoint` actually opens.
    """
    write_run(runs, "A", mtime=100)
    write_run(runs, "A", mtime=2000, file="student_latest.pt")
    write_run(runs, "B", mtime=1000)
    chosen, _ = W.latest_compatible_run()
    assert os.path.basename(chosen) == "B", "ranked by the ignored student file"


def test_a_mistyped_run_name_is_a_message_not_a_bare_stop_iteration(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        W.main(["--run", "no_such_run", "--device", "cpu"])
    assert "no checkpoint in" in str(e.value) and "no_such_run" in str(e.value)


def test_a_named_run_under_runs_dir_resolves(runs, monkeypatch):
    """`--run <name>` is resolved under RUNS_DIR; it used to only work as a path."""
    write_run(runs, "named", mtime=1000)
    base = os.path.join(runs, "named")
    assert os.path.exists(os.path.join(base, "ppo_latest.pt"))
    # resolution is what is under test, so stop before the heavy session build
    monkeypatch.setattr(W, "load_checkpoint", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("reached load")))
    with pytest.raises(RuntimeError, match="reached load"):
        W.main(["--run", "named", "--device", "cpu"])
