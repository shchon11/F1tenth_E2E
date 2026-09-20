"""`<run>/progress.jsonl`: the record a training run keeps of itself.

The console's dashboard was blank for every run of the last two days. It read the progress line out
of W&B's `output.log` -- which wandb 0.29 no longer writes -- and parsed it with one anchored regex
that any added term broke. Both of those are somebody else's side effect. These tests are about the
replacement: the trainers write their own curve, in the run directory, in a shape that gains keys
instead of failing to match.

The parser tests use lines taken verbatim from `~/f1sim_runs/*/wandb/*/files/output.log` and from
`work/interactive-teacher/work/logs/cl_it_calib.log`, copied in rather than read from disk: they are
a frozen record of formats that really shipped, and the files they came from belong to other runs.
"""
import json
import os
import sys

import pytest

from f1sim.learn import common
from f1sim.viewer.console import training as T


# ================================================================ the writer
def test_the_log_survives_everything_a_metric_dict_contains(tmp_path):
    """numpy scalars, 0-d tensors and NaN are what a metric dict is made of; `json.dumps` refuses
    the first two and a dropped key would silently shorten one series against the others."""
    import numpy as np
    import torch

    p = common.ProgressLog(str(tmp_path))
    p.write({"kind": "ppo", "update": 1, "np": np.float32(1.5), "t": torch.tensor(2.0),
             "lap_s": float("nan"), "gate": float("inf"), "name": "arm", "nothing": None})
    p.close()
    line = (tmp_path / "progress.jsonl").read_text().strip()
    rec = json.loads(line)
    assert rec["np"] == 1.5 and rec["t"] == 2.0
    assert rec["lap_s"] != rec["lap_s"] and rec["gate"] == float("inf")
    assert rec["name"] == "arm" and "nothing" not in rec


def test_every_write_is_on_disk_before_the_next_one(tmp_path):
    """The dashboard reads this file while the run is still going. A buffered write means the page
    shows nothing for the first four kilobytes of a run -- which is most of a short one."""
    p = common.ProgressLog(str(tmp_path))
    for k in range(3):
        p.write({"kind": "ppo", "update": k})
        assert len((tmp_path / "progress.jsonl").read_text().splitlines()) == k + 1
    p.close()


def test_a_directory_it_cannot_write_to_does_not_take_the_run_with_it(tmp_path):
    """The file is a side output, not a result: a full disk must not end a twelve-hour run."""
    p = common.ProgressLog(str(tmp_path / "nope"))
    p.write({"kind": "ppo", "update": 1})           # no exception
    p.close()
    assert p.error


# ================================================================ the trainers
@pytest.mark.slow
def test_ppo_writes_the_schema_the_dashboard_plots(tmp_path, monkeypatch):
    from f1sim.learn import ppo

    monkeypatch.setenv("F1SIM_RUNS", str(tmp_path))
    monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [
        "ppo", "--name", "prog_smoke", "--device", "cpu", "--sim-backend", "eager",
        "--tracks", "gen:control:1400", "--envs", "4", "--race-size", "2", "--opponent", "teacher",
        "--action-mode", "plan", "--horizon", "4", "--minibatch", "8", "--epochs", "1",
        "--total", "32", "--critic-warmup", "0", "--episode-s", "4.0", "--scan-stack", "3",
        "--hist-len", "4", "--log-every", "1", "--save-every", "1000", "--wandb", "disabled",
        "--car-contact-penalty", "1.0", "--overtake-sustained", "1.5", "1.0", "2.0",
        "--ttc-penalty", "1.0",
    ])
    ppo.main()

    path = tmp_path / "prog_smoke" / common.PROGRESS_FILENAME
    assert path.is_file(), "the run left no progress.jsonl next to its checkpoints"
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) >= 2, "a run with two updates logged fewer than two points"
    for rec in rows:
        assert rec["kind"] == "ppo"
        # the stable schema, exactly as `docs/training.md` documents it
        for key in ("update", "total", "steps", "rew_per_step", "coll_per_km", "prog_m", "lap_s",
                    "gate", "tk", "kl_ref", "sps", "wall_s"):
            assert key in rec, key
            assert isinstance(rec[key], (int, float))
    assert [r["update"] for r in rows] == list(range(1, len(rows) + 1))
    assert rows[-1]["total"] == rows[0]["total"] and rows[-1]["wall_s"] >= rows[0]["wall_s"]
    # every per-term reward component, under its W&B name
    assert {"reward/progress_per_step", "reward/collision_per_step",
            "reward/car_contact_per_step"} <= set(rows[-1])
    # and the traffic the opponents create, because these coefficients are on
    assert {"traffic/car_contacts_per_min", "traffic/passes_held_per_min",
            "traffic/wall_collisions_per_min", "traffic/ttc_share"} <= set(rows[-1])
    # written whatever --wandb was: this run had it disabled
    assert not (tmp_path / "prog_smoke" / "wandb").exists()

    p = T.read_progress(str(tmp_path / "prog_smoke"), runs_dir=str(tmp_path))
    assert p.kind == "ppo" and p.source_kind == "progress"
    assert p.n_points == len(rows) and not p.reason
    assert len(p.coll) == len(p.lap) == len(p.rew) == p.n_points


@pytest.mark.slow
def test_dagger_writes_the_student_and_the_teacher_side_by_side(tmp_path, monkeypatch):
    from f1sim.learn import dagger

    monkeypatch.setenv("F1SIM_RUNS", str(tmp_path))
    monkeypatch.setattr(common, "RUNS_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", [
        "dagger", "--name", "dag_smoke", "--tracks", "gen:control:1400", "--envs", "4",
        "--action-mode", "plan", "--scan-stack", "3", "--hist-len", "4",
        "--iters", "2", "--steps", "8", "--epochs", "1", "--batch", "16", "--eval-steps", "8",
        "--device", "cpu", "--eager", "--wandb", "disabled",
    ])
    dagger.main()

    path = tmp_path / "dag_smoke" / common.PROGRESS_FILENAME
    assert path.is_file()
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows) == 2
    for k, rec in enumerate(rows):
        assert rec["kind"] == "dagger" and rec["iter"] == k and rec["total"] == 2
        for key in ("beta", "samples", "loss", "student_coll_per_km", "student_prog_mps",
                    "student_lap_s", "teacher_coll_per_km", "teacher_prog_mps", "teacher_lap_s",
                    "wall_s"):
            assert key in rec, key
            assert isinstance(rec[key], (int, float))

    p = T.read_progress(str(tmp_path / "dag_smoke"), runs_dir=str(tmp_path))
    assert p.kind == "dagger" and p.n_points == 2
    assert len(p.get("student_coll_per_km")) == len(p.get("teacher_coll_per_km")) == 2
    assert [s.title for s in T.charts_for(p)][0].startswith("충돌 / km")


# ================================================================ the tolerant text parser
#: Real lines, and every one of them is a line the old single regex read as nothing.
PPO_PLAIN = ("upd 12/64 steps 0.20M cap 9.0 | rew/step 1.234 coll 0.900/km prog 41.2 m lap 12.3 s "
             "| gate 0.90 (149.0 tk) | kl_ref 0.012 | 4100 steps/s")
PPO_FUTURE = ("upd 1/1016 steps 0.0M cap 9.0 | rew/step 0.005 coll 947.7/km prog 1 m lap nan s "
              "| gate 947.7 (0 tk) | kl_ref 0.006 | fut 0.802 | 96 steps/s")
PPO_FLOOR = ("upd 3/100 steps 0.1M cap 9.0 | rew/step -0.010 coll 12.1/km prog 30 m lap 14.1 s "
             "| gate inf (0 tk) | kl_ref 0.108 | floor bce 0.120 P 0.83 R 0.44 rate 0.031 pw 27 "
             "| 210 steps/s")
DAGGER_LINE = ("iter 0: beta 1.00 samples 64500 loss 0.0676 | student 52.6 coll/km "
               "(worst 202.6 on gen:control:1402+hard3720) prog 3.09 m/s lap 14.8 s | "
               "teacher 7.2 coll/km (worst 58.8) lap 13.6 s | 271+21+873 s")


def test_the_plain_ppo_line_still_reads():
    p = T.parse_progress(PPO_PLAIN)
    assert p.kind == "ppo" and (p.update, p.n_updates) == (12, 64) and p.steps_m == 0.20
    assert p.cap == 9.0
    assert (p.rew[-1], p.coll[-1], p.prog[-1], p.lap[-1]) == (1.234, 0.9, 41.2, 12.3)
    assert (p.kl[-1], p.sps[-1]) == (0.012, 4100.0)
    assert (p.get("gate")[-1], p.get("tk")[-1]) == (0.90, 149.0)


def test_a_term_inserted_before_steps_per_second_does_not_blank_the_page():
    """`| fut 0.802 |` is an `--aux-future` arm. The old regex required `kl_ref … | N steps/s` with
    nothing between them, so 172 updates of `cl_future_s701` read as zero points."""
    p = T.parse_progress(PPO_FUTURE)
    assert p.n_points == 1 and p.coll[-1] == 947.7 and p.sps[-1] == 96.0
    assert p.get("fut")[-1] == 0.802                # and the new term is a series of its own
    assert p.lap[-1] != p.lap[-1]                   # `lap nan s` is a NaN, not a missing point


def test_five_floor_head_terms_and_an_infinite_gate_are_read():
    """`gate inf` is what an unscored curriculum gate prints. The old alternation knew `nan` and
    digits, so the whole line went unread the moment the gate had no track to score yet."""
    p = T.parse_progress(PPO_FLOOR)
    assert p.n_points == 1 and p.get("gate")[-1] == float("inf")
    assert (p.coll[-1], p.lap[-1], p.kl[-1], p.sps[-1]) == (12.1, 14.1, 0.108, 210.0)
    assert p.get("bce")[-1] == 0.120 and p.get("rate")[-1] == 0.031


def test_the_dagger_line_is_parsed_at_all():
    """Nothing read it before: a DAgger run's charts were blank by construction."""
    p = T.parse_progress(DAGGER_LINE)
    assert p.kind == "dagger" and p.update == 1
    assert (p.get("beta")[-1], p.get("samples")[-1], p.get("loss")[-1]) == (1.0, 64500.0, 0.0676)
    assert p.get("student_coll_per_km")[-1] == 52.6
    assert p.get("student_coll_per_km_worst")[-1] == 202.6
    assert p.get("student_prog_mps")[-1] == 3.09 and p.get("student_lap_s")[-1] == 14.8
    # the teacher's `lap 13.6 s` sits in its own segment and must not overwrite the student's
    assert p.get("teacher_coll_per_km")[-1] == 7.2 and p.get("teacher_lap_s")[-1] == 13.6


def test_a_track_name_full_of_digits_is_not_read_as_a_metric():
    """`(worst 202.6 on gen:control:1402+hard3720)`: `control` is followed by a colon, not by a
    number, and `hard3720` is part of a word."""
    p = T.parse_progress(DAGGER_LINE)
    assert "control" not in p.series and "hard3720" not in p.series


def test_mixed_formats_in_one_file_all_land_in_the_same_series():
    """A run resumed across the change has JSON and text in whatever file the console reads."""
    text = "\n".join([PPO_PLAIN,
                      json.dumps({"kind": "ppo", "update": 13, "total": 64, "steps": 221184,
                                  "coll_per_km": 0.8, "lap_s": 12.1, "sps": 4200}),
                      PPO_FUTURE])
    p = T.parse_progress(text)
    assert p.n_points == 3
    assert p.coll == [0.9, 0.8, 947.7]
    assert p.sps == [4100.0, 4200.0, 96.0]


def test_series_stay_aligned_when_a_metric_is_missing_from_a_point():
    """A chart draws y against its index. A `lap` series shorter than `coll` puts every lap time
    against the wrong update."""
    text = "\n".join([PPO_FUTURE, PPO_PLAIN, PPO_FLOOR])
    p = T.parse_progress(text)
    assert len(p.get("fut")) == len(p.get("bce")) == p.n_points == 3
    assert p.get("fut")[0] == 0.802 and p.get("fut")[1] != p.get("fut")[1]


def test_a_traceback_is_still_reported_as_the_error():
    p = T.parse_progress(PPO_PLAIN + "\nTraceback (most recent call last):\n")
    assert p.error and "Traceback" in p.error


def test_finished_is_the_last_update_of_the_run():
    p = T.parse_progress(PPO_PLAIN.replace("upd 12/64", "upd 64/64"))
    assert p.finished and p.fraction == 1.0
