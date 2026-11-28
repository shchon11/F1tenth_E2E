"""compbench unit tests: scoring functions on synthetic records, the statistics, suite hashing, determinism.

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q f1sim/f1sim/compbench/tests"""
import os
import json
import shutil
import tempfile

import numpy as np
import pytest

from f1sim.compbench import score as S
from f1sim.compbench.engine_exam import split_assign

C_REC = "/data/pm-audit-2026-11-28/C_rec/solo_scan31b475_map.npz"


def synth_rem(T, B, L, v, dt=0.1, start=None):
    """distance-to-go of cars at constant speed v (m/s) from rem = start: rem decreases, wraps L -> 0"""
    start = np.full(B, L * 0.3) if start is None else np.asarray(start, float)
    t = np.arange(T + 1)[:, None] * dt
    return np.mod(start[None] - v * t, L).astype(np.float32)


def test_first_crossing_and_lap_times():
    L, v, dt = 40.0, 5.0, 0.1
    rem = synth_rem(400, 1, L, v, start=[12.25])
    k0 = S.first_crossing(rem[:, 0], L)
    # 12.25 m to the line at 0.5 m a decision -> crossed between snapshot 24 and 25
    assert k0 == 25
    P = S.progress(rem, L)
    laps, lt = S.laps_from_progress(k0, P[:, 0], L, dt)
    assert len(lt) == int((400 * dt - k0 * dt) * v // L)
    assert np.allclose(lt, L / v, atol=1e-4)


def test_progress_unwraps_and_backwards():
    L = 30.0
    rem = np.array([[5.0], [3.0], [1.0], [29.0], [27.0]], np.float32)       # forward through the line
    P = S.progress(rem, L)
    assert np.allclose(P[:, 0], [0, 2, 4, 6, 8])
    back = np.array([[5.0], [6.0], [7.0]], np.float32)
    assert np.allclose(S.progress(back, L)[:, 0], [0, -1, -2])


def test_streak_and_new_counts():
    assert S.streak([0, 0, 1, 0, 0, 0, 2, 0]) == 3
    assert S.streak([]) == 0
    cnt = np.array([[[0]], [[1]], [[1]], [[2]], [[0]], [[1]]], np.float32)     # cumulative, reset after step 3
    rs = np.array([[False], [False], [False], [True], [False], [False]])
    d = S.new_counts(cnt, rs)
    assert d[:, 0, 0].tolist() == [0, 1, 0, 0, 0, 1]


def test_bootstrap_is_deterministic_and_brackets():
    rng = np.random.default_rng(3)
    v = rng.normal(2.0, 1.0, 128)
    cl = np.arange(128) % 64
    a, b = S.boot(v, cl), S.boot(v, cl)
    assert a == b
    assert a[1] <= a[0] <= a[2]
    r = S.boot_ratio(v.clip(0), np.ones(128), cl)
    assert r[1] <= r[0] <= r[2]


def test_paired_same_rule():
    rows = [dict(map_id="m", seed=1, b=i, layout=i % 8, x=float(i % 5)) for i in range(64)]
    same = S.paired(rows, rows, "x")
    assert same["est"] == 0.0 and same["same"]
    shifted = [dict(r, x=r["x"] + 1.0) for r in rows]
    p = S.paired(rows, shifted, "x")
    assert abs(p["est"] - 1.0) < 1e-9 and not p["same"] and p["n"] == 64


def test_mde_shrinks_with_sessions():
    assert S.mde(1.0, 400) < S.mde(1.0, 100)
    assert abs(S.mde(1.0, 100) - 0.28) < 1e-9


def make_record(path, T=600, B=2, L=40.0):
    """a minimal record: car 0 laps cleanly at 5 m/s, car 1 hits a box at 20 s and then stands"""
    rem = synth_rem(T, B, L, 5.0)
    rem[200:, 1] = rem[200, 1]                                    # car 1 stops at 20 s
    cnt = np.zeros((T, B, 6), np.float32)
    cnt[199:, 1, 1] = 1.0
    cnt[199:, 1, 4] = 1.0                                         # one hard box onset at decision 199
    st = np.zeros((T + 1, B, 4), np.float32)
    st[:, :, 3] = 5.0
    st[200:, 1, 3] = 0.0
    np.savez(path, dt=np.array(0.1), rs=np.zeros((T, B), bool), rm_t=np.array(30.0), tid=np.zeros((T + 1, B), np.int16),
             lapL=np.array([L], np.float32), st=st, sid=np.zeros((T + 1, B, 3), np.int16), on=np.zeros((T + 1, B), bool),
             cnt=cnt, rem=rem, name=np.array("synthetic"), mode=np.array("map"), seed=np.array(1), map=np.array("x"),
             held=np.concatenate([np.zeros((T, B, 3), np.float32), np.full((T, B, 1), 2.0, np.float32)], 2))


def test_score_record_synthetic(tmp_path):
    p = str(tmp_path / "r.npz")
    make_record(p)
    R = S.score_record(p)
    a, b = R["sessions"]
    assert a["dnf"] == 0.0 and a["clean"] == 1.0 and a["wall_hard"] == 0 and a["box_hard"] == 0
    assert abs(a["best"] - 8.0) < 1e-3                             # 40 m at 5 m/s
    assert a["streak_hard"] == a["n_laps"] >= 6
    assert b["box_hard"] == 1 and b["clean"] == 0.0
    assert b["dnf"] == 1.0 and b["dnf_why"] == "stuck10s"
    assert b["sev_1.5-3"] == 1.0                                   # the held severity 2.0 m/s


@pytest.mark.skipif(not os.path.exists(C_REC), reason="audit C record not on this machine")
def test_real_record_scores_reproduce_and_are_deterministic():
    R1 = S.score_record(C_REC)
    R2 = S.score_record(C_REC)
    assert json.dumps(R1, default=float) == json.dumps(R2, default=float)
    S_ = R1["sessions"]
    cl = [r["layout"] for r in S_]
    best = S.boot([r["best"] for r in S_], cl, "median")
    streak = S.boot([r["streak_hard"] for r in S_], cl)
    # audit C's table for this record (MAP): best 7.78, streak 11.1, DNF 10 %
    assert abs(best[0] - 7.78) < 0.01 and abs(streak[0] - 11.09) < 0.01
    assert abs(np.mean([r["dnf"] for r in S_]) - 0.1016) < 1e-3
    assert S.determinism(C_REC, C_REC)["all_equal"]


def test_suite_hash_stable_and_sensitive(tmp_path):
    from f1sim.compbench.suite import load_suite
    a, b = load_suite("dev"), load_suite("dev")
    assert a["_sha256"] == b["_sha256"]
    d = json.load(open(a["_path"]))
    d["qual"]["remove_at"] = 61
    p = tmp_path / "dev_changed.json"
    json.dump(d, open(p, "w"))
    assert load_suite(str(p))["_sha256"] != a["_sha256"]


def test_split_assign_spreads_layouts():
    who = split_assign(128, 16, 64)
    for k in range(16):
        rows = np.nonzero(who == k)[0]
        assert len(rows) == 8
        assert len(set((rows % 64).tolist())) == 8                 # 8 different layouts per candidate


C_RACE = "/data/pm-audit-2026-11-28/C_rec/race_race33b975_map.npz"


@pytest.mark.skipif(not os.path.exists(C_RACE), reason="audit C race record not on this machine")
def test_race_catch_up_on_real_record():
    R = S.score_record(C_RACE)
    rows = R["race"]
    slow = [r for r in rows if abs(r["c_pace"] - 0.7) < 1e-3 and r["c_start"] == 0]
    assert len(slow) == 16 and all(r["caught"] == 1.0 for r in slow)          # a 0.7x copy half a lap ahead is always caught
    assert all(r["t_catch"] > 5.0 for r in slow)
    # audit C: win rate 39 % over the 128 MAP matches
    assert abs(np.mean([r["win"] for r in rows]) - 0.39) < 0.01
    for r in rows:
        if r["caught"]:
            assert r["car_after"] <= r["car_onsets"] and r["behind_after_s"] <= r["behind_s"] + 1e-9
