"""M1 서버 시험: 자료(YAML) 검사, 예선 채점, 상자 배치, 계시선, 프로토콜, 그리고 CPU 세션 한 번(기준선) + 독립 검산.
실행: python3 game/tools/tenth.py test  (또는 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q f1sim/f1sim/server/tests)
"""
import json
import math
import os

import numpy as np
import pytest

from f1sim.server import SCHEMA_VERSION, protocol as P
from f1sim.server import library as LIB
from f1sim.server.rules import QualiScorer, rank, streak

RULES = {"clean_lap": {"breaks_on": "any", "kinds": ["wall", "box"], "hard_mps": 1.0}}


# ------------------------------------------------------------------ 자료
def test_library_loads_and_catalog_is_human():
    lib = LIB.Library()
    cat = lib.catalog()
    assert {c["id"] for c in cat["courses"]} >= {"kr26-hall"}
    assert {d["id"] for d in cat["drivers"]} >= {"scan31b-u475", "hs43b-u900", "ftg"}
    # 이름 규칙(11-29): 지어낸 이름 금지 -- 팀이 쓰는 식별자, 대회 이름
    assert {d["name"] for d in cat["drivers"]} >= {"scan31b u475", "hs43b u900", "Follow-the-Gap"}
    assert next(c for c in cat["courses"] if c["id"] == "kr26-hall")["name"] == "IFAC 2026"
    assert next(c for c in cat["courses"] if c["id"] == "kr25-iccas")["name"] == "ICCAS 2025"
    assert any(r["id"] == "kr-event-quali" and r["default"] for r in cat["rules"])
    assert any(p["default"] and p["detail"]["profile"] == "default_comp_car_pre_place" for p in cat["physics"])
    assert any(p["name"] == "params.py 기본값" for p in cat["physics"])
    for d in cat["drivers"]:
        assert "/" not in d["name"] and "~" not in d["name"]           # 화면 이름에 경로 없음
    a = next(d for d in cat["drivers"] if d["id"] == "scan31b-u475")
    assert a["detail"]["run"] == "scan31b" and a["detail"]["update"] == 475
    # 1대1 규칙은 자료에만(M6)
    h2h = next(r for r in cat["rules"] if r["id"] == "kr-event-h2h")
    assert not h2h["available"] and h2h["start"]["grid"] == "opposite"


def test_quantity_validation_rejects_bare_numbers():
    assert LIB.check_quantity(0.3, "x")                                   # 맨 숫자 거부
    assert LIB.check_quantity({"v": 0.3, "src": "guess"}, "x")            # 모르는 출처
    assert LIB.check_quantity({"v": 0.3, "src": "estimated"}, "x")        # 추정인데 오차 없음
    assert not LIB.check_quantity({"v": 0.3, "u": "m", "src": "estimated", "err": 0.01}, "x")
    with pytest.raises(LIB.LibraryError):
        LIB.validate("course", {"schema": "tenth.course/1", "id": "x", "name": "x", "track": "t", "layouts": {},
                                "duct_diameter": 0.3}, "bad.course.yaml")


def test_profile_summary_counts_sources():
    s = LIB.profile_summary("default_comp_car_pre_place")
    assert s["n"] > 20 and sum(s["counts"].values()) == s["n"]
    assert s["counts"].get("lower_bound", 0) >= 1                         # 마찰은 하한만
    assert LIB.profile_summary(None)["n"] == 0


# ------------------------------------------------------------------ 채점
def test_scorer_streak_and_best_lap():
    sc = QualiScorer(RULES, 46.0, [15.0, 30.0])
    assert sc.on_cross(1.0, True) is None                                 # 첫 통과: 계시 시작
    for k, (t, contact) in enumerate([(9.0, False), (17.5, True), (25.0, False), (32.4, False), (40.0, False)]):
        if contact:
            sc.on_contact(t - 2.0, 1, 0.4, 0.0, 0.0, 10.0)
        lap = sc.on_cross(t, k < 2)
        assert lap.n == k + 1
    r = sc.result()
    times = [x["time"] for x in r["laps"]]
    assert np.allclose(times, [8.0, 8.5, 7.5, 7.4, 7.6])
    assert [x["clean"] for x in r["laps"]] == [True, False, True, True, True]
    assert r["score"]["best_clean_streak"] == 3 and r["streak_check"] == 3
    assert abs(r["score"]["best_lap"] - 7.4) < 1e-9
    # 상자 여부는 바퀴를 시작할 때 기준: 1~3바퀴는 상자 있음(세 번째 통과 때 False가 넘어가 4바퀴부터 없음)
    assert abs(r["best_lap_boxes"] - 7.5) < 1e-9 and abs(r["best_lap_no_boxes"] - 7.4) < 1e-9
    assert r["contacts_counted"] == 1 and r["contacts_hard"] == 0
    assert r["contacts"][0]["kind_label"] == "벽" and r["contacts"][0]["severity_label"] == "약하게 닿음"


def test_scorer_hard_only_rule_and_car_not_counted():
    sc = QualiScorer({"clean_lap": {"breaks_on": "hard", "kinds": ["wall", "box"], "hard_mps": 1.0}}, 40.0, [])
    sc.on_cross(0.0, True)
    sc.on_contact(1.0, 2, 0.3, 0, 0, 1.0)          # 상자 약하게: 무충돌로 침(hard 규칙)
    sc.on_contact(2.0, 3, 2.0, 0, 0, 1.0)          # 차: 규칙이 세지 않음
    sc.on_cross(8.0, True)
    sc.on_contact(9.0, 1, 1.5, 0, 0, 1.0)          # 벽 세게
    sc.on_cross(16.0, True)
    r = sc.result()
    assert [x["clean"] for x in r["laps"]] == [True, False]
    assert r["score"]["best_clean_streak"] == 1
    assert sum(1 for c in r["contacts"] if not c["counted"]) == 1


def test_streak_matches_compbench_and_rank_ties():
    for bad in ([0, 0, 1, 0, 0, 0, 1], [1, 1], [], [0] * 5):
        assert streak(bad) == max((len(s) for s in "".join("x" if b else "." for b in bad).split("x")), default=0)
    res = [{"score": {"best_clean_streak": 3, "best_lap": 7.6}}, {"score": {"best_clean_streak": 5, "best_lap": 8.0}},
           {"score": {"best_clean_streak": 3, "best_lap": 7.6}}, {"score": {"best_clean_streak": 3, "best_lap": None}}]
    sc = [{"key": "best_clean_streak", "order": "desc"}, {"key": "best_lap", "order": "asc"}]
    assert rank(res, sc) == [1, 0, 1, 3]


# ------------------------------------------------------------------ 계시선
def test_gate_crossing_forward_backward_outside():
    from f1sim.server.quali import Gate
    g = Gate(0.0, 0.0, 0.0, 1.0, 1.0)          # x 방향으로 달림, 선분은 y = -1..1
    d, f = g.cross((-0.1, 0.2), (0.3, 0.2))
    assert d == 1 and abs(f - 0.25) < 1e-9
    assert g.cross((0.3, 0.2), (-0.1, 0.2))[0] == -1
    assert g.cross((-0.1, 3.0), (0.3, 3.0)) is None       # 선분 밖(옆 차로)
    assert g.cross((0.1, 0.0), (0.3, 0.0)) is None        # 지나지 않음


# ------------------------------------------------------------------ 상자
def test_box_catalogue_and_saved_layout():
    from f1sim.server import layouts as LY
    cat = LY.catalogue()
    assert all(c["style"] == "cardboard_box" for c in cat if c["real"])
    assert {32, 38, 39} <= {c["sid"] for c in cat if c["real"]}
    bank = "/data/compbench/banks/exam3_k26scan"
    if not os.path.isdir(bank):
        pytest.skip("시험 세트 은행이 이 PC에 없음")
    assert LY.bank_count(bank) == 64
    p = LY.bank_layout(bank, 0, 3)
    assert len(p) == 3 and all(32 <= x["sid"] <= 39 for x in p)
    d = LY.describe(p)
    assert d[0]["label"].startswith("골판지 상자") and d[0]["w"] > 0.15


def test_random_layout_is_seeded():
    from f1sim import maps
    from f1sim.server import layouts as LY
    tr = maps.load(maps.smooth_name("real:map12x16+duct"))
    a = LY.random_layout(tr, "centerline", 3, 7)
    b = LY.random_layout(tr, "centerline", 3, 7)
    c = LY.random_layout(tr, "centerline", 3, 8)
    assert a == b and a != c and len(a) == 3
    dist = LY.free_distance(tr)
    for p in a:
        assert dist(p["x"], p["y"]) > 0.25


# ------------------------------------------------------------------ 프로토콜
def test_check_client_m1_messages():
    ok = lambda m: P.check_client(dict(m, v=SCHEMA_VERSION))
    assert ok({"k": "req", "op": "catalog"}) is None
    assert ok({"k": "req", "op": "preview", "course": "kr26-hall"}) is None
    assert ok({"k": "session", "op": "start", "spec": {"driver": "ftg"}}) is None
    assert ok({"k": "cmd", "op": "speed", "x": 2.0}) is None
    assert ok({"k": "cmd", "op": "end"}) is None
    assert "요청" in ok({"k": "req", "op": "fly"})
    assert ok({"k": "session", "op": "start", "spec": [1]}) is not None


# ------------------------------------------------------------------ 세션 한 번(CPU, 기준선) + 독립 검산
def test_cpu_session_baseline_and_independent_rescore(tmp_path, monkeypatch):
    if not os.path.isfile(os.path.expanduser("~/f1sim_runs/pick_scan31b_u475/state_latest.pt")):
        pytest.skip("드라이버 저장 파일이 없음")
    from f1sim.server import quali as Q
    from f1sim.server import verify as V
    monkeypatch.setattr(Q, "RESULTS", str(tmp_path))
    q = Q.QualiSession(LIB.Library(), {"course": "kr26-hall", "driver": "ftg", "duration_s": 17.0, "remove_at_s": 5.0,
                                       "boxes": {"source": "saved", "index": 2, "count": 3}}, device="cpu")
    assert q.engine.facts()["walls"] == "smooth"                          # 매끄러운 판(1.25 cm 격자)
    chk = q.check
    assert chk["checked"] > 10 and chk["same"] == chk["checked"] and not chk["differs"]   # 물리 프로필이 실제로 들어감
    tm = q.track_message()
    assert tm["hoses"] and all(len(h["xy"]) == len(h["r"]) for h in tm["hoses"])
    removed = False
    while not q.done:
        for ev in q.step():
            removed |= ev["type"] == "boxes_removed"
    assert removed and not q.engine.boxes_on and q.engine.box_poses() == []
    res = q.result()
    paths = q.save(res)
    assert res["n_laps"] >= 1 and res["score"]["best_lap"] is not None
    cmp_ = V.compare(paths["json"])
    assert cmp_["agree"], cmp_
    q.close()
