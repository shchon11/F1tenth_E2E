"""어느 지도에서 학습한 checkpoint인지 (11-14, 사용자: "이런 실험들을 맵만 싹 바꿔서 할 수 있게"). 평가·시험·영상 도구가
plan_rl.WORLD(Korea 2025) 대신 checkpoint가 학습한 세계 json을 읽게 함. 옛 checkpoint(인자에 world 없음)는 WORLD 그대로.
시험 상자 bank: Korea 2025는 지금까지의 prop_banks_exam1 그대로(기록이 바뀌지 않게), 다른 지도는 prop_banks_exam1_<지도>."""
import os
import json

EXAM_BANKS = "/home/shchon11/f1sim_runs/prop_banks_exam1"


def world_path(ck) -> str:
    from f1sim.learn.plan_rl import WORLD
    w = (ck.get("args") or {}).get("world") or WORLD
    return w if os.path.exists(w) else WORLD


def map_name(world: str) -> str:
    """'real:korea_2025_iccas' -> 'korea_2025_iccas' (the world's first track)"""
    t = str(json.load(open(world)).get("tracks", ""))
    return t.split(",")[0].split(":")[-1]


def exam_bank(world: str) -> str:
    m = map_name(world)
    return EXAM_BANKS if m == "korea_2025_iccas" else f"{EXAM_BANKS}_{m}"
