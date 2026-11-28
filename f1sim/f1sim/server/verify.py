"""결과 화면 숫자의 독립 검산: 같은 세션의 제어 주기 기록(`<id>.steps.npz`)만으로 다시 채점해 결과 json과 대조한다.

서버의 채점(`rules.QualiScorer`)은 사건 방식이고, 바퀴는 계시선(선분)을 차 위치가 지나는 순간으로 잰다. 여기서는 그 코드를
쓰지 않고, 기록 배열에 **대회 형식 벤치마크(`f1sim.compbench.score`)의 함수**를 그대로 적용한다(다른 작업자의 모듈, import만):
  - 남은 거리 rem = 계시선까지의 최단 거리 장(벽에서 0.5 m 떨어진 길, 학습 보상의 `plan_reward._field_one`과 같은 계산,
    5 cm 칸)에서 읽은 값  ->  `progress`, `first_crossing`, `laps_from_progress`(바퀴와 바퀴 시간)
  - 누적 접촉 수(벽·상자·차, 그중 세게)  ->  `new_counts`(주기마다 새 접촉)  ->  바퀴마다 합  ->  `streak`
compbench는 첫 통과 시각을 기록 간격(25 ms)으로 자르고 그 위치를 기준선으로 삼으므로 바퀴 시간이 몇 ms 다를 수 있다.

사용:  python -m f1sim.server.verify /data/tenth/results/<id>.json
"""
from __future__ import annotations

import json
import sys

import numpy as np


def distance_field(track_name: str, gate0) -> tuple:
    """(필드 D[H, W] m, 원점, 칸 크기, 한 바퀴 거리): 계시선까지 차로 안으로 잰 최단 거리, 5 cm 칸.
    계시선 = gate0(x, y, yaw, 왼쪽 폭, 오른쪽 폭)의 선분. 선분 뒤쪽 칸이 0에서 시작해 거꾸로 한 바퀴 돌아 선분 앞쪽에 닿으면
    한 바퀴 거리가 된다(학습 보상의 `plan_reward._field_one`과 같은 생각, 장벽만 선분으로 한정 — 무한 직선은 접힌 코스의
    옆 차로까지 잘랐다). 이웃 그래프는 `plan_reward._grid_graph`를 그대로 쓴다."""
    from scipy import ndimage
    from scipy.sparse.csgraph import dijkstra
    from .. import maps
    from ..learn.plan_reward import _grid_graph
    tr = maps.load(maps.smooth_name(str(track_name)))
    occ = np.asarray(tr.occupancy, bool)
    f = max(1, int(round(0.05 / float(tr.resolution))))
    H, W = occ.shape[0] // f, occ.shape[1] // f
    free = ~occ[:H * f, :W * f].reshape(H, f, W, f).any((1, 3))
    res = float(tr.resolution) * f
    origin = (float(tr.origin[0]) + 0.5 * (res - float(tr.resolution)), float(tr.origin[1]) + 0.5 * (res - float(tr.resolution)))
    x0, y0, yaw = float(gate0[0]), float(gate0[1]), float(gate0[2])
    wl, wr = (float(gate0[3]), float(gate0[4])) if len(gate0) >= 5 else (3.0, 3.0)
    ii, jj = np.mgrid[0:H, 0:W]
    X = origin[0] + jj * res
    Y = origin[1] + ii * res
    t = np.array([np.cos(yaw), np.sin(yaw)])
    A = (X - x0) * t[0] + (Y - y0) * t[1]
    lat = -(X - x0) * t[1] + (Y - y0) * t[0]
    i0, j0 = int(round((y0 - origin[1]) / res)), int(round((x0 - origin[0]) / res))
    lab, _ = ndimage.label(free)
    free &= lab == lab[i0, j0]
    within = (lat >= -wr - 0.1) & (lat <= wl + 0.1)
    barrier = free & (np.abs(A) < res) & within
    src = free & ~barrier & ndimage.binary_dilation(barrier, iterations=2) & (A < 0) & within
    node = free & ~barrier
    G, idx = _grid_graph(node, res)
    d = dijkstra(G, directed=False, indices=idx[src], min_only=True)
    D = np.full((H, W), np.nan, np.float32)
    D[node] = d
    D[~np.isfinite(D)] = np.nan
    D[barrier] = 0.0
    return D, origin, res, float(np.nanmax(D))


def rescore(steps_npz: str, breaks_on: str = "any") -> dict:
    from ..compbench import score as CB
    z = np.load(steps_npz)
    dt = float(z["dt"])
    D, origin, res, Lf = distance_field(str(z["track"]), z["gate0"])
    x, y = z["x"].astype(float), z["y"].astype(float)
    j = np.floor((x - origin[0]) / res + 0.5).astype(int); i = np.floor((y - origin[1]) / res + 0.5).astype(int)
    ok = (i >= 0) & (i < D.shape[0]) & (j >= 0) & (j < D.shape[1])
    rem = np.full(len(x), np.nan); rem[ok] = D[i[ok], j[ok]]
    rem = rem[:, None]                                       # (T+1, 1) 계시선까지 남은 거리
    L = Lf
    P = CB.progress(rem, L)
    k0 = CB.first_crossing(rem[:, 0], L)
    laps, lt = CB.laps_from_progress(k0, P[:, 0], L, dt)
    cnt = z["counts"].astype(np.float32)[1:, None, :]        # (T, 1, 6) 누적: 시작 상태(0번) 다음부터
    rs = np.zeros(cnt.shape[:2], bool)
    d = CB.new_counts(cnt, rs)                               # (T, 1, 6) 주기마다 새 접촉: 벽·상자·차, 세게 벽·상자·차
    # 기록 배열의 시각 t[k]는 k번째 상태; 주기 k(상태 k -> k+1)의 접촉은 d[k]
    any_l = [float(d[a:b, 0, 0:2].sum()) for a, b in laps]
    hard_l = [float(d[a:b, 0, 3:5].sum()) for a, b in laps]
    bad = hard_l if breaks_on == "hard" else any_l
    return {"n_laps": len(lt), "lap_times": [float(x) for x in lt], "best_lap": (min(lt) if lt else None),
            "best_clean_streak": int(CB.streak(bad)), "contacts_per_lap": any_l, "hard_per_lap": hard_l,
            "contacts_total": float(d[:, 0, 0:2].sum()), "hard_total": float(d[:, 0, 3:5].sum()),
            "first_crossing_t": (float(k0 * dt) if k0 is not None else None)}


def compare(result_json: str) -> dict:
    """서버 결과와 독립 계산을 대조한다. 판정 규칙(두 방법의 정의 차이만 허용):
    - 바퀴 수: 같아야 한다. 단 서버의 마지막 바퀴가 세션 끝 0.1 s 안에 끝났으면 독립 쪽이 하나 적어도 된다
      (compbench는 첫 통과를 25 ms 기록 간격에 맞춰 늦게 잡고 그만큼 선을 앞으로 옮긴다).
    - 바퀴 시간: 첫 바퀴(서 있다 출발: 선 근처 속도가 느려 위 차이가 커짐)는 0.1 s, 그 뒤는 0.03 s 안.
      계시선(선분)과 거리 장의 같은 거리 선은 모양이 조금 달라 차가 선을 지나는 옆 위치마다 수 ms~20 ms 다르다.
    - 무충돌 연속 바퀴: 두 쪽이 공통으로 센 바퀴들에서 같아야 한다. 바퀴마다 접촉 수도 같아야 한다."""
    from .rules import streak as _streak
    res = json.load(open(result_json, encoding="utf-8"))
    steps = result_json[:-5] + ".steps.npz"
    ind = rescore(steps, res.get("breaks_on", "any"))
    srv_laps = [r["time"] for r in res["laps"]]
    n = min(len(srv_laps), len(ind["lap_times"]))
    diffs = [abs(srv_laps[i] - ind["lap_times"][i]) for i in range(n)]
    srv_contacts = res.get("contacts_counted", 0)
    end_t = float(res.get("t_end", 0.0))
    last_end = res["laps"][-1]["t_end"] if res["laps"] else None
    edge_ok = (len(srv_laps) == len(ind["lap_times"]) or
               (len(srv_laps) == len(ind["lap_times"]) + 1 and last_end is not None and end_t - last_end < 0.1))
    srv_clean = [0.0 if r["clean"] else 1.0 for r in res["laps"]][:n]
    ind_bad = (ind["hard_per_lap"] if res.get("breaks_on") == "hard" else ind["contacts_per_lap"])[:n]
    srv_cpl = [r["contacts"] for r in res["laps"]][:n]
    out = {
        "server": {"best_clean_streak": res["score"]["best_clean_streak"], "best_lap": res["score"]["best_lap"],
                   "n_laps": res["n_laps"], "contacts": srv_contacts, "lap_times": srv_laps,
                   "contacts_per_lap": [r["contacts"] for r in res["laps"]]},
        "independent": ind,
        "lap_time_max_abs_diff_s": max(diffs) if diffs else None,
        "lap1_abs_diff_s": diffs[0] if diffs else None,
        "later_laps_max_abs_diff_s": max(diffs[1:]) if len(diffs) > 1 else None,
        "same_streak": res["score"]["best_clean_streak"] == ind["best_clean_streak"],
        "same_streak_common_laps": _streak(srv_clean) == _streak([float(x > 0) for x in ind_bad]),
        "same_contacts_per_lap": [int(a) for a in srv_cpl] == [int(round(b)) for b in ind["contacts_per_lap"][:n]],
        "same_n_laps": len(srv_laps) == len(ind["lap_times"]),
        "n_laps_ok": bool(edge_ok),
        "best_lap_diff_s": (abs(res["score"]["best_lap"] - ind["best_lap"])
                            if res["score"]["best_lap"] is not None and ind["best_lap"] is not None else None),
    }
    out["agree"] = bool(out["n_laps_ok"] and out["same_streak_common_laps"] and out["same_contacts_per_lap"]
                        and (out["lap1_abs_diff_s"] or 0.0) < 0.1 and (out["later_laps_max_abs_diff_s"] or 0.0) < 0.03)
    return out


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1]), ensure_ascii=False, indent=1))
