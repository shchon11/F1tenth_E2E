"""speed_gap.py 결과 두 개(예: 학습 조건 / 새 제어기)를 바퀴 구간별로 비교: 구간마다 한 바퀴에 머문 시간, 중앙 속도, 옆 가속이 타이어 한계의 몇 %인지.
인자: a.json b.json [구간 수]"""
import sys, json
import numpy as np
A, Bj = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
name = {"ideal": "학습 조건", "track": "새 제어기"}
na, nb_ = name.get(A["mode"], A["mode"]), name.get(Bj["mode"], Bj["mode"])
ka = np.array_split(np.arange(len(A["bins"])), n)
print(f"바퀴 {A['lap']:.1f} m | 한 바퀴: {na} {A['lap_time']:.2f} s, {nb_} {Bj['lap_time']:.2f} s")
print(f"{'구간 [m]':>12} | {na} 시간 / 속도 / 타이어 사용 | {nb_} 시간 / 속도 / 타이어 사용 | 시간 차")
for idx in ka:
    def agg(J):
        b = [J["bins"][i] for i in idx]
        t = sum(x["t"] for x in b)
        vs = [x["v"] for x in b if np.isfinite(x["v"])]
        us = [x["alat"] / x["alim"] for x in b if np.isfinite(x["alim"]) and x["alim"] > 0]
        v = float(np.median(vs)) if vs else float("nan")
        u = float(np.max(us)) if us else float("nan")
        return t, v, u
    ta, va, ua = agg(A); tb, vb, ub = agg(Bj)
    s0, s1 = A["bins"][idx[0]]["s0"], A["bins"][idx[-1]]["s0"] + 0.5
    print(f"{s0:5.1f}-{s1:5.1f} | {ta:5.2f} s {va:4.1f} m/s {ua * 100:4.0f} % | {tb:5.2f} s {vb:4.1f} m/s {ub * 100:4.0f} % | {tb - ta:+.2f} s")
