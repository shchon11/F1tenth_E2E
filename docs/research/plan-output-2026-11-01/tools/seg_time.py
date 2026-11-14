"""(11-14) B가 세게 닿은 구간마다(진행 거리 기준, 닿기 1.5 s 전 ~ 1.5 s 뒤): 닿은 그 통과에 B가 쓴 시간 vs 같은 구간을 A가 지난 시간(같은 세션의 모든 통과 중앙).
진행 거리 = 결승선까지 남은 최단 거리(벽 사이 빈 공간으로; rem). 인자: A.npz B.npz 이름A 이름B"""
import sys
import numpy as np
from rec_lib import Rec
A, B = Rec(sys.argv[1]), Rec(sys.argv[2]); nA, nB = sys.argv[3], sys.argv[4]
T = min(A.T, B.T); dt = B.dt; W = int(round(1.5 / dt))
remA, remB = A.z["rem"][:T].astype(float), B.z["rem"][:T].astype(float)
L = float(A.z["lapL"][0])


def passes(rem, r_in, r_out, start=None):
    out = []
    rng = [start] if start is not None else range(1, len(rem))
    for i in rng:
        if not (rem[i - 1] > r_in >= rem[i]) and start is None:
            continue
        for j in range(i, min(len(rem), i + int(8 / dt))):
            if j > i and (rem[j - 1] - rem[j] > 0.5 * L or rem[j] - rem[j - 1] > 0.5 * L):
                break
            if rem[j] <= r_out:
                out.append((j - i) * dt); break
    return out


for k, kn in enumerate(("벽", "상자", "차")):
    dB, dA = [], []
    for t, b in zip(*np.nonzero(B.d[:T, :, 3 + k] > 0)):
        if t - W < 1 or t + W >= T:
            continue
        r_in, r_out = remB[t - W, b], remB[t + W, b]
        if not (r_in > r_out and r_in - r_out < 0.5 * L):
            continue
        tb = passes(remB[:, b], r_in, r_out, start=t - W)
        ta = passes(remA[:, b], r_in, r_out)
        if tb and ta:
            dB.append(tb[0]); dA.append(np.median(ta))
    dB, dA = np.array(dB), np.array(dA); d = dB - dA
    print(f"[{kn}] {nB}이 세게 닿은 구간 {len(d)}개(진행 거리 중앙 {np.median([0]) if False else ''}): 닿은 그 통과 {nB} 중앙 {np.median(dB):.2f} s · 같은 구간 {nA} 중앙 {np.median(dA):.2f} s"
          f" | 차 평균 {d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(len(d)):.2f} s, {nB}이 더 빠른 구간 {int((d < 0).sum())}/{len(d)}")
