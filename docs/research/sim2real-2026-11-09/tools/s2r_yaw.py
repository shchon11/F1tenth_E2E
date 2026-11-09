"""조향 명령 -> 실제 회전 (11-09). 대회장 bag들에서 자이로로 잰 곡률 kappa = r / v 를 지연 lag만큼 앞선 조향 명령과 비교.
1) 지연: L kappa(t) 와 명령 d(t - lag) 의 상관이 가장 큰 lag.
2) 거의 정상 상태인 순간(지난 0.3 s 동안 명령·속도·회전이 거의 안 변함, v > 1.5 m/s)만 골라
   L kappa = g d + b - K a_y   (a_y = v r, 구심 가속)
   를 맞춤: g = 명령 대비 실제 바퀴 각(서보·링크 이득), b = 직진 명령에서의 실제 각(트림), K [rad/(m/s^2)] = 언더스티어 기울기.
   옆 가속이 0에 가까우면 타이어 미끄럼각이 0이라 L kappa = 실제 바퀴 각 -> g, b는 낮은 a_y 자료가 정하고 K는 a_y 폭이 정함.
3) a_y 구간별 L kappa / d 와 적합값 비교, 잔차.
인자: npz 파일들(같은 노면만)"""
import sys
import numpy as np
import s2r_signals as G

L = 0.3302
rows = []
lags = []
for f in sys.argv[1:]:
    Z = G.load(f)
    S = G.signals(Z)
    v, r, d = S["v_wheel"], S["gz"], S["steer_cmd"]
    mv = v > 1.5
    kap = np.where(mv, r / np.maximum(v, 0.1), 0.0)
    best = None
    for lag in range(0, 31):                                  # 0..0.30 s
        dd = np.roll(d, lag)
        c = np.corrcoef(L * kap[mv][40:], dd[mv][40:])[0, 1]
        if best is None or c > best[1]:
            best = (lag, c)
    lags.append(best)
    lag = best[0]
    dl = np.roll(d, lag)
    # quasi steady: over the last 0.3 s the delayed command, the speed and the yaw rate barely moved
    w = 30
    def span(x):
        from numpy.lib.stride_tricks import sliding_window_view as sw
        a = sw(x, w)
        out = np.full_like(x, np.inf); out[w - 1:] = a.max(1) - a.min(1)
        return out
    qs = mv & (span(dl) < 0.03) & (span(v) < 0.4) & (span(r) < 0.25)
    qs[:200] = False
    ay = v * r
    for i in np.nonzero(qs)[0][::5]:                           # every 50 ms
        rows.append((dl[i], L * kap[i], ay[i], v[i], S["ay"][i], f))
R = np.array([x[:5] for x in rows], dtype=float)
print("지연(상관 최대) bag별:", ", ".join(f"{l * 10:.0f} ms (r {c:.3f})" for l, c in lags))
d_, lk, ay, v, ay_imu = R.T
print(f"정상 상태 표본 {len(R)}개 | 속도 {np.quantile(v, .1):.1f}-{np.quantile(v, .9):.1f} m/s | |a_y| 중앙 {np.median(np.abs(ay)):.1f}, p90 {np.quantile(np.abs(ay), .9):.1f} m/s^2")
print(f"구심 a_y(v r) 대 IMU a_y: 기울기 {np.sum(ay * ay_imu) / np.sum(ay * ay):.3f}, 상관 {np.corrcoef(ay, ay_imu)[0, 1]:.3f}")
# the fit; signs: K a_y with a_y signed (left turn +), understeer reduces |kappa|
A = np.vstack([d_, np.ones_like(d_), -ay]).T
for lo, hi, nm in ((0, 99, "전체"), (0, 4, "|a_y| < 4"), (0, 2.5, "|a_y| < 2.5")):
    m = (np.abs(ay) >= lo) & (np.abs(ay) < hi)
    if m.sum() < 50:
        continue
    c, *_ = np.linalg.lstsq(A[m], lk[m], rcond=None)
    res = lk[m] - A[m] @ c
    print(f"[{nm}] n {m.sum()}: L kappa = {c[0]:.3f} d {c[1]:+.4f} - {c[2]:.4f} a_y  | 잔차 중앙 {np.median(np.abs(res)):.4f} rad")
c, *_ = np.linalg.lstsq(A, lk, rcond=None)
print("a_y 구간별 (같은 부호 조향만, |d| > 0.03): 실제 L kappa / 명령 d 중앙, 적합 예측")
for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 11)):
    m = (np.abs(ay) >= lo) & (np.abs(ay) < hi) & (np.abs(d_) > 0.03) & (np.sign(d_) == np.sign(lk))
    if m.sum() < 20:
        continue
    pred = A[m] @ c
    print(f"  |a_y| {lo}-{hi}: n {m.sum():5d}, 속도 중앙 {np.median(v[m]):.1f}, L kappa / d = {np.median(lk[m] / d_[m]):.3f} (적합 {np.median(pred / d_[m]):.3f})")
# with the trim split by turn direction (an off-centre trim shows as left/right asymmetry)
for sg, nm in ((1, "왼쪽(+)"), (-1, "오른쪽(-)")):
    m = (np.sign(d_) == sg) & (np.abs(d_) > 0.03) & (np.abs(ay) < 3)
    if m.sum() > 30:
        cc, *_ = np.linalg.lstsq(np.vstack([d_[m], np.ones(m.sum())]).T, lk[m], rcond=None)
        print(f"  {nm} 조향, |a_y|<3: L kappa = {cc[0]:.3f} d {cc[1]:+.4f}  (n {m.sum()})")
np.save("/data/sim2real_1109/yaw_rows.npy", R)
