"""(11-15) MAP이 plan을 얼마나 늦게/바깥으로 따라가나: 시험 기록(st = 매 0.1 s 차 상태 x, y, yaw, v; pl = 그때 낸 plan의 10점 world 좌표,
0.1 s 간격)에서 plan 점 k(0.1(k+1) s 뒤)와 실제 그 시각 위치·속도를 비교. 감속 구간 / 벽 충돌 1 s 전으로 나눠 봄.
usage: map_lag.py rec.npz [rec2.npz ...]"""
import sys
import numpy as np

for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=True)
    st, pl, rs = z["st"], z["pl"], z["rs"]
    cols = list(z["cols"]); cnt = z["cnt"]
    hw = cnt[:, :, cols.index("hard_wall")]                     # cumulative per car -> new hard wall hits per decision
    wall = np.diff(np.concatenate([hw[:1] * 0, hw], 0), axis=0) > 0
    T, B = pl.shape[:2]
    K = 4                                                     # plan points 0..3 = 0.1..0.4 s ahead
    rows = []
    for t in range(T - K - 1):
        ok = ~rs[t:t + K + 1].any(0)
        p_prev = np.concatenate([st[t, :, None, :2], pl[t, :, :K]], 1)          # (B, K+1, 2)
        seg = p_prev[:, 1:] - p_prev[:, :-1]
        vpl = np.linalg.norm(seg, axis=-1) / 0.1                                # plan speed per segment
        tang = seg / np.maximum(np.linalg.norm(seg, axis=-1, keepdims=True), 1e-6)
        act = st[t + 1:t + K + 1, :, :2].transpose(1, 0, 2)                     # (B, K, 2)
        d = act - pl[t, :, :K]
        e_long = (d * tang).sum(-1)
        e_crs = tang[..., 0] * d[..., 1] - tang[..., 1] * d[..., 0]             # + = left of the plan
        h = np.arctan2(seg[..., 1], seg[..., 0])
        turn = np.angle(np.exp(1j * (h[:, -1] - h[:, 0])))                      # + = plan turns left
        out = -np.sign(turn)[:, None] * e_crs                                    # + = outside of the turn
        vact = st[t + 1:t + K + 1, :, 3].T
        dec = (vpl[:, 2] - vpl[:, 0]) / 0.2
        hit = wall[t:t + 10].sum(0) > 0
        arc = np.linalg.norm(seg, axis=-1).sum(1)
        alat = vpl.mean(1) ** 2 * np.abs(turn) / np.maximum(arc, 0.05)            # the plan's own lateral acceleration
        for b in np.nonzero(ok)[0]:
            rows.append((dec[b], abs(turn[b]), hit[b], vpl[b, 0], vact[b, 2] - vpl[b, 2], e_long[b, 3], out[b, 3], alat[b]))
    a = np.array(rows)
    print(f"\n== {fn.split('/')[-1]}  ({len(a)} decisions)")
    # each new hard wall hit: was the car outside (+) or inside (-) of the turn its last plan made, 0.4 s after that plan
    print(f"{'구간':<26}{'수':>7}{'속도차 0.3s (실제-plan)':>22}{'앞뒤 0.4s':>11}{'바깥 0.4s':>11}")
    def row(name, m):
        if m.sum() == 0:
            return
        p10, p90 = np.percentile(a[m, 6], [10, 90]) * 100
        print(f"{name:<26}{int(m.sum()):>7}{np.median(a[m, 4]):>+20.2f} m/s{np.median(a[m, 5]):>+9.2f} m{np.median(a[m, 6]) * 100:>+8.1f} cm"
              f"  (10-90 %: {p10:+.1f} ~ {p90:+.1f} cm)")
    fast = a[:, 3] > 4.0
    row("전체", np.ones(len(a), bool))
    row("감속 >3 m/s^2 (plan)", (a[:, 0] < -3) & fast)
    row("감속 >3 + 회전 >10도", (a[:, 0] < -3) & (a[:, 1] > np.radians(10)) & fast)
    row("회전 >10도, 감속 없음", (a[:, 0] > -1) & (a[:, 1] > np.radians(10)) & fast)
    row("벽 충돌 1 s 전", a[:, 2] > 0)
    row("벽 충돌 1 s 전 + 감속>3", (a[:, 2] > 0) & (a[:, 0] < -3))
    for lo, hi in ((1, 3), (3, 5), (5, 7), (7, 9), (9, 30)):
        row(f"plan 옆가속 {lo}-{hi} m/s^2", (a[:, 7] >= lo) & (a[:, 7] < hi) & fast)
    for lo, hi in ((4, 5.5), (5.5, 7), (7, 10)):
        row(f"회전>10도, plan {lo}-{hi} m/s", (a[:, 1] > np.radians(10)) & (a[:, 3] >= lo) & (a[:, 3] < hi))
