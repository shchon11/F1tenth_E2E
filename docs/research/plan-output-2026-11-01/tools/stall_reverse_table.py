"""(11-16) 앞이 막혀 선 차를 복제했을 때 '그대로'(A) vs '조향 0으로 이어서 후진'(V)이 어떻게 갈리나 (box_replay.py 갈래 V 기록).

복제 순간(주기마다) 가운데 원본이 그 전 1 s 동안 내내 서 있던(|v| < 0.3 m/s, 다시 출발 없음) 것만 골라, 같은 묶음의 A 복제와 V 복제를
주기가 끝날 때까지 비교: 끝에 앞으로 1 m/s 넘게 달리고 있나, 끝에도 서 있나, 처음으로 앞으로 1 m/s를 넘은 때, 시작 자리에서 얼마나
벗어났나, 그동안 세게 닿음. 인자: 기록npz [...]
"""
import sys

import numpy as np

STILL_V, PRE = 0.3, 10        # stalled = |v| < 0.3 m/s over the 10 decisions (1 s) before the copy

for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=True)
    st, rs, inc = z["st"].astype(float), z["rs"].astype(bool), z["inc"].astype(float)
    arm, grp, PER = z["arm"].astype(int), z["grp"].astype(int), int(z["period"])
    dt = float(z["dt"]); T = rs.shape[0]
    K = int((grp == 0).sum())
    rev = f"{float(z['rev_v']):g} m/s x {int(z['rev_n'])}결정" if "rev_v" in z.files else "?"
    rows = {"A": [], "V": []}
    n_sync = 0
    for kd in z["sync_t"]:
        kd = int(kd)
        if kd < PRE or kd + PER > T:
            continue
        for g in range(int(grp.max()) + 1):
            src = g * K
            v_pre = st[kd - PRE:kd + 1, src, 3]
            if not (np.abs(v_pre) < STILL_V).all() or rs[kd - PRE:kd, src].any():
                continue
            n_sync += 1
            for name, a in (("A", 1), ("V", 7)):
                for r in np.nonzero((grp == g) & (arm == a))[0]:
                    if rs[kd:kd + PER, r].any():
                        continue
                    # st[kd + PER] is taken after the next copy (the replicate is the source again): the window ends one before
                    v = st[kd:kd + PER, r, 3]
                    fwd = np.nonzero(v > 1.0)[0]
                    yaw0 = st[kd, r, 2]
                    dp = st[kd + PER - 1, r, :2] - st[kd, r, :2]
                    d = dp[0] * np.cos(yaw0) + dp[1] * np.sin(yaw0)        # ahead of where it stood, along its heading
                    rows[name].append((v[-1] > 1.0, abs(v[-1]) < STILL_V, fwd[0] * dt if fwd.size else np.inf, d,
                                       inc[kd:kd + PER, r, 3:6].sum()))
    print(f"== {fn}: 후진 {rev}, 주기 {PER * dt:g} s | 원본이 1 s 넘게 서 있던 복제 순간 {n_sync}개")
    for name in ("A", "V"):
        X = np.array(rows[name], float)
        if not len(X):
            continue
        t_go = X[:, 2]
        print(f"  {'그대로' if name == 'A' else '이어서 후진'} ({name}) 복제 {len(X)}대: 끝에 앞으로 달림 {100 * X[:, 0].mean():.0f} % · 끝에도 서 있음 "
              f"{100 * X[:, 1].mean():.0f} % · 앞으로 1 m/s 넘은 때 중앙 {np.median(t_go):.1f} s (못 넘음 {100 * np.isinf(t_go).mean():.0f} %) · "
              f"선 자리보다 앞으로 간 거리 중앙 {np.median(X[:, 3]):+.2f} m (1 m 넘게 지나감 {100 * (X[:, 3] > 1).mean():.0f} %) · 세게 닿음 {X[:, 4].mean():.2f}/대")
