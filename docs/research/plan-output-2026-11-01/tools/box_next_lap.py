"""(11-17, 사용자: "최소한 박는게 있어도 다음바퀴에는 안박아야지") frozen exam 기록에서 상자를 지나갈 때마다(차가 그 상자 2 m 안에 든 구간 =
한 번 지나감) 세게 닿았나를 세고, 바로 앞 바퀴에 그 상자를 박았을 때 / 안 박았을 때 이번에 박을 확률을 비교. 상자를 치우기 전만.
인자: 기록npz [...]"""
import sys
import numpy as np
sys.path.insert(0, ".")
from rec_lib import Rec

for fn in sys.argv[1:]:
    R = Rec(fn); z = R.z
    st, sid, pose = R.st.astype(float), z["sid"], z["pose"].astype(float)
    T, B = R.rs.shape
    rm = int(round(float(z["rm_t"]) / R.dt)) if "rm_t" in z.files else T
    hit_t = R.d[..., 4] > 0
    after_hit, after_clean, first = [], [], []
    n_pass = 0
    for b in range(B):
        for j in np.nonzero(sid[0, b] >= 0)[0]:
            alive = sid[:rm, b, j] >= 0
            d = np.hypot(pose[:rm, b, j, 0] - st[:rm, b, 0], pose[:rm, b, j, 1] - st[:rm, b, 1])
            near = (d < 2.0) & alive & ~R.rs[:rm, b]
            x = np.r_[0, near.astype(int), 0]; s = np.nonzero(np.diff(x) == 1)[0]; e = np.nonzero(np.diff(x) == -1)[0]
            passes = []
            for a_, b_ in zip(s, e):
                if b_ - a_ < 3:
                    continue
                # a hit on this box in this pass: a hard box contact while this box is the nearest live one
                h = False
                for t in range(a_, b_):
                    if hit_t[t, b]:
                        live = np.nonzero(sid[t, b] >= 0)[0]
                        dd = np.hypot(pose[t, b, live, 0] - st[t + 1, b, 0], pose[t, b, live, 1] - st[t + 1, b, 1])
                        if live[dd.argmin()] == j:
                            h = True; break
                passes.append(h)
            n_pass += len(passes)
            if passes:
                first.append(passes[0])
            for k in range(1, len(passes)):
                (after_hit if passes[k - 1] else after_clean).append(passes[k])
    ah, ac = np.array(after_hit, float), np.array(after_clean, float)
    print(f"{fn.split('/')[-1]}: 상자 지나감 {n_pass}번 | 처음 지나갈 때 박음 {100 * np.mean(first):.1f} % | "
          f"앞 바퀴에 그 상자를 박았으면 이번에도 박음 {100 * ah.mean() if len(ah) else float('nan'):.0f} % ({int(ah.sum())}/{len(ah)}) · "
          f"앞 바퀴에 안 박았으면 {100 * ac.mean():.1f} % ({int(ac.sum())}/{len(ac)})")
