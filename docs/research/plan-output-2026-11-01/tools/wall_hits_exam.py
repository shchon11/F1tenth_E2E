"""시험 기록의 벽 세게 닿음이 어디서·어떤 상황에서 나나 (11-15 05:10: plan26r u300이 plan26p u250보다 벽 세게 +0.09~0.19/세션).
세게 닿은 벽 접촉마다: 닿기 2 s 안에 차 중심 1.5 m 안을 지난 상자가 있었나(상자를 비키다 벽), 상대 차가 2 m 안에 있었나, 닿을 때 속도,
닿은 자리. 표 = 기록별 상황 비율, 그림 = 지도 위 닿은 자리(기록마다 색).
인자: 출력png 이름=기록npz [이름=기록npz ...]"""
import sys
import numpy as np
from rec_lib import Rec, plt

out = sys.argv[1]
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
cols = ("#1c7ed6", "#e8590c", "#2b8a3e", "#ae3ec9")
for i, a in enumerate(sys.argv[2:]):
    nm, fn = a.split("=", 1)
    R = Rec(fn); st = R.st.astype(float); op = R.op.astype(float); T = R.d.shape[0]; B = R.d.shape[1]
    on = np.asarray(R.z["on"]).astype(bool)
    rows = []
    for t, b in zip(*np.nonzero(R.d[:, :, 3] > 0)):
        a0 = max(0, t - 20)
        near_box = False
        for s in range(a0, t + 1, 2):
            for P in R.boxes(s, b):
                if np.min(np.linalg.norm(P.mean(0) - st[s, b, :2])) < 1.5:
                    near_box = True; break
            if near_box:
                break
        near_car = bool(on[t, b]) and np.linalg.norm(op[t, b, :2] - st[t, b, :2]) < 2.0
        has_box = (R.sid[t, b] >= 0).any()
        rows.append((t, b, near_box, near_car, has_box, abs(st[t, b, 3]), st[t, b, 0], st[t, b, 1]))
    r = np.array(rows, dtype=float)
    n = len(r); S = B
    print(f"== {nm}: 벽 세게 {n}번 ({n / S:.2f}/세션)")
    if n:
        print(f"  2 s 안에 상자 1.5 m 안을 지남 {int(r[:, 2].sum())} ({100 * r[:, 2].mean():.0f} %) · 상대 차 2 m 안 {int(r[:, 3].sum())} ({100 * r[:, 3].mean():.0f} %) · "
              f"상자 있는 세션 {int(r[:, 4].sum())} ({100 * r[:, 4].mean():.0f} %) · 닿을 때 속도 중앙 {np.median(r[:, 5]):.1f} m/s")
        if i == 0:
            R.draw_map(ax, int(R.tid[0, 0]))
        ax.scatter(r[:, 6], r[:, 7], s=26, color=cols[i % 4], alpha=0.75, label=f"{nm} ({n}번)", edgecolors="none")
ax.legend(fontsize=9); ax.set_aspect("equal"); ax.set_title("벽에 세게 닿은 자리 (frozen exam, 두 기록)", fontsize=11)
fig.savefig(out, dpi=110, bbox_inches="tight"); print("그림", out)
