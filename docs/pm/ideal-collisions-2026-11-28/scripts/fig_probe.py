"""그림: 입력 조작 시험. (왼쪽) 상자까지 거리별, 상자가 계획을 바꾼 크기 |dlat10| (1.0 s 점 옆 차이) -- 안 닿은 통과 vs 세게 닿은 접근,
(가운데/오른쪽) 같은 상자의 안 닿은 통과들(회색)과 세게 닿은 통과(빨강): 차 틀에서 상자의 옆 위치, 그리고 상자가 바꾼 계획 옆 차이.
인자: probe32_rows.json out.png b j"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import plt
rows = json.load(open(sys.argv[1])); b0, j0 = int(sys.argv[3]), int(sys.argv[4])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
edges = np.array([0, 1, 2, 3, 4, 6, 8, 10])
for hit, col, nm in ((False, "#1f77b4", "안 닿은 통과"), (True, "#d62728", "세게 닿은 접근")):
    D = [(d, abs(x)) for s in rows if s["hit"] == hit and (hit or not s["any"]) for x, n, d in zip(s["dl_series"], s["nb_series"], s["dist_series"]) if n >= 2]
    D = np.array(D)
    med, q1, q3, xs = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (D[:, 0] >= lo) & (D[:, 0] < hi)
        if m.sum() >= 3:
            xs.append(0.5 * (lo + hi)); med.append(100 * np.median(D[m, 1])); q1.append(100 * np.percentile(D[m, 1], 25)); q3.append(100 * np.percentile(D[m, 1], 75))
    ax[0].plot(xs, med, "o-", color=col, label=f"{nm} (중앙, 띠 = 25-75 %)"); ax[0].fill_between(xs, q1, q3, color=col, alpha=0.15)
ax[0].invert_xaxis(); ax[0].set_xlabel("상자까지 거리 [m] (상자 빔 >= 2개인 결정)"); ax[0].set_ylabel("상자가 바꾼 1.0 s 계획점 옆 위치 [cm]")
ax[0].set_title("상자를 입력에서 지웠을 때와의 계획 차이 (scan31b u475, 이상 실행)", fontsize=9); ax[0].legend(fontsize=8)
S = [s for s in rows if s["b"] == b0 and s["j"] == j0]
for s in S:
    col = "#d62728" if s["hit"] else "#999999"; lw = 2 if s["hit"] else 1
    lab = f"{s['enc'] + 1}번째 ({'세게 닿음' if s['hit'] else '여유 %.0f cm' % (100 * s['clear'])})"
    ax[1].plot(s["dist_series"], 100 * np.array(s["yb_series"]), "-", color=col, lw=lw, label=lab)
    ax[2].plot(s["dist_series"], 100 * np.array(s["dl_series"]), "-", color=col, lw=lw, label=lab)
for a in ax[1:]:
    a.invert_xaxis(); a.axhline(0, color="k", lw=0.6); a.set_xlabel("상자까지 거리 [m]"); a.legend(fontsize=7)
ax[1].set_ylabel("차 틀에서 상자 중심의 옆 위치 [cm] (+ 왼쪽)"); ax[1].set_title(f"세션 {b0} 상자 {j0}: 다가가는 동안 상자의 옆 위치", fontsize=9)
ax[2].set_ylabel("상자 있음 - 지움, 1.0 s 계획점 옆 차이 [cm]"); ax[2].set_title("같은 통과들에서 상자가 계획을 바꾼 크기", fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=110); print("saved", sys.argv[2])
