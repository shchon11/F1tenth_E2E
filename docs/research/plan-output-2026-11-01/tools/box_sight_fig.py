"""box_sight.py 결과 그림: 왼쪽 = 지나가기 전 보인 시간별 세게 닿은 비율(그 상자를 두 번째부터 지나간 건), 오른쪽 = 장면 둘
(늦게 보였지만 깨끗이 지나간 건, 일찍 보였는데 세게 닿은 건). 장면: 처음 보인 순간의 차(빈 원)와 시선(점선), 그 뒤 지나갈 때까지의 경로.
인자: npz 출력png"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False

D = np.load(sys.argv[1]); R = np.load(sys.argv[1].replace(".npz", "_sight.npy"))
st, sid, pose, tid = D["st"], D["sid"], D["pose"], D["tid"]
_edt, _off, _H, _W, _res, _org = D["edt"], D["t_off"], D["t_H"].astype(int), D["t_W"].astype(int), D["t_res"], D["t_origin"]
U = R[(R[:, 8] < 0.5) & (R[:, 2] >= 0.5)]
BINS = ((0.0, 1.0, "1 s 안 됨"), (1.0, 1.5, "1~1.5 s"), (1.5, 2.5, "1.5~2.5 s"), (2.5, 99.0, "2.5 s 넘게"))
fig = plt.figure(figsize=(16, 5.6), dpi=110)
ax = fig.add_axes([0.05, 0.14, 0.27, 0.70])
INK, MUTED, BAR = "#1f2328", "#6b7280", "#3b6fb6"
for i, (lo, hi, lab) in enumerate(BINS):
    m = U[(U[:, 0] >= lo) & (U[:, 0] < hi)]
    n, h = len(m), int(m[:, 4].sum())
    r = h / max(n, 1) * 100
    ax.bar(i, r, width=0.6, color=BAR)
    ax.text(i, r + 0.08, f"{r:.1f} %\n{h} / {n}건", ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(range(len(BINS))); ax.set_xticklabels([b[2] for b in BINS], fontsize=10)
ax.set_ylim(0, 4.6); ax.set_ylabel("세게 닿은 비율 (%)", color=INK)
ax.set_xlabel("지나가기 전에 그 상자가 벽에 안 가리고 보인 시간", color=INK)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(axis="y", color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
ax.set_title("늦게 보인 상자가 더 많이 닿지 않음\n(같은 상자를 두 번째부터 지나간 건)", fontsize=11, loc="left", color=INK)


def scene(axs, row, title):
    b, j, t, u = (int(row[k]) for k in (10, 11, 12, 13))
    k = int(tid[0, b])
    occ = _edt[_off[k]: _off[k] + _H[k] * _W[k]].reshape(_H[k], _W[k]) <= 0.0
    ext = [_org[k][0], _org[k][0] + _W[k] * _res[k], _org[k][1], _org[k][1] + _H[k] * _res[k]]
    axs.imshow(np.where(occ, 0.80, 1.0), origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for jj in range(sid.shape[2]):
        if sid[t, b, jj] < 0:
            continue
        q = D[f"fp_{int(sid[t, b, jj])}"]; c, s_ = np.cos(pose[t, b, jj, 2]), np.sin(pose[t, b, jj, 2])
        poly = np.stack([pose[t, b, jj, 0] + q[:, 0] * c - q[:, 1] * s_, pose[t, b, jj, 1] + q[:, 0] * s_ + q[:, 1] * c], 1)
        axs.fill(poly[:, 0], poly[:, 1], color="#c2410c" if jj == j else "#9ca3af", zorder=3)
    t0 = max(u - 10, 0)
    axs.plot(st[t0: u + 1, b, 0], st[t0: u + 1, b, 1], color=MUTED, lw=1.5, zorder=4)
    axs.plot(st[u: t + 6, b, 0], st[u: t + 6, b, 1], color=BAR, lw=2.2, zorder=5)
    axs.plot([st[u, b, 0], pose[u, b, j, 0]], [st[u, b, 1], pose[u, b, j, 1]], ls=(0, (4, 3)), color="#c2410c", lw=1.2, zorder=4)
    axs.scatter([st[u, b, 0]], [st[u, b, 1]], s=60, facecolor="white", edgecolor=BAR, lw=2, zorder=6)
    axs.scatter([st[t, b, 0]], [st[t, b, 1]], s=40, color=BAR, zorder=6)
    cx, cy = 0.5 * (st[u, b, 0] + pose[t, b, j, 0]), 0.5 * (st[u, b, 1] + pose[t, b, j, 1])
    hw_ = max(4.0, 0.5 * np.hypot(st[u, b, 0] - pose[t, b, j, 0], st[u, b, 1] - pose[t, b, j, 1]) + 2.0)
    axs.set_xlim(cx - hw_, cx + hw_); axs.set_ylim(cy - hw_, cy + hw_); axs.set_aspect("equal")
    axs.set_xticks([]); axs.set_yticks([])
    axs.set_title(f"{title}\n처음 보임: {row[0]:.1f} s 전, {row[1]:.1f} m | 그때 속도 {abs(st[u, b, 3]):.1f} m/s", fontsize=10.5, loc="left", color=INK)


late = U[(U[:, 0] >= 0.4) & (U[:, 0] < 1.0) & (U[:, 3] < 0.5) & (U[:, 7] > 5.0)]
early = U[(U[:, 0] >= 1.5) & (U[:, 0] < 2.5) & (U[:, 4] > 0.5)]
rng = np.random.default_rng(3)
if len(late):
    scene(fig.add_axes([0.36, 0.06, 0.29, 0.78]), late[rng.integers(len(late))], "늦게 보였고 닿지 않은 건")
if len(early):
    scene(fig.add_axes([0.68, 0.06, 0.29, 0.78]), early[rng.integers(len(early))], "일찍 보였는데 세게 닿은 건")
fig.text(0.36, 0.955, "빈 원 = 그 상자(주황)가 처음 보인 순간의 차, 점선 = 시선, 파란 선 = 그 뒤 지나갈 때까지의 경로, 회색 선 = 그 전 1 s", fontsize=10, color=MUTED)
fig.savefig(sys.argv[2], facecolor="white")
print("저장", sys.argv[2])
