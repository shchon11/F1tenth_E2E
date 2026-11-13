"""One figure: what the training batch can teach with exploration noise drawn fresh every decision vs noise that stays (11-13).

    python3 batch_compare.py OUT.png "LABEL=DUMP_DIR=RHO=batch_signal txt" [...]      (the first is the reference)

Left: how far 1 degree of steering noise moves the car sideways, by time after the decision (5-6.5 m/s; from the txt).
Middle: the advantage a noise of 1 SD earns [s], with its 95 % range: going faster with no box ahead, going faster with a
box 1.5-6 m ahead that the car then passes within 0.8 m, and the size of the steering direction the batch favours in cells of
similar state (batch_cells.measure; 0 = no direction).
Right: the same as seen in ONE update (slope / its standard error within one update).
"""
import sys, re
import numpy as np
import rec_lib  # noqa: F401  (Korean font)
import matplotlib.pyplot as plt
import batch_cells as BC

OUT = sys.argv[1]
runs = [a.split("=") for a in sys.argv[2:]]
COL = ["#868e96", "#4dabf7", "#1864ab", "#0b3d6b"]
res = []
for label, d, rho, txt in runs:
    rho = float(rho)
    D = BC.load(d, rho)
    T = open(txt).read(); T = T[T.index("[5~6.5 m/s]"):]
    lat = [(float(a), float(b)) for a, b in re.findall(r"([\d.]+) s 뒤: 옆으로 ([+-][\d.]+) cm", T)[:5]]
    rows = []
    nobox = [~q["has"] & q["full"] for q in D]
    near = [q["has"] & q["full"] & (q["dist"] >= 1.5) & (q["dist"] < 6.0) & (q["dmin"] < 0.8) for q in D]
    for m in (nobox, near):
        b, se = BC.slope(np.concatenate([q["adv"][mm] for q, mm in zip(D, m)]), np.concatenate([q["zv"][mm] for q, mm in zip(D, m)]))
        one = np.median([BC.slope(q["adv"][mm], q["zv"][mm])[0] / BC.slope(q["adv"][mm], q["zv"][mm])[1] for q, mm in zip(D, m)])
        rows.append((b, 1.96 * se, one))
    for cells in (BC.cells_free, BC.cells_box):
        s_, se1, chi, nc, n, eff = BC.measure(D, cells, "z")
        rows.append((max(s_, 0.0), 0.0, max(s_, 0.0) / se1))
    res.append((label, lat, rows)); print(label, lat, [(round(a, 4), round(b, 4), round(c, 2)) for a, b, c in rows], flush=True)

fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={"width_ratios": [1.0, 1.35, 1.35]})
for i, (label, lat, rows) in enumerate(res):
    ax[0].plot([a for a, _ in lat], [b for _, b in lat], color=COL[i], lw=2.2, marker="o", ms=5, label=label)
    ax[0].annotate(f"{lat[2][1]:.0f} cm" if lat[2][1] >= 10 else f"{lat[2][1]:.1f} cm", (lat[2][0], lat[2][1]), textcoords="offset points", xytext=(6, 6), fontsize=9, color="#343a40")
ax[0].axhspan(10, 20, color="#e03131", alpha=0.08, lw=0); ax[0].text(1.52, 10.6, "세게 닿은 경우에 모자랐던 폭 (10~20 cm)", fontsize=9, color="#c92a2a", va="bottom", ha="right")
ax[0].set_xlabel("결정 뒤 지난 시간 [s]"); ax[0].set_ylabel("조향 잡음 1도가 차를 옆으로 옮긴 거리 [cm]")
ax[0].set_title("잡음 한 번이 차를 실제로 얼마나 옮기나\n(5~6.5 m/s)", fontsize=11); ax[0].legend(fontsize=9, loc="upper left")
NAMES = ["앞에 상자 없음:\n더 빠르게", "가까이 지나갈 상자가\n1.5~6 m 앞: 더 빠르게", "앞에 상자 없음:\n조향을 틀 방향", "상자 1.5~6 m 앞:\n조향을 틀 방향"]
w = 0.8 / len(res)
for i, (label, lat, rows) in enumerate(res):
    x = np.arange(4) + (i - (len(res) - 1) / 2) * w
    ax[1].bar(x, [r[0] for r in rows], w * 0.92, color=COL[i], label=label)
    ax[1].errorbar(x[:2], [r[0] for r in rows[:2]], yerr=[r[1] for r in rows[:2]], fmt="none", ecolor="#212529", lw=1.2, capsize=3)
    ax[2].bar(x, [r[2] for r in rows], w * 0.92, color=COL[i], label=label)
    for xx, r in zip(x, rows):
        ax[1].text(xx, r[0] + (0.004 if r[0] >= 0 else -0.004) + np.sign(r[0] or 1) * r[1], f"{r[0]:+.3f}", ha="center", va="bottom" if r[0] >= 0 else "top", fontsize=8, color="#343a40")
        ax[2].text(xx, r[2] + (0.12 if r[2] >= 0 else -0.12), f"{r[2]:+.1f}", ha="center", va="bottom" if r[2] >= 0 else "top", fontsize=8, color="#343a40")
ax[1].set_ylabel("잡음 1 표준편차가 받은 advantage [s]\n(조향 쪽은 batch가 가리키는 방향의 크기, 0 = 방향 없음)")
ax[1].set_title("batch가 가르쳐 주는 것의 크기\n(막대 위 선 = 95 % 구간)", fontsize=11)
ax[2].axhline(2, color="#868e96", lw=0.9, ls="--"); ax[2].axhline(-2, color="#868e96", lw=0.9, ls="--")
ax[2].set_ylabel("update 한 번 안에서 크기 / 표준오차"); ax[2].set_title("update 한 번이 그것을 얼마나 또렷이 보나\n(점선 ±2 밖이면 한 번에 방향이 보임)", fontsize=11)
for a in ax[1:]:
    a.axhline(0, color="#495057", lw=0.8); a.set_xticks(np.arange(4)); a.set_xticklabels(NAMES, fontsize=9); a.legend(fontsize=9, loc="best")
for a in ax:
    a.grid(alpha=0.25, axis="y")
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("plan25y u150이 학습 때 받는 batch (update 없이 rollout 7개씩): 탐색 잡음을 결정마다 새로 뽑을 때와 이어지게 뽑을 때", fontsize=13)
fig.tight_layout(); fig.savefig(OUT, dpi=110); print("그림", OUT)
