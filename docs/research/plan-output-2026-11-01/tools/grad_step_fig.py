"""Figure for the stepped-checkpoint exams (grad_step.py, 11-13): what moving the u150 actor along the measured gradient does.

Left: one lap of one empty-track solo session (same world for every checkpoint), the base path and the moved ones on the map.
Middle: distance driven alone on the empty track [m / 90 s] minus the base, per step (paired sessions, +- standard error).
Right: hard contacts per session minus the base (box / car; walls listed in the title when off the scale).

    python3 grad_step_fig.py OUT.png BASE.npz "STEP=EXAM.npz" [...]      (STEP a number: the distance moved, sign = direction)
"""
import sys
import numpy as np
import rec_lib as RL
import matplotlib.pyplot as plt
import grad_step_table as GT

OUT, BASE = sys.argv[1], sys.argv[2]
runs = sorted([(float(a.split("=")[0]), a.split("=")[1]) for a in sys.argv[3:]])
b0 = GT.load(BASE); es = ~b0["box"] & ~b0["opp"]
rows = []
for s, fn in runs:
    q = GT.load(fn); dd = q["dist"] - b0["dist"]; dh = q["hard"] - b0["hard"]
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    rows.append((s, dd[es].mean(), se(dd[es]), dh[:, 1].mean(), se(dh[:, 1]), dh[:, 2].mean(), se(dh[:, 2]), dh[:, 0].mean(), se(dh[:, 0])))
rows.insert(0, (0.0, 0, 0, 0, 0, 0, 0, 0, 0)); rows.sort()
S = np.array([r[0] for r in rows])

fig, ax = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]})
r0 = RL.Rec(BASE); box = (r0.sid[0] >= 0).any(-1); opp = r0.on.any(0); b = int(np.nonzero(~box & ~opp)[0][0])
r0.draw_map(ax[0], 0)
cmap = plt.get_cmap("coolwarm_r"); smax = max(abs(S).max(), 1e-9)
t0, t1 = 300, 390
for s, fn in [(0.0, BASE)] + runs:
    r = RL.Rec(fn); P = r.st[t0:t1, b, :2]
    col = "#212529" if s == 0 else cmap(0.5 + 0.5 * s / smax)
    ax[0].plot(P[:, 0], P[:, 1], color=col, lw=2.4 if s == 0 else 1.8, label="기준 (u150)" if s == 0 else f"{s:+g} 옮김", zorder=3 if s == 0 else 2)
ax[0].legend(fontsize=9, loc="best", framealpha=0.9)
ax[0].set_title("빈 트랙 혼자 한 바퀴 (같은 세계, 30~39 s)\n트랙은 반시계 방향으로 돎: 왼쪽 = 코너 안쪽", fontsize=11)
for k, (j, sj, nm, col) in enumerate(((1, 2, "빈 트랙 혼자 달린 거리 [m / 90 s]", "#1864ab"),)):
    y = np.array([r[j] for r in rows]); e = np.array([r[sj] for r in rows])
    ax[1].errorbar(S, y, yerr=1.96 * e, color=col, marker="o", ms=6, lw=2, capsize=4)
    for x_, y_ in zip(S, y):
        ax[1].annotate(f"{y_:+.1f}", (x_, y_), textcoords="offset points", xytext=(6, 6), fontsize=9, color="#343a40")
    ax[1].set_ylabel(nm + ", 기준과의 차")
ax[1].set_title("더 빨리 도나\n(세션끼리 짝지음, 막대 = 95 % 구간)", fontsize=11)
off = []
for j, sj, nm, col in ((3, 4, "상자", "#a0522d"), (5, 6, "차", "#d62728"), (7, 8, "벽", "#495057")):
    y = np.array([r[j] for r in rows]); e = np.array([r[sj] for r in rows])
    keep = np.abs(y) < 2.0
    ax[2].errorbar(S[keep], y[keep], yerr=1.96 * e[keep], color=col, marker="o", ms=6, lw=2, capsize=4, label=nm)
    off += [f"{nm} {s:+g}: {v:+.1f}" for s, v, k_ in zip(S, y, keep) if not k_]
ax[2].set_ylabel("세션당 세게 닿음, 기준과의 차"); ax[2].legend(fontsize=9)
ax[2].set_title("세게 닿음이 줄었나" + (f"\n(그림 밖: {', '.join(off)} /세션)" if off else ""), fontsize=11)
for a in ax[1:]:
    a.axhline(0, color="#495057", lw=0.8); a.axvline(0, color="#adb5bd", lw=0.8, ls=":")
    a.set_xlabel("u150 actor를 옮긴 거리 (+ = 0.9 남김 rollout들의 평균 gradient가 가리킨 쪽)")
    a.grid(alpha=0.25)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.suptitle("학습 없이 u150 actor를 측정한 gradient 방향으로 조금 옮긴 뒤 frozen exam (128 세션 x 90 s, 기준과 같은 세계). 25 update 동안 학습이 실제로 옮긴 거리 = 5.9", fontsize=12.5)
fig.tight_layout(); fig.savefig(OUT, dpi=105); print("그림", OUT)
