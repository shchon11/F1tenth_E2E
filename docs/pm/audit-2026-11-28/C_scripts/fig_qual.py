"""Audit C: qualifier summary figure. Small multiples (one measure per panel, one axis each): best lap (session median),
hard contacts per lap (wall + box), DNF share. x = driver, bars = execution condition (sequential blue: ideal -> MAP ->
MAP + stress), 95 % cluster-bootstrap whiskers. Args: out.png driver=cond:json [driver=cond:json ...]"""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/C")
from tables import ci_stat, med_f, mean_f, ratio
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import plt

COND = ["ideal", "MAP", "MAP+스트레스"]
COL = {"ideal": "#9ec5f4", "MAP": "#3987e5", "MAP+스트레스": "#104281"}
data = {}
drivers = []
for arg in sys.argv[2:]:
    key, fn = arg.split(":", 1)
    drv, cond = key.split("="); drv = drv.replace("\\n", "\n")
    if drv not in drivers:
        drivers.append(drv)
    S = json.load(open(fn))[0]["sessions"]; cl = [s["cluster"] for s in S]
    g = lambda k: np.array([s[k] for s in S], float)
    data[(drv, cond)] = dict(best=ci_stat(g("best"), cl, med_f), hard=ratio(S, ["wall_hard", "box_hard"], "n_laps"),
                             dnf=ci_stat(g("dnf"), cl, mean_f))
fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
W = 0.26
for ax, (k, title, scale) in zip(axs, (("best", "최고 랩타임 [s] (세션 중앙값)", 1), ("hard", "세게 닿음 / 바퀴 (벽+상자)", 1),
                                       ("dnf", "완주 실패 세션 [%]", 100))):
    for j, c in enumerate(COND):
        xs, ys, lo, hi = [], [], [], []
        for i, d in enumerate(drivers):
            if (d, c) not in data:
                continue
            e, l, h = data[(d, c)][k]
            xs.append(i + (j - 1) * W); ys.append(e * scale); lo.append((e - l) * scale); hi.append((h - e) * scale)
        if xs:
            ax.bar(xs, ys, W * 0.92, color=COL[c], label=c, edgecolor="white", linewidth=1.5)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#444444", elinewidth=1, capsize=2)
            for x, y in zip(xs, ys):
                ax.text(x, y, f"{y:.2f}" if k != "dnf" else f"{y:.0f}", ha="center", va="bottom", fontsize=7, color="#333333")
    ax.set_xticks(range(len(drivers))); ax.set_xticklabels(drivers, fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
    if k == "best":
        ax.set_ylim(6.5, max(9.5, ax.get_ylim()[1]))
axs[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("예선 형식 (120 s, 상자 3개 → 60 s에 치움, 세션 128개 = exam3 배치 64개 × 2) · 조건별", fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=110); print("saved", sys.argv[1])
