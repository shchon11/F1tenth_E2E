"""Audit C: 1v1 vs the frozen self-copy (race4): win rate and passes by the copy's pace, ideal vs MAP, per start type.
Two rows (win rate, passes per race) x two columns (half a lap apart, 5 m behind); bars = condition. 95 % cluster bootstrap.
Args: out.png cond:json [cond:json ...]"""
import sys, json
import numpy as np
sys.path.insert(0, "/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/C")
from tables import ci_stat, mean_f
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import plt

COL = ["#9ec5f4", "#3987e5", "#104281"]
runs = [(a.split(":", 1)[0], json.load(open(a.split(":", 1)[1]))[0]["race"]) for a in sys.argv[2:]]
paces = sorted(set(round(x["c_pace"], 2) for x in runs[0][1]))
fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharey="row")
W = 0.8 / len(runs)
for c, sname in ((0, "반 바퀴 떨어져 출발 (사용자 대회 방식)"), (1, "5 m 뒤에서 출발")):
    for r_, (k, ylab, sc) in enumerate((("win", "승률 [%] (10바퀴 먼저)", 100), ("passes", "앞지름 / 경기", 1))):
        ax = axs[r_, c]
        for j, (lab, rr) in enumerate(runs):
            xs, ys, lo, hi = [], [], [], []
            for i, p in enumerate(paces):
                sel = [x for x in rr if abs(x["c_pace"] - p) < 1e-3 and x["c_start"] == c]
                e, l, h = ci_stat([x[k] for x in sel], [x["cluster"] for x in sel], mean_f)
                xs.append(i + (j - (len(runs) - 1) / 2) * W); ys.append(e * sc); lo.append((e - l) * sc); hi.append((h - e) * sc)
            ax.bar(xs, ys, W * 0.92, color=COL[j], label=lab, edgecolor="white", linewidth=1.5)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#444444", elinewidth=1, capsize=2)
        ax.set_xticks(range(len(paces))); ax.set_xticklabels([f"{p:.2f}배" for p in paces])
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
        if c == 0:
            ax.set_ylabel(ylab)
        if r_ == 0:
            ax.set_title(sname, fontsize=11)
        if r_ == 1:
            ax.set_xlabel("상대 = scan31b u500 얼린 복사본의 페이스 배율 (복사본은 늘 plan대로 정확히 달림)")
axs[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle("본선 형식: race33b u975 vs 자기 복사본, 120 s 안에 10바퀴 먼저 (조건마다 16경기)", fontsize=12)
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=110); print("saved", sys.argv[1])
