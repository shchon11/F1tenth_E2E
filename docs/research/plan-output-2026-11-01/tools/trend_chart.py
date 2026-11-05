"""지금 학습 중인 두 run의 추이를 한 장으로 (11-05 11:45, 사용자: "좀 알아먹기 편하게좀 보여줘라 숫자 싸놓으면 제대로 읽히겠냐고").
칸마다 숫자 하나: 2025 빈 트랙 lap / obstacle 세게 부딪힘(평가) / 차 세게 부딪힘(평가) / 학습 중 충돌, 2026 빈 트랙 lap / 학습 중 충돌.
점선 = 목표. 평가 숫자는 one_<run>.log와 one_srp25owa_extra.log에서, 학습 중 숫자는 progress.jsonl을 50 update 구간 평균으로. 인자: 출력png"""
import json, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
D = "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out"
R = "/home/shchon11/f1sim_runs"
SER, TXT, MUT, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
out = sys.argv[1]


def solo(run, start):
    xs, ys = [0], [start]
    for l in open(f"{D}/one_{run}.log"):
        m = re.search(r"u(\d+) solo: .*랩 \[([^\]]*)\]", l)
        if m and m.group(2).strip():
            xs.append(int(m.group(1))); ys.append(float(np.median([float(v) for v in m.group(2).split(",")])))
    return xs, ys


def train(run, key):
    L = [json.loads(l) for l in open(f"{R}/{run}/progress.jsonl")]
    xs, ys = [], []
    for a in range(0, len(L), 50):
        seg = [d[key] for d in L[a:a + 50] if d.get(key) is not None]
        if len(seg) >= 10:
            xs.append(a + len(seg)); ys.append(float(np.mean(seg)))
    return xs, ys


def obs30(run, start):
    xs, ys = [0], [start]
    for l in open(f"{D}/one_{run}.log"):
        m = re.search(r"u(\d+) obstacle 30 layout: .*벽 \d+\(세게 (\d+)\) 장애물 \d+\(세게 (\d+)\)", l)
        if m:
            xs.append(int(m.group(1))); ys.append(int(m.group(2)) + int(m.group(3)))
    return xs, ys


def car06(run, start):
    xs, ys = [0], [start]
    for l in open(f"{D}/one_{run}.log"):
        m = re.search(r"u(\d+) forzaeth x0.6 .*차 접촉 \d+ \(세게 (\d+)\)", l)
        if m:
            xs.append(int(m.group(1))); ys.append(int(m.group(2)))
    return xs, ys


def panel(ax, x, y, title, target=None, tlabel="목표", unit="", lower=True, fmt="{:.2f}", zero=False):
    ax.plot(x, y, color=SER, lw=2, marker="o", ms=8, markeredgecolor="white", markeredgewidth=2, zorder=3)
    if target is not None:
        ax.axhline(target, color=MUT, lw=1.2, ls="--", zorder=1)
        ax.text(1.0, target, f" {tlabel} {fmt.format(target)}{unit}", color=MUT, fontsize=10, va="center",
                ha="left", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(x[0], y[0], f"{fmt.format(y[0])}{unit}  ", color=TXT, fontsize=10, ha="right", va="center")
    ax.text(x[-1], y[-1], f"  {fmt.format(y[-1])}{unit}", color=TXT, fontsize=11, fontweight="bold", ha="left", va="center")
    ax.set_title(title + ("  (낮을수록 좋음)" if lower else ""), loc="left", fontsize=12, color=TXT)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.set_xlabel("학습 update", color=MUT, fontsize=9)
    lo, hi = min(min(y), target if target is not None else min(y)), max(max(y), target if target is not None else max(y))
    pad = (hi - lo) * 0.25 + 1e-3
    ax.set_ylim(0 if zero else lo - pad, hi + pad)
    ax.set_xlim(-15, max(max(x), 50) * 1.18)


fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), facecolor="#fcfcfb")
for ax in axs.flat:
    ax.set_facecolor("#fcfcfb")
x, y = solo("srp25owa", 7.8); panel(axs[0, 0], x, y, "2025 · 빈 트랙 한 바퀴 [s]", 6.65, unit=" s")
x, y = obs30("srp25owa", 41); panel(axs[0, 1], x, y, "2025 · obstacle 세게 부딪힘 (30 layout)", 47, fmt="{:.0f}", unit="번", zero=True)
x, y = car06("srp25owa", 4); panel(axs[0, 2], x, y, "2025 · 상대 0.6배 차와 세게 부딪힘 (20판)", 8, fmt="{:.0f}", unit="번", zero=True)
x, y = train("srp25owa", "hits_wall_prop_per_km"); panel(axs[1, 0], x, y, "2025 · 학습 중 벽·obstacle 세게 부딪힘 [/km]", fmt="{:.1f}", zero=True)
x, y = solo("srp26owa", 9.4); panel(axs[1, 1], x, y, "2026 · 빈 트랙 한 바퀴 [s]", 7.55, unit=" s")
x, y = train("srp26owa", "hits_wall_prop_per_km"); panel(axs[1, 2], x, y, "2026 · 학습 중 벽·obstacle 세게 부딪힘 [/km]", fmt="{:.1f}", zero=True)
fig.suptitle("지금 학습 중인 두 run (네트워크 하나, 빈 상태만 solo 정답 + PPO) · 점선 = 목표 · 0 = 시작 checkpoint",
             x=0.01, ha="left", fontsize=13, color=TXT)
plt.tight_layout(rect=(0, 0, 0.97, 0.95))
plt.savefig(out, dpi=110, facecolor=fig.get_facecolor())
print(out)
