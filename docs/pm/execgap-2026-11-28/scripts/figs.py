"""보고서 그림 (PNG, 밝은 바탕). 인자: 단계표json 분석json 같은plan npz 그림폴더"""
import sys, json, os
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import plt
from rec_analysis import yaw_tf

TBL, AN, SP, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
S1, S2, S3, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993", "#0b0b0b", "#52514e"
plt.rcParams.update({"axes.edgecolor": GRAY, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.8, "font.size": 11})

T = json.load(open(TBL))
labs = list(T)
# ---- fig 1: stages, small multiples (one measure per panel)
fig, axs = plt.subplots(1, 3, figsize=(16, 0.45 * len(labs) + 1.8), sharey=True)
y = np.arange(len(labs))[::-1]
for ax, (k, title, sc, col) in zip(axs, (("clean", "120 s 동안 세게 한 번도 안 닿은 세션 [%]", 100, S1),
                                        ("wall_hard", "세게 닿음 / 세션 : 벽", 1, S2), ("box_hard", "세게 닿음 / 세션 : 상자", 1, S3))):
    v = np.array([T[l][k] for l in labs]) * sc
    ax.hlines(y, v[:, 1], v[:, 2], color=col, lw=2)
    ax.plot(v[:, 0], y, "o", color=col, ms=8, mec="white", mew=2)
    for yi, vi in zip(y, v[:, 0]):
        ax.annotate(f"{vi:.0f}" if sc == 100 else f"{vi:.2f}", (vi, yi), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=9, color=MUTED)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlim(left=0)
axs[0].set_yticks(y); axs[0].set_yticklabels(labs)
fig.text(0.01, 0.005, "점 = 64세션 값, 선 = 세션 부트스트랩 95 % 구간. 같은 정책(scan31b u475)·같은 상자 배치·같은 차 물리 추첨.", color=MUTED, fontsize=9)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{OUT}/fig1_stages.png", dpi=130); plt.close(fig)

A = json.load(open(AN))
# ---- fig 2: spectrum of the requested steering vs the car's response
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
ax = axs[0]
for nm, col, lab in (("R0_ideal", S1, "R0 이상 실행: plan이 요구한 조향"), ("R4_map", S2, "R4 MAP: plan이 요구한 조향")):
    if nm in A:
        f = np.array(A[nm]["B"]["f"]); p = np.array(A[nm]["B"]["psd_steer_req"])
        ax.semilogy(f[1:], p[1:], color=col, lw=2, label=lab)
if "R4_map" in A:
    f = np.array(A["R4_map"]["B"]["f"]); p = np.array(A["R4_map"]["B"]["psd_steer_car"])
    ax.semilogy(f[1:], p[1:], color=GRAY, lw=2, ls="--", label="R4 MAP: 실제 바퀴 조향각")
ax.axvline(5.0, color=GRAY, lw=1); ax.text(5.2, 1e-9, "5 Hz = 10 Hz 갱신의 절반", fontsize=9, color=MUTED)
ax.set_xlabel("주파수 [Hz]"); ax.set_title("조향 신호의 파워 스펙트럼 [rad²/Hz]", loc="left", fontsize=11); ax.legend(frameon=False, fontsize=9, loc="upper right")
ax = axs[1]
ff = np.linspace(0.05, 10, 400)
for (d, tau, ls, lab) in ((0.03, 0.04, "-", "명목 차 (지연 0.03 s, 서보 0.04 s)"), (0.05, 0.07, "--", "무작위 범위 끝 (0.05 s, 0.07 s)"),
                          (0.10, 0.04, ":", "실차 bag 단순 모형 지연 0.10 s (D 보고서)")):
    Hf = yaw_tf(ff, 5.0, d, tau)
    ax.plot(ff, np.abs(1 - Hf), color=S1, lw=2, ls=ls, label=lab)
ax.axhline(0.5, color=GRAY, lw=1)
ax.set_xlabel("주파수 [Hz]"); ax.set_ylim(0, 2)
ax.set_title("5 m/s에서 조향 명령 → 요 각속도: 사인 성분을 따라갈 때 상대 오차 |1 − H(f)|", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_spectrum.png", dpi=130); plt.close(fig)

# ---- fig 3: consecutive plan mismatch by situation (0.5 s, time matched), R0 vs R4
sits = ["전체", "직선(|곡률|<0.1)", "코너(|곡률|>0.4)", "상자 3 m 안", "세게 닿기 직전 1 s"]
fig, ax = plt.subplots(figsize=(11, 4.4))
yy = np.arange(len(sits))[::-1]
for off, (nm, col, lab) in zip((0.15, -0.15), (("R0_ideal", S1, "R0 이상 실행"), ("R4_map", S2, "R4 MAP"))):
    if nm not in A:
        continue
    v = np.array([A[nm]["A"][s]["shape_path_0.5s"][:3] for s in sits]) * 100
    ax.hlines(yy + off, v[:, 0], v[:, 2], color=col, lw=2, alpha=0.5)
    ax.plot(v[:, 0], yy + off, "o", color=col, ms=8, mec="white", mew=2, label=f"{lab}: 중앙값 (선 끝 = 99번째 백분위)")
    ax.plot(v[:, 1], yy + off, "|", color=col, ms=14, mew=2)
ax.set_yticks(yy); ax.set_yticklabels(sits); ax.set_xlabel("[cm]")
ax.set_title("0.1 s 간격 두 plan의 의도 차이: 앞 plan을 그 0.1 s 자리에 맞춰 놓고, 새 plan의 0.5 s 점이 앞 plan의 길에서 떨어진 거리\n(차가 실제로 어디 있었는지는 뺌) — 점 중앙값, 세로 막대 90번째, 선 끝 99번째 백분위", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_plan_mismatch.png", dpi=130); plt.close(fig)

# ---- fig 4: event classes (exclusive), stages with events
CL = ["계획이 장애물을 지나감", "계획이 직전에 튐", "계획이 못 따라갈 것을 요구", "계획은 괜찮았는데 못 따라감", "기타(계획 여유 2 cm 이상·추종 오차 작음)"]
CC = [S2, "#eda100", "#6250d6", S1, GRAY]
order = [n for n in ("R0_ideal", "R1_delay", "R2s_servoonly", "R2_servo", "R2b_motor", "R3b_pforacle", "R3_mpc", "R6_pf", "R4_map", "R5a_la", "R5b_gain", "R5c_lut") if n in A]
fig, axs = plt.subplots(1, 2, figsize=(15, 0.45 * len(order) + 1.8), sharey=True)
yy = np.arange(len(order))[::-1]
for ax, kind, title in zip(axs, ("wall", "box"), ("벽에 세게 닿은 사건", "상자에 세게 닿은 사건")):
    left = np.zeros(len(order))
    tot = np.array([max(1, A[n]["D_n"][kind]) for n in order])
    for c, col in zip(CL, CC):
        v = np.array([A[n]["D_counts"].get(kind, {}).get(c, 0) for n in order]) / tot * 100
        ax.barh(yy, v, left=left, color=col, height=0.7, edgecolor="white", linewidth=2, label=c)
        left += v
    for yi, n in zip(yy, order):
        ax.text(101, yi, f"n={A[n]['D_n'][kind]}", va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, 112); ax.set_title(title + " [%] (우선순위 순서로 하나만)", loc="left", fontsize=11)
axs[0].set_yticks(yy); axs[0].set_yticklabels(order)
axs[1].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.0, -0.08), ncol=3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_events.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ---- fig 5: same plan, followers
if os.path.exists(SP):
    Z = np.load(SP)
    names = [k[len("trj__"):] for k in Z.files if k.startswith("trj__") and k != "trj__plan"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.6))
    ax = axs[0]
    for i, nm in enumerate(names):
        lat = np.abs(Z[f"m__{nm}__lat"])                       # (n, 10) at 0.1..1.0 s
        med = np.median(lat, 0) * 100; p90 = np.percentile(lat, 90, 0) * 100
        col = [S1, "#9ec5f4", "#104281", "#5d9ce9", S2, S3][i % 6]
        ax.plot(np.arange(1, 11) * 0.1, med, color=col, lw=2, label=nm)
        ax.plot(np.arange(1, 11) * 0.1, p90, color=col, lw=1, ls="--")
    ax.set_xlabel("plan을 받은 뒤 [s]"); ax.set_ylabel("[cm]")
    ax.set_title("같은 plan 1 s 열린 고리: plan과의 옆 오차 (실선 중앙값, 점선 90번째)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1]
    P = {nm: Z[f"trj__{nm}"] for nm in names}
    M = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = np.median(np.hypot(P[a][:, 20, 0] - P[b][:, 20, 0], P[a][:, 20, 1] - P[b][:, 20, 1])) * 100
    im = ax.imshow(M, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=9, color=INK if M[i, j] < M.max() * 0.6 else "white")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.grid(False)
    ax.set_title("같은 plan, 0.5 s 뒤 두 추종기 차 위치 차이의 중앙값 [cm]", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_same_plan.png", dpi=130); plt.close(fig)
print("figs ->", OUT)
