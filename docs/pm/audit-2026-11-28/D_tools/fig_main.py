import json, pickle, numpy as np
from figcommon import plt, COL, NAME, FIG
import models as M
T = json.load(open("tables.json"))
splits = [("fit", "대회장 조정용 bag 6개"), ("val", "대회장 확인용 bag 6개"), ("pre", "사전 bag 8개 (다른 노면·설정)")]
mets = [("yaw", "요레이트 RMS [rad/s]"), ("pos", "위치 오차 [m]"), ("v", "속도 오차 [m/s]")]
models = ["K1", "K2", "S0", "S1"]
fig, ax = plt.subplots(3, 3, figsize=(15, 11))
H = [50, 100, 200]
for c, (sp, title) in enumerate(splits):
    for r, (mk, lab) in enumerate(mets):
        a = ax[r, c]
        for j, m in enumerate(models):
            med = [T[sp]["all"][m][str(h)][mk][0] for h in H]
            p90 = [T[sp]["all"][m][str(h)][mk][1] for h in H]
            x = np.array([0.5, 1.0, 2.0]) + (j - 1.5) * 0.06
            a.plot(x, med, "-o", color=COL[m], lw=2, ms=7, label=NAME[m])
            a.vlines(x, med, p90, color=COL[m], lw=1.2, alpha=0.6)
        a.set_xticks([0.5, 1, 2]); a.set_xlabel("예측 구간 [s]")
        a.set_ylabel(lab + "  (점=중앙값, 선=p90)" if c == 0 else "")
        if r == 0: a.set_title(f"{title}  (n={T[sp]['all']['n']})", fontsize=11)
        a.grid(alpha=0.25); a.set_ylim(bottom=0)
ax[0, 0].legend(fontsize=9, loc="upper left")
fig.suptitle("기록된 명령 개방 루프 재생 — 복잡한 시뮬 vs 단순 기준선 (낮을수록 좋음)", fontsize=13)
fig.tight_layout(); fig.savefig(FIG + "D1_openloop_errors.png", dpi=110)

# CDF of 1 s yaw RMS on the held-out competition split
Wa = pickle.load(open("windows.pkl", "rb")); P = pickle.load(open("preds.pkl", "rb"))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for k, sp in enumerate(["val", "pre"]):
    W = Wa[sp]
    for m in ["K2", "S0", "S1", "K1"]:
        e = np.sort(M.errors(W, P[sp][m])[100]["yaw"])
        ax[k].plot(e, np.arange(1, len(e) + 1) / len(e), color=COL[m], lw=2, label=NAME[m])
    ax[k].set_xscale("log"); ax[k].set_xlabel("1초 요레이트 RMS [rad/s] (로그)"); ax[k].set_ylabel("구간 누적 비율")
    ax[k].set_title(dict(val="대회장 확인용 bag", pre="사전 bag (다른 노면)")[sp]); ax[k].grid(alpha=0.25)
    ax[k].axvline(0.6, color="#888", ls=":", lw=1)
ax[0].legend(fontsize=9, loc="lower right")
fig.suptitle("오차 분포: 시뮬은 중앙값은 비슷하지만 꼬리(약 10 %)가 길다 — 실차가 버틴 곳에서 시뮬 차가 스핀", fontsize=12)
fig.tight_layout(); fig.savefig(FIG + "D2_yaw_error_cdf.png", dpi=110)
print("ok")
