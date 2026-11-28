"""Figures for REPORT.md (F1 tyre curves, F2 profiles, F3 final check, F4 transfer, F6 excitation)."""
import json, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import WORK, FIGS, SPLIT
from f1sim.bag_profile import models as M, windows as WN
from f1sim.bag_map import _use_korean_font
from f1sim.params import Config
_use_korean_font(plt)
which = sys.argv[1:] or ["F1", "F2", "F3", "F4", "F6"]
R = json.load(open(f"{WORK}/fit1_comp.json"))
b = R["st_C13"]["p"]
if "F1" in which:
    D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
    W = WN.concat([D[k]["W"] for k in SPLIT["comp"]["train"]])
    pr = M.st_rollout(W, b)
    vx = np.maximum(pr["v"], 0.3)
    af = pr["delta"] - np.arctan2(pr["vy"] + M.LF * pr["r"], vx)
    ar = -np.arctan2(pr["vy"] - M.LR * pr["r"], vx)
    a = np.linspace(0, 0.6, 300)
    c = Config().vehicle
    s3 = json.load(open(f"{WORK}/fit_sim3_comp.json"))["over"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, (axle, al, ttl) in enumerate((("f", af, "앞축"), ("r", ar, "뒤축"))):
        x = ax[i]
        lo, hi = np.percentile(np.abs(al), [50, 99.5])
        x.axvspan(0, np.degrees(hi), color="0.9", label=f"재생에서 쓰인 미끄럼각 (99.5 % ≤ {np.degrees(hi):.1f}°)")
        x.plot(np.degrees(a), M.tyre_curve(b, axle, a), "C0", lw=2, label="(b') 채택 구조, 대회장 학습 맞춤")
        if axle == "f":
            nom = c.mu * c.mu_f_scale * np.sin(c.C_f * np.arctan(c.B_f * a)); ad = s3["mu"] * s3["mu_f_scale"] * np.sin(1.3 * np.arctan(s3["B_f"] * a))
        else:
            nom = c.mu * np.sin(c.C_r * np.arctan(c.B_r * a)); ad = s3["mu"] * np.sin(1.3 * np.arctan(s3["B_r"] * a))
        x.plot(np.degrees(a), nom, "C3--", label="시뮬 명목 (기본값)")
        x.plot(np.degrees(a), ad, "C2-.", label="(c2*) 시뮬 프로필")
        x.axhline(9.66 / 9.81, color="k", lw=0.6, ls=":", label="실차가 0.3 s 버틴 최대 횡가속 / g")
        x.set_xlabel("미끄럼각 [°]"); x.set_ylabel("횡력 / 수직하중"); x.set_title(ttl); x.legend(fontsize=7, loc="lower right")
        x.set_ylim(0, 1.6)
    fig.tight_layout(); fig.savefig(f"{FIGS}/F1_tyre_curves.png", dpi=110); plt.close(fig)
    print("F1 alpha p99.5 front/rear deg", np.degrees(np.percentile(np.abs(af), 99.5)), np.degrees(np.percentile(np.abs(ar), 99.5)))
if "F2" in which:
    I = json.load(open(f"{WORK}/ident_comp.json"))
    keys = ["k", "b", "tau_d", "tau_s", "sf", "sr", "mf", "mr"]
    lab = dict(k="조향 이득 k", b="트림 b [rad]", tau_d="순수 지연 τd [s]", tau_s="서보 지연 τs [s]", sf="앞 강성 sf [1/rad]",
               sr="뒤 강성 sr [1/rad]", mf="앞 최대 마찰 μf", mr="뒤 최대 마찰 μr")
    fig, ax = plt.subplots(2, 5, figsize=(19, 7))
    for i, k in enumerate(keys + ["shared_mu"]):
        x = ax.flat[i]
        d = I[k]["profile"] if k == "shared_mu" else I[k]
        pts = d["points"]
        xs = [q["x"] for q in pts]; dj = [1e4 * q["dJ"] for q in pts]; se = [2e4 * q["se"] for q in pts]
        x.errorbar(xs, dj, yerr=se, fmt="o-", ms=3)
        x.axhline(0, color="k", lw=0.5)
        x.axvspan(*d["interval"], color="C2", alpha=0.15)
        x.set_title(f"{lab.get(k, '공통 마찰 μ')}: {d['verdict']}", fontsize=9)
        x.set_ylabel("1 s 요레이트 MSE 증가 ×1e4", fontsize=8)
        if k in ("sf", "sr", "mf", "mr", "shared_mu"):
            x.set_xscale("log")
    S = I["sensitivity"]
    x = ax.flat[9]
    x.bar(list(S), [100 * S[k]["dJ_rel"] for k in S]); x.set_ylabel("학습 오차 변화 [%]"); x.set_title("고정값 민감도 (나머지 재맞춤)", fontsize=9)
    x.tick_params(axis="x", labelsize=7)
    fig.suptitle("(b') 값별 profile likelihood (대회장 학습 bag, 20 s 블록 짝 검정, 막대 = ±2 표준오차, 초록 = 구분 안 되는 범위)")
    fig.tight_layout(); fig.savefig(f"{FIGS}/F2_profiles.png", dpi=100); plt.close(fig)
if "F3" in which:
    T = json.load(open(f"{WORK}/eval_comp_final.json"))
    ms = [m for m in T["all"] if m != "지속"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for x, (sub, key, j, ttl) in zip(ax, (("all", "yaw", 1, "1 s 요레이트 RMS 중앙값 [rad/s]"), ("all", "pos", 2, "2 s 위치 오차 중앙값 [m]"),
                                          ("ay>=7", "yaw", 1, "한계 근처 1 s 요레이트 [rad/s]"))):
        vals = [T[sub][m][key][j] for m in ms]
        cols = ["C1" if m.startswith("(a)") else "C2" if m.startswith("(c2*)") else "C3" if "시뮬" in m and not m.startswith("(c") else "C0" for m in ms]
        x.barh(range(len(ms)), vals, color=cols); x.set_yticks(range(len(ms))); x.set_yticklabels(ms, fontsize=7); x.invert_yaxis()
        x.set_title(ttl, fontsize=10)
        base = T[sub]["(a) 선형 자전거+지연, 감사 값"][key][j]
        x.axvline(base, color="C1", ls="--", lw=0.8)
    fig.suptitle("최종 확인 bag 3개 (874 구간, 한 번만 평가)")
    fig.tight_layout(); fig.savefig(f"{FIGS}/F3_final_check.png", dpi=100); plt.close(fig)
if "F4" in which:
    T = json.load(open(f"{WORK}/transfer.json"))
    names = ["comp->map12x16", "map12x16->map17x18", "comp->map17x18"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    x = ax[0]
    sets = list(T[names[0]]["table"]["all"])
    w = 0.27
    for i, nm in enumerate(names):
        vals = [T[nm]["table"]["all"][s]["yaw"][1] for s in sets]
        x.bar(np.arange(len(sets)) + (i - 1) * w, vals, w, label=nm)
    x.set_xticks(range(len(sets))); x.set_xticklabels(sets, rotation=30, ha="right", fontsize=8)
    x.set_ylabel("새 장소 held-out bag 1 s 요레이트 [rad/s]"); x.legend(fontsize=8); x.set_title("새 장소에서 다시 맞추는 값 (나머지는 원래 장소 값)")
    x = ax[1]
    for k, rows in T["curve"].items():
        x.plot([r["seconds"] for r in rows], [r["yaw1"] for r in rows], "o-", label=k)
    x.axhline(T["comp->map12x16"]["table"]["all"]["없음"]["yaw"][1], color="k", ls="--", lw=0.8, label="다시 맞추지 않음")
    x.axhline(T["comp->map12x16"]["table"]["all"]["전부 8개"]["yaw"][1], color="0.5", ls=":", lw=0.8, label="8개 전부, 전체 데이터")
    x.set_ylim(0, 0.18)
    x.set_xscale("log"); x.set_xlabel("새 장소 주행 시간 [s] (맞춤에 쓴 구간 시작 시각 범위)"); x.set_ylabel("held-out 1 s 요레이트 [rad/s]")
    x.set_title("comp->map12x16: 필요한 주행 시간"); x.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/F4_transfer.png", dpi=100); plt.close(fig)
if "F6" in which:
    X = json.load(open(f"{WORK}/excitation.json"))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    for k in ("mu_ay<6", "mu_6<=ay<8", "mu_ay>=8"):
        if k in X:
            pts = X[k]["points"]
            ax.errorbar([p[0] for p in pts], [1e4 * p[1] for p in pts], yerr=[2e4 * p[2] for p in pts], fmt="o-", ms=3,
                        label=f"{k[3:]} ({X[k]['n']}구간): {X[k]['verdict']} {X[k]['interval'][0]:.2f}~")
    ax.axhline(0, color="k", lw=0.5); ax.set_xscale("log"); ax.set_xlabel("공통 최대 마찰 μ"); ax.set_ylabel("오차 증가 ×1e4 (±2 SE)")
    ax.set_title("마찰은 한계 근처 구간에만 보인다 (대회장 학습 bag, 구간 첫 1 s 최대 |a_y|로 나눔)", fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/F6_excitation.png", dpi=100); plt.close(fig)
print("figs done")
