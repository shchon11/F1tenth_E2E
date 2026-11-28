"""그림: (왼쪽) 안 닿은 통과의 여유 분포, (가운데) 이번 통과 여유별 다음 통과 접촉률(배치 단위 부트스트랩 95 %), (오른쪽) 다음 통과에서
지나는 쪽을 바꿨을 때와 같은 쪽일 때의 접촉률. 인자: events.json cls_passes.json out.png"""
import sys, json
import numpy as np
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
from rec_lib import plt
E = json.load(open(sys.argv[1])); Pr = json.load(open(sys.argv[2]))
rng = np.random.default_rng(0)
def boot(f, cl, n=2000):
    f = np.asarray(f, float); cl = np.asarray(cl); u = np.unique(cl); idx = {c: np.nonzero(cl == c)[0] for c in u}
    o = [f[np.concatenate([idx[c] for c in rng.choice(u, len(u))])].mean() for _ in range(n)]
    return f.mean(), np.percentile(o, 2.5), np.percentile(o, 97.5)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
cols = {"scan31b": "#1f77b4", "hs43b": "#ff7f0e", "scan31b_HO": "#2ca02c"}
names = {"scan31b": "scan31b u475", "hs43b": "hs43b u900", "scan31b_HO": "scan31b u475 · 새 변형 지도"}
for lab, r in E.items():
    cl = np.array([e["clear"] for e in r["enc"] if e["any"] == 0 and e["v"] > 1.0])
    ax[0].hist(100 * cl, bins=np.arange(0, 80, 2), histtype="step", lw=1.6, color=cols[lab], label=f"{names[lab]} (중앙 {100 * np.median(cl):.0f} cm)")
ax[0].axvline(10, color="k", ls=":", lw=1)
ax[0].set_xlabel("안 닿고 지난 통과의 차체-상자 최소 여유 [cm]"); ax[0].set_ylabel("통과 수"); ax[0].legend(fontsize=8)
ax[0].set_title("상자 통과 여유 (이상 실행, 세션 128개씩)")
bins = [0, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.50, 9]; lbl = ["0-3", "3-6", "6-10", "10-15", "15-20", "20-30", "30-50", ">50"]
m = np.array([p["m"] for p in Pr])
for k, key, col, nm in ((0, "nany", "#d62728", "다음 통과에서 닿음(약한 것 포함)"), (1, "nhard", "#7f0000", "다음 통과에서 세게 닿음")):
    xs, ys, lo, hi = [], [], [], []
    for i in range(len(lbl)):
        sel = [p for p in Pr if bins[i] <= p["m"] < bins[i + 1]]
        a = boot([p[key] for p in sel], [p["cl"] for p in sel])
        xs.append(i + 0.12 * (k - 0.5)); ys.append(100 * a[0]); lo.append(100 * (a[0] - a[1])); hi.append(100 * (a[2] - a[0]))
    ax[1].errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=col, capsize=3, label=nm)
ax[1].set_xticks(range(len(lbl))); ax[1].set_xticklabels(lbl); ax[1].set_xlabel("이번 통과 여유 [cm] (안 닿은 통과)")
ax[1].set_ylabel("다음 통과 접촉률 [%]"); ax[1].legend(fontsize=8)
n_ = [sum(bins[i] <= p["m"] < bins[i + 1] for p in Pr) for i in range(len(lbl))]
ax[1].set_title("여유별 다음 바퀴 접촉률 (세 기록 합, 통과 수 " + "/".join(map(str, n_)) + ")", fontsize=9)
fl = [p for p in Pr if p["flip"]]; sm = [p for p in Pr if not p["flip"]]
a1 = boot([p["nany"] for p in fl], [p["cl"] for p in fl]); a2 = boot([p["nany"] for p in sm], [p["cl"] for p in sm])
h1 = boot([p["nhard"] for p in fl], [p["cl"] for p in fl]); h2 = boot([p["nhard"] for p in sm], [p["cl"] for p in sm])
for i, (a, h, nm) in enumerate(((a2, h2, f"같은 쪽\n({len(sm)}통과)"), (a1, h1, f"쪽을 바꿈\n({len(fl)}통과)"))):
    ax[2].bar(i - 0.18, 100 * a[0], 0.36, color="#d62728", yerr=[[100 * (a[0] - a[1])], [100 * (a[2] - a[0])]], capsize=3, label="닿음" if i == 0 else None)
    ax[2].bar(i + 0.18, 100 * h[0], 0.36, color="#7f0000", yerr=[[100 * (h[0] - h[1])], [100 * (h[2] - h[0])]], capsize=3, label="세게" if i == 0 else None)
ax[2].set_xticks([0, 1]); ax[2].set_xticklabels([f"같은 쪽\n({len(sm)}통과)", f"쪽을 바꿈\n({len(fl)}통과)"])
ax[2].set_ylabel("다음 통과 접촉률 [%]"); ax[2].legend(fontsize=8)
ax[2].set_title("다음 바퀴에 같은 상자를 지나는 쪽 (가장 가까운 순간 기준)", fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[3], dpi=110); print("saved", sys.argv[3])
