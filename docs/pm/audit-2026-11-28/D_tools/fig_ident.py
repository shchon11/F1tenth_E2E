import json, numpy as np
from figcommon import plt, COL, FIG
I = json.load(open("ident.json")); K = json.load(open("ident_k2.json")); D = json.load(open("ident_pre_delay.json"))
ven = lambda b: "대회장" if b.startswith("202608") else ("map12x16" if b.startswith("20260725") else "map17x18")
VC = {"대회장": COL["S0"], "map12x16": COL["K2"], "map17x18": COL["S1"]}
fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
# sim-based per-bag estimates
bags = [b for b in I]
def strip(a, vals, title, ylab):
    for b, v in vals:
        a.scatter([list(VC).index(ven(b))], [v], color=VC[ven(b)], s=60, alpha=0.8)
    a.set_xticks(range(3)); a.set_xticklabels(list(VC)); a.set_title(title, fontsize=10); a.set_ylabel(ylab); a.grid(alpha=0.25)
strip(ax[0, 0], [(b, I[b]["best"]["steer_gain"]) for b in bags], "시뮬로 맞춘 조향 이득 (bag별)", "steer_gain")
dl = []
for b in bags:
    d = I[b]["best"]["cmd_delay"]
    if b in D: d = min(D[b], key=lambda x: x[1])[0]
    dl.append((b, d))
strip(ax[0, 1], dl, "시뮬로 맞춘 명령 지연 (bag별, 서보 0.04 별도)", "cmd_delay [s]")
a = ax[0, 2]
for b in bags:
    mu = [float(k) for k in I[b]["mu_rmse"]]; e = np.array(list(I[b]["mu_rmse"].values()))
    a.plot(mu, e / e.min(), "-o", color=VC[ven(b)], alpha=0.7, lw=1.6, ms=4)
a.set_xlabel("시뮬 마찰 mu"); a.set_ylabel("1초 요레이트 RMSE / 최솟값"); a.set_ylim(0.9, 2.2)
a.set_title("마찰 프로필: 대회장은 격자 끝(1.5)까지 계속 내려감,\n사전 노면은 0.9 이상 평평 = 하한만", fontsize=10); a.grid(alpha=0.25)
kb = [b for b in K]
strip(ax[1, 0], [(b, K[b]["tau"]) for b in kb], "선형 자전거로 맞춘 총 지연 (bag별)", "tau [s]")
strip(ax[1, 1], [(b, K[b]["speed"]["a_b"]) for b in kb], "선형 자전거로 맞춘 제동 한계 (bag별)", "a_brake [m/s²]")
a = ax[1, 2]
ky = lambda b, v: K[b]["k"] * 1.0 / (0.3302 + K[b]["kus"] * v * v)
for b in kb:
    a.scatter([list(VC).index(ven(b))], [ky(b, 5.0) * 0.3302], color=VC[ven(b)], s=60, alpha=0.8)
a.set_xticks(range(3)); a.set_xticklabels(list(VC)); a.grid(alpha=0.25)
a.set_title("선형 자전거 요 이득 (5 m/s, 운동학 대비 비)", fontsize=10); a.set_ylabel("k·L/(L+kus·v²)")
for c, n in VC.items(): ax[0, 0].scatter([], [], color=n, label=c)
ax[0, 0].legend(fontsize=9)
fig.suptitle("bag별 식별 재현성: 같은 장소면 같은 값이 나와야 한다", fontsize=13)
fig.tight_layout(); fig.savefig(FIG + "D5_per_bag_identification.png", dpi=105)
print("ok")
