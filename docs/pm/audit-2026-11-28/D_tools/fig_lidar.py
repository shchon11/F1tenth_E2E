import json, numpy as np
from figcommon import plt, COL, FIG
ex = json.load(open("lidar_examples.json")); R = json.load(open("lidar_cmp.json"))
ang = np.linspace(-2.35619449, 2.35619449, 1081)
fig = plt.figure(figsize=(16, 9.5))
for k, e in enumerate(ex[:2]):
    a = fig.add_subplot(2, 3, 1 + k)
    for key, col, lab, ms in (("real", COL["real"], "실제 /scan", 3), ("sim", COL["S0"], "시뮬 (같은 자세, 대회장 지도)", 2)):
        r = np.array(e[key]); ok = (r > 0.02) & (r < 29.9)
        a.scatter(r[ok] * np.cos(ang[ok]), r[ok] * np.sin(ang[ok]), s=ms, color=col, label=lab, alpha=0.8)
    a.plot([0], [0], "k^", ms=8); a.set_aspect("equal"); a.set_xlim(-8, 16); a.set_ylim(-9, 9)
    a.set_title(f"20260827-115713 t={e['t']:.1f}s  v={e['v']:.1f} m/s (라이다 좌표)", fontsize=10); a.grid(alpha=0.25)
    if k == 0: a.legend(fontsize=9, loc="lower left", markerscale=4)
# residual / validity summary per bag
a = fig.add_subplot(2, 3, 3)
bags = list(R)
x = np.arange(len(bags))
a.bar(x - 0.2, [R[b]["within_10cm"] for b in bags], 0.4, color=COL["S0"], label="|실제-시뮬| < 10 cm")
a.bar(x + 0.2, [R[b]["real_shorter_50cm"] for b in bags], 0.4, color=COL["K2"], label="실제가 50 cm+ 짧음 (지도에 없는 물체)")
a.set_xticks(x); a.set_xticklabels([b[:8] + "\n" + b[9:] for b in bags], fontsize=9); a.set_ylim(0, 1)
for i, b in enumerate(bags):
    a.text(i - 0.2, R[b]["within_10cm"] + 0.02, f"{R[b]['within_10cm']:.2f}", ha="center", fontsize=9)
    a.text(i + 0.2, R[b]["real_shorter_50cm"] + 0.02, f"{R[b]['real_shorter_50cm']:.2f}", ha="center", fontsize=9)
a.set_title("빔별 거리 일치 (PF 자세 기준, 지도 5 cm 격자)", fontsize=10); a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.25, axis="y")
# no-return vs range
a = fig.add_subplot(2, 3, 4)
for j, b in enumerate(bags):
    br = R[b]["by_range"]
    xs = np.array([(r[0] + r[1]) / 2 for r in br])
    a.plot(xs, [100 * r[3] for r in br], "-o", color=COL["real"], alpha=0.5 + 0.2 * j, lw=2, label="실제" if j == 0 else None)
    a.plot(xs, [100 * r[4] for r in br], "-s", color=COL["S0"], alpha=0.5 + 0.2 * j, lw=2, label="시뮬" if j == 0 else None)
a.set_xlabel("예측 거리 (덕트 벽) [m]"); a.set_ylabel("무반사 비율 [%]"); a.set_title("무반사 vs 거리: 실제는 거리에 따라 증가, 시뮬은 평평", fontsize=10)
a.legend(fontsize=9); a.grid(alpha=0.25)
a = fig.add_subplot(2, 3, 5)
for j, b in enumerate(bags):
    bc = R[b]["noreturn_by_cos"]
    xs = np.array([(r[0] + r[1]) / 2 for r in bc])
    a.plot(xs, [100 * r[3] for r in bc], "-o", color=COL["real"], alpha=0.5 + 0.2 * j, lw=2, label="실제" if j == 0 else None)
    a.plot(xs, [100 * r[4] for r in bc], "-s", color=COL["S0"], alpha=0.5 + 0.2 * j, lw=2, label="시뮬" if j == 0 else None)
a.set_xlabel("|cos 입사각| (0 = 스치듯)"); a.set_ylabel("무반사 비율 [%]"); a.set_title("무반사 vs 입사각: 시뮬 스침 무반사 2-3배 과다", fontsize=10)
a.legend(fontsize=9); a.grid(alpha=0.25)
a = fig.add_subplot(2, 3, 6)
for j, b in enumerate(bags):
    sn = R[b].get("static_noise", [])
    xs = np.array([(r[0] + r[1]) / 2 for r in sn])
    sn = [r for r in sn if r[1] <= 10]
    xs = np.array([(r[0] + r[1]) / 2 for r in sn])
    a.plot(xs, [1000 * 1.4826 * r[4] for r in sn], "-o", color=COL["real"], alpha=0.5 + 0.2 * j, lw=2, label="실제 빔별 1.48×MAD (정지, 중앙값)" if j == 0 else None)
xs = np.linspace(0.5, 8, 20)
a.plot(xs, 1000 * (0.0074 + 0.001 * xs), "--", color=COL["S0"], lw=2, label="시뮬 잡음 σ = 7.4 mm + 0.1 %")
a.set_xlabel("거리 [m]"); a.set_ylabel("거리 잡음 [mm]"); a.set_title("정지 시 거리 잡음", fontsize=10); a.legend(fontsize=9); a.grid(alpha=0.25)
fig.suptitle("LiDAR: 실제 스캔 vs 같은 자세에서 시뮬이 그린 스캔 (대회장 bag 3개, 10 Hz 표본)", fontsize=13)
fig.tight_layout(); fig.savefig(FIG + "D4_lidar.png", dpi=105)
print("ok")
