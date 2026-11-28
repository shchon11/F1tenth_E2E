import pickle, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import models as M
from windows import WARM
fp = [f for f in font_manager.findSystemFonts() if "NanumGothic" in f or "NotoSansCJK" in f or "Noto Sans CJK" in f]
if fp:
    font_manager.fontManager.addfont(fp[0]); plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp[0]).get_name()
plt.rcParams["axes.unicode_minus"] = False
C = dict(real="#222222", S0="#2a78d6", K2="#e87ba4", S1="#1baf7a")
Wa = pickle.load(open("windows.pkl","rb")); P = pickle.load(open("preds.pkl","rb"))
W = Wa["val"]; s0 = P["val"]["S0"]; k2 = P["val"]["K2"]
e = M.errors(W, s0)[100]["yaw"]
order = np.argsort(-e)
# pick 3 bad windows from different bags/time + 1 typical window
pick, seen = [], []
for i in order:
    if all(abs(W["t0"][i] - W["t0"][j]) > 3 or W["bag"][i] != W["bag"][j] for j in pick):
        pick.append(i)
    if len(pick) == 3: break
typ = np.argsort(np.abs(e - np.median(e)))[0]
pick.append(typ)
t = np.arange(201) / 100
fig, ax = plt.subplots(3, 4, figsize=(16, 9.5), sharex="col")
for c, i in enumerate(pick):
    a = ax[0, c]
    a.plot(t, W["r"][i, WARM:], color=C["real"], lw=2, label="실차 자이로")
    a.plot(t, s0["r"][i], color=C["S0"], lw=2, label="시뮬 (명목)")
    a.plot(t, k2["r"][i], color=C["K2"], lw=2, ls="--", label="선형 자전거+지연 (K2)")
    a.set_title(f"{W['bag'][i]} t={W['t0'][i]:.1f}s  v0={W['v'][i, WARM]:.1f} m/s\n1초 요레이트 RMS: 시뮬 {e[i]:.2f}" + ("  (전형적 구간)" if c == 3 else ""), fontsize=10)
    a.set_ylabel("요레이트 [rad/s]" if c == 0 else "")
    b = ax[1, c]
    b.plot(t, W["ay"][i, WARM:], color=C["real"], lw=2); b.plot(t, s0["ay"][i], color=C["S0"], lw=2); b.plot(t, k2["ay"][i], color=C["K2"], lw=2, ls="--")
    b.set_ylabel("횡가속 [m/s²]" if c == 0 else "")
    d = ax[2, c]
    beta = np.degrees(np.arctan2(s0["vy"][i], np.maximum(np.abs(s0["v"][i]), 0.3)))
    d.plot(t, beta, color=C["S0"], lw=2, label="시뮬 차체 미끄럼각")
    d2 = d.twinx(); d2.plot(t, W["dcmd"][i, WARM:], color="#888", lw=1.2, label="조향 명령(적용값)"); d2.set_ylim(-0.5, 0.5)
    d.set_ylabel("시뮬 미끄럼각 [deg]" if c == 0 else ""); d2.set_ylabel("조향 [rad]" if c == 3 else "")
    d.set_xlabel("예측 시간 [s]")
    for x in (a, b, d): x.grid(alpha=0.25)
ax[0, 0].legend(fontsize=9, loc="best")
ax[2, 0].legend(loc="upper left", fontsize=9)
fig.suptitle("개방 루프 재생 예: 시뮬이 크게 틀리는 구간(왼쪽 3개)과 전형적 구간(오른쪽) — 대회장 확인용 bag", fontsize=12)
fig.tight_layout()
fig.savefig("/home/shchon11/F1tenth_E2E/docs/pm/audit-2026-11-28/D_figs/D3_examples.png", dpi=110)
print(pick, e[pick])
