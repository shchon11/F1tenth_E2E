"""plan 출력 자체가 흔들리나 (11-10, 사용자: "결국 planner 출력 자체가 흔들리게 나온다는거야?"). console_wobble.py가 F1SIM_WOBBLE_NPZ로
저장한 같은 차·같은 출발의 실행들을 줄마다 하나씩: policy가 낸 plan의 첫 조향 명령(결정마다 새로, 다음 결정까지 그대로)과 차의 실제
조향각, 앞 10 s. 줄 제목에 30 s 전체의 plan 조향 흔들림·지그재그, 실제 조향 흔들림, 요 각속도 떨림(console_wobble과 같은 식).
인자: 출력png 이름=npz 이름=npz ..."""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break

out = sys.argv[1]
runs = [a.split("=", 1) for a in sys.argv[2:]]
PLAN, CAR = "#2a78d6", "#eb6834"                                  # plan = slot 1, the car = slot 2, in every row


def jitter(x):
    d = np.diff(x)
    return float(np.mean(np.abs(d))), float(np.mean((d[1:] * d[:-1] < 0) & (np.abs(d[1:]) > 0.5) & (np.abs(d[:-1]) > 0.5)))


fig, axs = plt.subplots(len(runs), 1, figsize=(14, 2.6 * len(runs)), sharex=True, sharey=True)
axs = np.atleast_1d(axs)
for ax, (label, path) in zip(axs, runs):
    z = np.load(path)
    sd, yr, pc = z["sd"], z["yr"], z["pc"]
    pj, pz = jitter(pc[::4])
    aj, az = jitter(sd[::4])
    yrs = np.convolve(yr, np.ones(40) / 40, "same")
    wob = float(np.std((yr - yrs)[20:-20]))
    n = min(len(sd), 400)                                          # the first 10 s
    t = 2.0 + np.arange(n) * 0.025
    ax.plot(t, pc[:n], color=PLAN, lw=1.6, drawstyle="steps-post", label="policy가 낸 plan의 조향 명령")
    ax.plot(t, sd[:n], color=CAR, lw=1.3, label="차의 실제 조향각")
    ax.set_title(f"{label}:  plan 조향 흔들림 {pj:.2f}°/결정 · 지그재그 {pz * 100:.0f} %   |   실제 조향 흔들림 {aj:.2f}°/0.1 s · "
                 f"지그재그 {az * 100:.0f} %   |   요 각속도 떨림 {wob:.0f}°/s", fontsize=10.5, loc="left")
    ax.grid(alpha=0.25, lw=0.6); ax.set_ylabel("조향 [deg]")
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
axs[0].legend(loc="upper right", fontsize=9, frameon=False, ncol=2)
axs[-1].set_xlabel("시간 [s] (혼자 빈 트랙, 같은 출발, 콘솔 경로)")
fig.suptitle("plan 출력 자체가 흔들리나 -- 같은 policy(plan25n5 u300), 실행 방식만 다르게", fontsize=12, x=0.01, ha="left")
plt.tight_layout(); plt.savefig(out, dpi=80)
print(out, flush=True)
