"""포팅본이 자기 선에서 어디서 얼마나 벗어나는가 (aur_track_where.py, 11-12). 제어기를 견주는 값 = 받은 경로와의 옆 거리이고
한 바퀴 시간·접촉이 아님(10-20 사용자 규칙). 기준이 고정된 기록(빈 트랙, 장애물 점 없음 -> 늘 자기 선)에 씀.
표: 기록 x 속도 배율마다 옆 거리 RMSE · 99 % · 굽은 곳에서 안쪽으로 치우친 평균(+ = 안쪽으로 파고듦, - = 바깥으로 밀림),
    선의 굽은 정도(곧음 |κ|<0.15 / 완만 0.15-0.5 / 급함 >0.5 [1/m])마다 따로. 회피 중·선을 잃음·다시 출발 뒤 3초는 뺌.
그림: 기록(줄) x 속도 배율(칸)마다 달린 자리를 옆 거리로 색칠(파랑 = 선의 왼쪽, 빨강 = 오른쪽, 회색 = 선 위; ±0.3 m).
인자: 출력png 기록npz [기록npz ...]"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rec_lib as RL

DIV = LinearSegmentedColormap.from_list("d", ["#1f5fa8", "#8fb4dc", "#d9d7d0", "#e8a08f", "#b3261e"])
SKIP = 30                                                             # decisions after the car was put somewhere
BINS = (("곧음", 0.0, 0.15), ("완만", 0.15, 0.5), ("급함", 0.5, 1e9))


def label(z):
    cfg = str(z["aur_cfg"]) if "aur_cfg" in z.files else ""
    how = "MAP 제어기" if '"map"' in cfg else "pure pursuit(버린 옵션)" if '"race"' in cfg else "원본(겨눈 점 조향)"
    return how + ("" if "calibrated" in z.files else ", 차 보정 없음")


out_png, files = sys.argv[1], sys.argv[2:]
recs = []
for fn in files:
    r = RL.Rec(fn); z = r.z
    T, B = r.T, r.B
    st = r.st.astype(float)
    off = z["off"][:T].astype(float); kap = z["kap"][:T].astype(float)
    ps = z["pscale"][:T]; scales = [float(x) for x in z["scales"]]
    rs = z["rs"][:T] | z["stuck"][:T]
    jump = np.zeros((T, B), bool); jump[1:] = np.hypot(*(st[1:T, :, :2] - st[:T - 1, :, :2]).transpose(2, 0, 1)) > 1.5
    new = jump | np.concatenate([np.ones((1, B), bool), rs[:-1]], 0)
    since = np.zeros((T, B), int)
    for t in range(T):
        since[t] = np.where(new[t], 0, since[t - 1] + 1 if t else 0)
    ok = (since >= SKIP) & ~z["avoid"][:T] & ~z["lost"][:T]
    inward = off * np.sign(kap)                                       # + = toward the inside of the bend
    name = os.path.basename(fn).replace(".npz", "")
    print(f"== {name}: {label(z)} · 차 {B}대 x {T * r.dt:.0f} s · 회피 중이던 결정 {100 * z['avoid'][:T].mean():.1f} % ==")
    print("속도 배율 | 옆 거리 RMSE · 99 % (결정 수) | " + " | ".join(f"{b[0]}: RMSE · 안쪽으로 치우친 평균" for b in BINS))
    for sc in scales:
        m = ok & (np.abs(ps - sc) < 1e-3)
        if m.sum() < 50:
            continue
        line = f"{sc:g}배 | {np.sqrt((off[m] ** 2).mean()):.3f} m · {np.percentile(np.abs(off[m]), 99):.2f} m ({int(m.sum())})"
        for _, lo, hi in BINS:
            mb = m & (np.abs(kap) >= lo) & (np.abs(kap) < hi)
            line += (f" | {np.sqrt((off[mb] ** 2).mean()):.3f} m · {inward[mb].mean():+.3f} m" if mb.sum() >= 30 else " | -")
        print(line)
    recs.append((r, name, label(z), st, off, ps, scales, ok))

nr, nc = len(recs), max(len(x[6]) for x in recs)
fig, axes = plt.subplots(nr, nc, figsize=(5.4 * nc, 5.0 * nr), squeeze=False, constrained_layout=True)
last = None
for i, (r, name, lab, st, off, ps, scales, ok) in enumerate(recs):
    for j in range(nc):
        ax = axes[i, j]
        if j >= len(scales):
            ax.axis("off"); continue
        sc = scales[j]; m = ok & (np.abs(ps - sc) < 1e-3)
        r.draw_map(ax, int(r.tid[0, 0]))
        last = ax.scatter(st[:r.T][m][:, 0], st[:r.T][m][:, 1], c=off[m], cmap=DIV, vmin=-0.3, vmax=0.3, s=2.2, lw=0, zorder=4)
        rm = np.sqrt((off[m] ** 2).mean()) if m.sum() else float("nan")
        ax.set_title(f"{lab} · 선 속도의 {sc:g}배\n옆 거리 RMSE {rm:.3f} m · 99 % {np.percentile(np.abs(off[m]), 99) if m.sum() else float('nan'):.2f} m",
                     fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(last, ax=list(axes.ravel()), fraction=0.015, pad=0.01, label="자기 선과의 옆 거리 [m] (파랑 = 선의 왼쪽, 빨강 = 오른쪽)")
fig.suptitle("빈 트랙에서 자기 선을 따라갈 때 선과의 옆 거리 (회피 중·다시 출발 뒤 3초 제외)", fontsize=12, x=0.01, ha="left")
fig.savefig(out_png, dpi=100); plt.close(fig)
print(f"그림: {out_png}")
