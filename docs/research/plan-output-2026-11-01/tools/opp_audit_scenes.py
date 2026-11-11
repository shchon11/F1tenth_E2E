"""상대 차 주행 점검 그림: 기록마다(순수 주행 / 학습 세계의 끼어들기·급제동 포함 / 선만 따라가는 차 / FTG / Disparity) 종류별로
'보통' 차 한 대(평균 속도가 그 종류의 중앙값인 차)가 16 s 동안 달린 길. 색 = 속도(모든 칸 같은 눈금), 회색 가는 선 = 같은 경주의
우리 차, 검은 x = 벽에 닿음, 빨간 x = 차끼리 닿음. 마지막 칸 = 우리 차(같은 방식).
인자: 출력png 기록npz:제목[:종류 번호들, 쉼표] ..."""
import sys
import numpy as np
from rec_lib import Rec, plt
from matplotlib.collections import LineCollection

out = sys.argv[1]; jobs = []
for a in sys.argv[2:]:
    p = a.split(":")
    jobs.append((p[0], p[1], [int(x) for x in p[2].split(",")] if len(p) > 2 else None))
T0, T1 = 8.0, 24.0
cells = []
for fn, title, ks in jobs:
    rc = Rec(fn); z = rc.z; T, dt = rc.T, rc.dt
    on = z["on"][:T]; kind = z["kind"][:T]
    for k in (ks if ks is not None else [None]):
        m = on.all(0) if k is None else (on & (kind == k)).all(0)
        cars = np.nonzero(m)[0]
        if len(cars) == 0:
            continue
        v = np.abs(z["op"][:T, cars, 3]).mean(0)
        b = int(cars[np.argsort(v)[len(v) // 2]])
        cells.append((rc, b, "op", f"{title}" + (f" · {rc.kinds[k]}" if k is not None else ""), len(cars)))
rc0 = Rec(jobs[0][0]); v = np.abs(rc0.z["st"][:rc0.T, :, 3]).mean(0)
cells.append((rc0, int(np.argsort(v)[len(v) // 2]), "st", f"우리 차 ({rc0.name.split('/')[0]})", rc0.B))

n = len(cells); cols = 4; rows = (n + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.3 * rows)); axs = np.atleast_2d(axs)
lc = None
for i, (rc, b, key, title, ncar) in enumerate(cells):
    ax = axs[i // cols, i % cols]; z = rc.z; dt = rc.dt
    a, e = int(T0 / dt), int(T1 / dt)
    rc.draw_map(ax, int(z["tid"][a, b]))
    s = z[key][a:e + 1, b].astype(float); c = z["ocnt" if key == "op" else "cnt"][a:e, b]
    ok = np.linalg.norm(s[1:, :2] - s[:-1, :2], axis=1) < 2.0
    if key == "op":
        o = z["st"][a:e + 1, b].astype(float); oo = np.linalg.norm(o[1:, :2] - o[:-1, :2], axis=1) < 2.0
        for j in np.nonzero(oo)[0]:
            ax.plot(o[j:j + 2, 0], o[j:j + 2, 1], color="#999999", lw=0.6, zorder=1)
    pts = np.stack([s[:-1, :2], s[1:, :2]], 1)[ok]
    lc = LineCollection(pts, cmap="viridis", norm=plt.Normalize(0, 8), lw=1.8, zorder=2)
    lc.set_array(np.abs(s[:-1, 3])[ok]); ax.add_collection(lc)
    d = np.clip(np.diff(c, axis=0), 0, None) * ok[1:len(c), None]
    for col, colr in ((0, "k"), (2, "r")):
        h = np.nonzero(d[:, col] > 0)[0] + 1
        ax.plot(s[h, 0], s[h, 1], "x", color=colr, ms=7, mew=2, zorder=4)
    vv = np.abs(s[:-1, 3])[ok]
    ax.set_title(f"{title}\n차 {b} ({ncar}대 중 보통), 평균 {vv.mean():.1f} · 최고 {vv.max():.1f} · 최저 {vv.min():.1f} m/s", fontsize=9)
    ax.set_xlim(s[:, 0].min() - 1.5, s[:, 0].max() + 1.5); ax.set_ylim(s[:, 1].min() - 1.5, s[:, 1].max() + 1.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for i in range(n, rows * cols):
    axs[i // cols, i % cols].axis("off")
cb = fig.colorbar(lc, ax=axs, fraction=0.012, pad=0.01); cb.set_label("속도 [m/s]")
fig.suptitle(f"상대 차가 달린 길 ({T0:.0f}~{T1:.0f} s, 상자 없음, 원래 트랙). 색 = 속도, 회색 = 같은 경주의 우리 차, 검은 x = 벽 접촉, 빨간 x = 차 접촉", fontsize=11)
fig.savefig(out, dpi=105, bbox_inches="tight"); print("저장", out)
