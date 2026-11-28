"""매끄러운 벽 전후 비교(물리·LiDAR 쪽): 같은 코스의 5 cm 격자 지도(전)와 `+smooth` 지도(후)에서, 같은 차 자세로
LiDAR 빔 1081개를 쏘아 맞은 점을 지도 위에 그린다. 시뮬의 LiDAR처럼 점유 격자를 따라 빔을 진행한다(0.5 cm 걸음).
곧은 벽 한 토막의 점이 직선에서 얼마나 벗어나는지(표준편차)도 잰다.

  .venv/bin/python docs/pm/sim-m1-2026-11-28/tools/wall_compare.py <출력 PNG>
"""
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from f1sim import maps

for f in font_manager.findSystemFonts():
    if "NanumGothic" in f or "NotoSansCJK" in f or "IBMPlexSansKR" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = [n.name for n in font_manager.fontManager.ttflist
                               if any(k in n.name for k in ("Nanum", "Noto Sans CJK", "IBM Plex Sans KR"))][:1] or ["DejaVu Sans"]


def cast(tr, x, y, yaw, n=1081, fov=math.radians(270), rmax=4.0, step=0.005):
    occ = np.asarray(tr.occupancy, bool)
    res = float(tr.resolution)
    a = yaw - fov / 2 + np.arange(n) * fov / (n - 1)
    r = np.arange(step, rmax, step)
    px = x + np.cos(a)[:, None] * r[None]
    py = y + np.sin(a)[:, None] * r[None]
    j = np.floor((px - tr.origin[0]) / res).astype(int)
    i = np.floor((py - tr.origin[1]) / res).astype(int)
    ok = (i >= 0) & (i < occ.shape[0]) & (j >= 0) & (j < occ.shape[1])
    hit = np.zeros_like(ok)
    hit[ok] = occ[i[ok], j[ok]]
    first = np.where(hit.any(1), hit.argmax(1), -1)
    sel = first >= 0
    return px[sel, first[sel]], py[sel, first[sel]]


def line_std(xs, ys):
    """벽 토막의 점이 매끄러운 곡선(2차)에서 벗어난 표준편차: 휘어짐은 빼고 계단·잔물결만 남긴다."""
    P = np.stack([xs, ys], 1)
    c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c)
    a, b = (P - c) @ vt[0], (P - c) @ vt[1]
    fit = np.polyval(np.polyfit(a, b, 2), a)
    return float(np.std(b - fit))


def main(out):
    before = maps.load("real:korea26_scan+duct")
    after = maps.load("real:korea26_scan+duct+smooth")
    cl = np.asarray(after.centerline, float)
    k = 330
    x, y = cl[k]
    t = cl[k + 2] - cl[k - 2]
    yaw = math.atan2(t[1], t[0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), dpi=110)
    stats = []
    for ax, tr, title in zip(axes, (before, after), ("전: 5 cm 격자(계단)", "후: 매끄러운 판(+smooth, 1.25 cm 격자)")):
        occ = np.asarray(tr.occupancy, bool)
        res = float(tr.resolution)
        ext = [tr.origin[0], tr.origin[0] + occ.shape[1] * res, tr.origin[1], tr.origin[1] + occ.shape[0] * res]
        ax.imshow(~occ, origin="lower", extent=ext, cmap="gray", vmin=-0.4, vmax=1.2, interpolation="nearest")
        hx, hy = cast(tr, x, y, yaw)
        ax.scatter(hx, hy, s=4, c="#1aa6c0", zorder=3)
        ax.annotate("", xy=(x + 0.25 * math.cos(yaw), y + 0.25 * math.sin(yaw)), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color="#7a9a10", lw=2.5), zorder=4)
        ax.set_xlim(x - 1.6, x + 1.6)
        ax.set_ylim(y - 1.6, y + 1.6)
        ax.set_title(title, fontsize=13)
        ax.set_aspect("equal")
        # 비스듬한 벽 한 토막(왼쪽 위 벽, x -0.9..-0.2, y 8.0..8.9): 2차 곡선에서 벗어난 정도
        m = (hx > -0.9) & (hx < -0.2) & (hy > 8.0) & (hy < 8.9)
        sd = line_std(hx[m], hy[m]) if m.sum() > 10 else float("nan")
        stats.append(sd)
        ax.plot(hx[m], hy[m], "o", ms=3, color="#c6f135", zorder=5)
        ax.text(0.02, 0.02, f"비스듬한 벽(연두) 점 {int(m.sum())}개, 매끄러운 곡선에서 벗어남 표준편차 {sd * 1000:.1f} mm",
                transform=ax.transAxes, fontsize=11, color="black", backgroundcolor="white")
    fig.suptitle("같은 자세에서 쏜 LiDAR 빔이 맞은 점 (2026 한국 대회장, 같은 확대, 화살표 = 차)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out)
    print({"before_std_mm": stats[0] * 1000, "after_std_mm": stats[1] * 1000, "pose": [x, y, yaw]})


if __name__ == "__main__":
    main(sys.argv[1])
