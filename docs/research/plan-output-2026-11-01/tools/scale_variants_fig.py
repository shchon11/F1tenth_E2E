"""크기를 바꾼 트랙 변형을 같은 축척으로 나란히 (11-12 사용자: "트랙 변형도 스케일 자체가 변하는것도").
원본 + 학습 세계의 크기 변형(state_rl.TRACK_SCALES_25, 뒤튼 변형 하나씩에 크기를 곱한 것)을 크기 순서로, 모든 칸을 같은 m/픽셀로 그림.
칸마다: 배율, 경기장의 긴 변 x 짧은 변(달리는 구역의 크기), 길 폭(달리는 구역 한가운데에서 가까운 벽까지 거리 x 2의 중앙값). 차(0.58 x 0.31 m)와 1 m
막대도 같은 축척. 아래 줄: 기존 뒤튼 변형 16개의 크기 범위와 실제 대회장 폭.
인자: 그림 png"""
import os, sys
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rec_lib  # noqa: F401  (한글 글꼴)
from f1sim import maps
from f1sim.learn import state_rl as SR

BASE = "real:korea_2025_iccas"
out = sys.argv[1]


def measure(t):
    """driving region (largest free blob); the venue's long and short side (extent of the region along its own principal
    axes) and the lane width = twice the distance to the nearest wall on the region's middle ridge (median: the ridge also
    runs through the gaps between the inner walls, so no minimum is quoted)"""
    occ = np.asarray(t.occupancy) > 0
    lab, n = ndi.label(~occ)
    big = lab == (1 + int(np.argmax(ndi.sum(~occ, lab, range(1, n + 1)))))
    edt = ndi.distance_transform_edt(~occ) * t.resolution
    w = 2 * edt[skeletonize(big)]
    ys, xs = np.nonzero(big)
    p = np.stack([xs, ys], 1) * t.resolution
    p = p - p.mean(0)
    _, vec = np.linalg.eigh(np.cov(p[::7].T))
    q = p @ vec
    side = np.sort(q.max(0) - q.min(0))[::-1]
    return big, occ, float(side[0]), float(side[1]), float(np.median(w))


RES0 = float(maps.load(BASE).resolution)
VENUES = (("ICRA 2022", "real:icra2022"), ("한국 2026", "real:korea_2026_competition"), ("map16x07", "real:map16x07"),
          ("TU Wien 2021", "real:blackbox2021_1"))
venue_w = []
for label, nm in VENUES:
    try:
        venue_w.append((label, measure(maps.load(nm))[4]))
        print(f"실제 대회장 {label:<14} 길 폭(중앙값) {venue_w[-1][1]:.2f} m", flush=True)
    except Exception as e:                                             # a venue whose files are not on this machine
        print(f"실제 대회장 {label}: 못 읽음 ({type(e).__name__})", flush=True)
names = [BASE] + SR.scale_variants([BASE], 16, SR.TRACK_SCALES_25)
items = []
for nm in names:
    t = maps.load(nm) if hasattr(maps, "load") else maps._load_base(nm)
    sc = float(t.resolution) / RES0                                   # the size actually applied (the warp's own few % included)
    big, occ, la, lb, wmed = measure(t)
    items.append(dict(name=nm, t=t, sc=sc, big=big, occ=occ, la=la, lb=lb, wmed=wmed))
    print(f"{nm:<44} x{sc:.2f} | 격자 한 칸 {t.resolution * 100:.1f} cm | 경기장 {la:4.1f} x {lb:4.1f} m | 길 폭(중앙값) {wmed:.2f} m", flush=True)
items.sort(key=lambda d: d["sc"])
half = 0.0                                                             # half the side every panel shows [m]
for d in items:
    ys, xs = np.nonzero(d["big"]); res = d["t"].resolution
    d["cx"] = d["t"].origin[0] + (xs.min() + xs.max() + 1) / 2 * res; d["cy"] = d["t"].origin[1] + (ys.min() + ys.max() + 1) / 2 * res
    half = max(half, (xs.max() - xs.min() + 1) * res / 2, (ys.max() - ys.min() + 1) * res / 2)
half *= 1.05
GREY = (0.90, 0.90, 0.88)
n = len(items); ncol = 6; nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.55 * nrow + 1.0), facecolor="white")
for ax in axes.ravel():
    ax.axis("off")
for ax, d in zip(axes.ravel(), items):
    t = d["t"]; res = t.resolution; H, W = d["occ"].shape
    img = np.ones((H, W, 3)); img[~d["big"]] = GREY
    wall = d["occ"] & ndi.binary_dilation(d["big"], structure=np.ones((3, 3), bool), iterations=2)
    img[wall] = (0.15, 0.15, 0.17)
    ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor(GREY)
    orig = d["name"] == BASE
    for s_ in ax.spines.values():
        s_.set_color("#c0392b" if orig else "#bbbbbb"); s_.set_linewidth(2.0 if orig else 0.8)
    ax.imshow(img, origin="lower", extent=(t.origin[0], t.origin[0] + W * res, t.origin[1], t.origin[1] + H * res), interpolation="nearest")
    ax.set_xlim(d["cx"] - half, d["cx"] + half); ax.set_ylim(d["cy"] - half, d["cy"] + half)
    ax.set_aspect("equal")
    x0, y0 = d["cx"] - half * 0.93, d["cy"] - half * 0.90
    ax.plot([x0, x0 + 1.0], [y0, y0], color="#222", lw=2.5, solid_capstyle="butt")
    ax.text(x0 + 1.3, y0, "1 m", va="center", fontsize=8, color="#222")
    ax.add_patch(Rectangle((x0, y0 + 0.75), 0.58, 0.31, color="#c0392b", lw=0))
    ax.text(x0 + 1.3, y0 + 0.9, "차", va="center", fontsize=8, color="#c0392b")
    ax.set_title(("원본  " if orig else "") + f"x{d['sc']:.2f}  |  경기장 {d['la']:.1f} x {d['lb']:.1f} m  |  길 폭 {d['wmed']:.1f} m",
                 fontsize=9.5, color="#c0392b" if orig else "#111", fontweight="bold" if orig else "normal")
w0 = [d["wmed"] for d in items if d["name"] == BASE][0]
scs = [d["sc"] for d in items if d["name"] != BASE]
fig.suptitle(f"학습 세계에 더한 크기 변형 16개 + 원본 (모두 같은 축척, 크기 순서). 기존 뒤튼 변형 16개는 크기가 0.96~1.04배뿐이었음", fontsize=13, y=0.995)
fig.text(0.5, 0.012, "같은 방법으로 잰 실제 대회장의 길 폭(중앙값): " + ", ".join(f"{l} {w:.1f} m" for l, w in venue_w) + f", 한국 2025 {w0:.1f} m"
         f"   |   크기 변형 {min(scs):.2f}~{max(scs):.2f}배 (0.96~1.04배는 기존 변형이 채움). 벽 두께도 같이 변함. 상자와 차는 실제 크기 그대로.",
         ha="center", fontsize=10, color="#333")
fig.tight_layout(rect=(0, 0.035, 1, 0.965), h_pad=1.6)
fig.savefig(out, dpi=110)
print("저장", out)
