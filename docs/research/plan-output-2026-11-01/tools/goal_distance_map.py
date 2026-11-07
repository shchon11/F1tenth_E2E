"""결승선까지 남은 최단 거리 그림 (11-07 17:00, 사용자: "결승선까지 남은 최단 거리가 어떻게 되는건데 시각화좀"). 차 중심이 갈 수 있는
빈 공간(벽에서 차 반폭 0.16 m 넘게 떨어진 곳)에서, 결승선(중앙선 첫 점을 지나 진행 방향에 수직인 선분)을 거꾸로 넘지 않고 결승선까지
가는 가장 짧은 거리를 모든 칸에 계산(격자 최단 경로, 16방향 이동). 왼쪽: 그 거리 + 2 m마다 같은 거리 선 + 거리가 줄어드는 방향 화살표.
오른쪽: 지금 진행 보상이 쓰는 중앙선 기준(가장 가까운 중앙선 점의 남은 길이) + 같은 선. 인자: 맵id 출력png"""
import sys
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim import maps, tracks
for _f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(_f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break

mp, out = sys.argv[1], sys.argv[2]
m = maps.load(tracks.resolve(mp))
occ, res = m.occupancy, float(m.resolution)
ox, oy = float(m.origin[0]), float(m.origin[1])
H, W = occ.shape
cl = np.asarray(m.centerline, float)
ii, jj = np.mgrid[0:H, 0:W]
X = ox + (jj + 0.5) * res; Y = oy + (ii + 0.5) * res
free = (~occ) & (np.asarray(m.edt) > 0.16)
lab, _ = ndimage.label(free)
c0 = cl[0]
i0, j0 = int((c0[1] - oy) / res), int((c0[0] - ox) / res)
free &= lab == lab[i0, j0]                                   # the track's own free space
# the finish line: across the track at the centreline's first point, square to the way it runs
t = cl[3] - cl[0]; t /= np.linalg.norm(t); n = np.array([-t[1], t[0]])
A = (X - c0[0]) * t[0] + (Y - c0[1]) * t[1]
band = free & (np.abs(A) < res)
bl, _ = ndimage.label(band, structure=np.ones((3, 3)))
barrier = bl == bl[i0, j0]
src = free & ~barrier & ndimage.binary_dilation(barrier, iterations=2) & (A < 0)
node = free & ~barrier
idx = -np.ones((H, W), int); idx[node] = np.arange(node.sum())
rows, cols, wts = [], [], []
moves = [(0, 1), (1, 0), (1, 1), (1, -1), (1, 2), (2, 1), (2, -1), (1, -2)]
for di, dj in moves:
    a = node[max(0, -di):H - max(0, di), max(0, -dj):W - max(0, dj)]
    b = node[max(0, di):H - max(0, -di) if di < 0 else H, max(0, dj):W if dj >= 0 else W + dj]
    # shift by (di, dj): cell (i, j) -> (i + di, j + dj)
    ia, ja = np.nonzero(node)
    ib, jb = ia + di, ja + dj
    ok = (ib >= 0) & (ib < H) & (jb >= 0) & (jb < W)
    ia, ja, ib, jb = ia[ok], ja[ok], ib[ok], jb[ok]
    ok = node[ib, jb]
    if abs(di) + abs(dj) == 3:                               # a knight's move: both cells it passes must be free
        si, sj = np.sign(di), np.sign(dj)
        ok &= node[np.clip(ia + si, 0, H - 1), np.clip(ja + sj, 0, W - 1)]
        ok &= node[np.clip(ia + (di if abs(di) == 1 else si), 0, H - 1), np.clip(ja + (dj if abs(dj) == 1 else sj), 0, W - 1)]
    elif abs(di) + abs(dj) == 2:
        ok &= node[ia + di, ja] & node[ia, ja + dj]
    ia, ja, ib, jb = ia[ok], ja[ok], ib[ok], jb[ok]
    w = res * np.hypot(di, dj)
    rows += [idx[ia, ja], idx[ib, jb]]; cols += [idx[ib, jb], idx[ia, ja]]; wts += [np.full(ia.size, w)] * 2
G = sparse.csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(node.sum(), node.sum()))
d = dijkstra(G, directed=False, indices=idx[src], min_only=True)
D = np.full((H, W), np.nan); D[node] = d
D[~np.isfinite(D)] = np.nan
# the centreline measure: the remaining length from the nearest centreline point
seg = np.linalg.norm(np.diff(np.vstack([cl, cl[:1]]), axis=0), axis=1)
s = np.concatenate([[0.0], np.cumsum(seg)[:-1]]); L = float(seg.sum())
_, k = cKDTree(cl).query(np.stack([X[node], Y[node]], 1))
C = np.full((H, W), np.nan); C[node] = (L - s[k]) % L
ext = [ox, ox + W * res, oy, oy + H * res]
vmax = float(np.nanmax(D))
fig, axs = plt.subplots(1, 2, figsize=(22, 9))
for ax, F, title in ((axs[0], D, f"결승선까지 남은 최단 거리 (벽만, 차 반폭 0.16 m 빼고) -- 한 바퀴 최단 {vmax:.1f} m"),
                     (axs[1], C, f"지금 진행 보상: 가장 가까운 중앙선 점의 남은 길이 -- 중앙선 한 바퀴 {L:.1f} m")):
    ax.imshow(~occ, cmap="gray", origin="lower", extent=ext, alpha=0.35)
    im = ax.imshow(np.ma.masked_invalid(F), cmap="turbo", origin="lower", extent=ext, vmin=0, vmax=max(vmax, L), alpha=0.85)
    ax.contour(X, Y, np.ma.masked_invalid(F), levels=np.arange(1.0, max(vmax, L), 1.0), colors="k", linewidths=0.6)
    ax.plot([c0[0] - n[0] * 1.2, c0[0] + n[0] * 1.2], [c0[1] - n[1] * 1.2, c0[1] + n[1] * 1.2], "r-", lw=3)
    ax.annotate("", xy=(c0[0] + t[0] * 0.9, c0[1] + t[1] * 0.9), xytext=(c0[0], c0[1]), arrowprops=dict(color="r", width=2))
    ax.set_title(title, fontsize=12); ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, fraction=0.03, label="남은 거리 [m] (검은 선 = 1 m마다 같은 거리)")
# arrows: the way the remaining distance falls fastest (what the progress reward pays for)
gy, gx = np.gradient(np.nan_to_num(D, nan=np.nanmax(D)))
st = 9
sl = (slice(0, H, st), slice(0, W, st))
mk = node[sl] & np.isfinite(D[sl])
u, v = -gx[sl], -gy[sl]
nn = np.hypot(u, v) + 1e-9
axs[0].quiver(X[sl][mk], Y[sl][mk], (u / nn)[mk], (v / nn)[mk], color="w", scale=45, width=0.0025)
cl_pts = cl[::4]
axs[1].plot(cl[:, 0], cl[:, 1], "w-", lw=1.0)
plt.tight_layout(); plt.savefig(out, dpi=70)
print(out, f"| 최단 한 바퀴 {vmax:.1f} m, 중앙선 한 바퀴 {L:.1f} m")
