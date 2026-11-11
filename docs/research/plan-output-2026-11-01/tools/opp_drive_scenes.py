"""상자가 없을 때 상대 차가 달린 길을 종류별로 그림 (11-12 사용자: "그냥 주행이 된다~ 수준" -> 숫자보다 먼저 장면).
drive_record.py 기록에서 종류마다 상자 없이 가장 오래 달린 차를 속도 배율 구간별로 하나씩(위 줄: 배율 0.95 이상, 아래 줄: 0.4-0.7),
마지막 칸은 같은 기록의 우리 차(혼자, 상자 없음). 길의 색 = 속도(모든 칸 같은 눈금), 검은 x = 벽에 닿은 자리, 빨간 x = 차끼리 닿은 자리,
회색 가는 선 = 같은 경주의 우리 차.
인자: 기록 npz 출력png"""
import sys
import numpy as np
from rec_lib import Rec, plt, trk_name
from matplotlib.collections import LineCollection

rc = Rec(sys.argv[1]); z = rc.z; out = sys.argv[2]
T, B, dt = rc.T, rc.B, rc.dt
on = z["on"][:T]; kind = z["kind"][:T]; scale = z["scale"][:T]; tid = z["tid"][:T]
nobox = ~(z["sid"][:T] >= 0).any(-1)
op, st, ocnt, cnt = z["op"], z["st"], z["ocnt"], z["cnt"]


def longest(m):
    """per car: (length, start) of the longest run of True"""
    best = np.zeros((B, 2), int)
    for b in range(B):
        run = 0
        for t in range(T):
            run = run + 1 if m[t, b] else 0
            if run > best[b, 0]:
                best[b] = (run, t - run + 1)
    return best


def pick(m, want):
    """the car with the longest run under mask m whose scale is nearest `want`"""
    L = longest(m)
    c = np.nonzero(L[:, 0] >= 150)[0]
    if len(c) == 0:
        c = np.argsort(-L[:, 0])[:1]
        if L[c[0], 0] < 50:
            return None
    b = c[np.argmin(np.abs(scale[L[c, 1], c] - want))] if want else c[np.argmax(L[c, 0])]
    return int(b), int(L[b, 1]), int(L[b, 0])


def step_ok(s):
    return np.linalg.norm(s[1:, :2] - s[:-1, :2], axis=1) < 2.0


def panel(ax, s, c, t0, n, title, other=None):
    k = int(tid[t0, b_])
    rc.draw_map(ax, k)
    seg = s[t0:t0 + n + 1].astype(float)
    ok = step_ok(seg)
    if other is not None:
        o = other[t0:t0 + n + 1].astype(float); oo = step_ok(o)
        for i in np.nonzero(oo)[0]:
            ax.plot(o[i:i + 2, 0], o[i:i + 2, 1], color="#999999", lw=0.6, zorder=1)
    pts = np.stack([seg[:-1, :2], seg[1:, :2]], 1)[ok]
    lc = LineCollection(pts, cmap="viridis", norm=plt.Normalize(0, 8), lw=1.8, zorder=2)
    lc.set_array(np.abs(seg[:-1, 3])[ok]); ax.add_collection(lc)
    d = np.clip(np.diff(c[t0:t0 + n, :], axis=0), 0, None) * ok[1:n, None]
    for col, colr in ((0, "black"), (2, "#d62728")):
        h = np.nonzero(d[:, col] > 0)[0] + 1
        ax.plot(seg[h, 0], seg[h, 1], "x", color=colr, ms=7, mew=2, zorder=4)
    v = np.abs(seg[:-1, 3])[ok]
    ax.set_title(f"{title}\n{trk_name(k)}, {n * dt:.0f} s, 평균 {v.mean():.1f} m/s, 최고 {v.max():.1f} m/s", fontsize=9)
    xs, ys = seg[:, 0][np.r_[ok, True]], seg[:, 1][np.r_[ok, True]]
    ax.set_xlim(xs.min() - 1.5, xs.max() + 1.5); ax.set_ylim(ys.min() - 1.5, ys.max() + 1.5)
    return lc


fig, axs = plt.subplots(2, len(rc.kinds[:4]) + 1, figsize=(4.4 * (len(rc.kinds[:4]) + 1), 9.0))
lc = None
for j, kn in enumerate(rc.kinds[:4]):
    for i, (lo, hi, want) in enumerate(((0.95, 9.9, 1.0), (0.0, 0.7, 0.55))):
        ax = axs[i, j]
        p = pick(on & nobox & (kind == j) & (scale >= lo) & (scale < hi), want)
        if p is None:
            ax.axis("off"); continue
        b_, t0, n = p
        lc = panel(ax, op[:, b_], ocnt[:, b_], t0, n, f"상대 {kn}, 속도 배율 {scale[t0, b_]:.2f} (차 {b_})", other=st[:, b_])
for i in range(2):
    p = pick(~on & nobox, None) if i == 0 else pick(~on & nobox & (np.arange(B)[None, :] != p[0]), None)
    b_, t0, n = p
    lc = panel(axs[i, -1], st[:, b_], cnt[:, b_], t0, n, f"우리 차 혼자 ({rc.name.split('/')[0]}, 차 {b_})")
cb = fig.colorbar(lc, ax=axs, fraction=0.012, pad=0.01); cb.set_label("속도 [m/s]")
fig.suptitle("상자 없을 때 달린 길: 상대 차 종류별(위 = 속도 배율 0.95 이상, 아래 = 0.4-0.7)과 우리 차. 검은 x = 벽에 닿음, 빨간 x = 차끼리 닿음, 회색 = 같은 경주의 우리 차", fontsize=11)
fig.savefig(out, dpi=110, bbox_inches="tight")
print("저장", out)
