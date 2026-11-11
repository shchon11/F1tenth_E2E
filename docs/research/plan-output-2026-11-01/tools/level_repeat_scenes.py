"""같은 경주를 두 번 낸 기록(level_repeat.py와 같은 입력)에서, 두 번 다 세게 닿은 경주의 장면: A에서 처음 세게 닿은 자리 둘레 지도,
닿기 3 s 전부터 1 s 뒤까지 A(파랑)·B(주황)의 주행 궤적, 세게 닿은 자리(x), 닿는 순간의 우리 차·상대 차 외곽, 상자.
칸 제목: 경주 번호, 트랙, 종류, 두 번의 세게 닿은 횟수. 인자: 기록 A npz, 기록 B npz, 그림 png [종류마다 장면 수=3]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt, C_BOX, C_CAR, trk_name

A, B = Rec(sys.argv[1]), Rec(sys.argv[2]); out = sys.argv[3]; n_each = int(sys.argv[4]) if len(sys.argv) > 4 else 3
NAMES = ("벽", "상자", "상대 차")
picks = []
for i in (1, 2, 0):
    ha, hb = A.d[:, :, 3 + i], B.d[:, :, 3 + i]
    both = np.nonzero((ha.sum(0) > 0) & (hb.sum(0) > 0))[0]
    both = both[np.argsort(-(ha[:, both].sum(0) + hb[:, both].sum(0)), kind="stable")]
    rng = np.random.default_rng(0)
    sel = list(both[:1]) + list(rng.permutation(both[1:])[:n_each - 1])          # the worst one + random ones
    picks += [(int(b), i) for b in sel]
n = len(picks); cols = 3; rows_ = (n + cols - 1) // cols
fig, axs = plt.subplots(rows_, cols, figsize=(6.2 * cols, 6.0 * rows_))
axs = np.atleast_2d(axs)
k3, k1 = int(round(3.0 / A.dt)), int(round(1.0 / A.dt))
for ax, (b, i) in zip(axs.ravel(), picks):
    tid = int(A.tid[0, b]) if A.tid.ndim == 2 else int(A.tid[b])
    A.draw_map(ax, tid)
    tA = int(np.nonzero(A.d[:, b, 3 + i] > 0)[0][0])
    c = A.st[tA + 1, b, :2].astype(float)
    for R, col, lab in ((A, "#1f77b4", "첫 번째"), (B, "#ff7f0e", "두 번째")):
        hits = np.nonzero(R.d[:, b, 3 + i] > 0)[0]
        near = [t for t in hits if np.linalg.norm(R.st[t + 1, b, :2] - c) < 4.0]
        t0 = near[0] if near else (hits[0] if len(hits) else tA)
        seg = R.st[max(t0 - k3, 0): min(t0 + k1, R.T) + 1, b, :2].astype(float)
        brk = np.linalg.norm(np.diff(seg, axis=0), axis=1) > 1.0                 # a restart inside the window: do not join
        seg = seg.copy(); seg[1:][brk] = np.nan
        ax.plot(seg[:, 0], seg[:, 1], color=col, lw=1.8, label=f"{lab}: 세게 {int(R.d[:, b, 3 + i].sum())}번", zorder=4)
        for t in hits:
            p = R.st[t + 1, b, :3].astype(float)
            ax.plot(p[0], p[1], "x", color=col, ms=11, mew=2.5, zorder=6)
        p = R.st[t0 + 1, b, :3].astype(float); q = R.body(*p)
        ax.plot(*np.vstack([q, q[:1]]).T, color=col, lw=1.2, zorder=5)
        if R.on is not None and R.on[min(t0, R.T - 1), b]:
            o = R.op[t0 + 1, b, :3].astype(float); q = R.body(*o)
            ax.plot(*np.vstack([q, q[:1]]).T, color=C_CAR, lw=1.2, ls="-" if R is A else "--", zorder=5)
            so = R.op[max(t0 - k3, 0): t0 + 2, b, :2].astype(float)
            ax.plot(so[:, 0], so[:, 1], color=C_CAR, lw=0.8, ls="-" if R is A else "--", zorder=3)
        if R is A:
            for poly in R.boxes(min(t0, R.T - 1), b):
                ax.fill(poly[:, 0], poly[:, 1], color=C_BOX, zorder=2)
    ax.set_xlim(c[0] - 4.0, c[0] + 4.0); ax.set_ylim(c[1] - 4.0, c[1] + 4.0)
    ax.set_title(f"경주 {b} · {trk_name(tid)} · {NAMES[i]}에 세게 닿음", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
for ax in axs.ravel()[n:]:
    ax.axis("off")
fig.suptitle(f"{A.name}: 같은 경주를 탐색 잡음만 바꿔 두 번 냈을 때 두 번 다 세게 닿은 경주 (x = 세게 닿은 자리, 갈색 = 상자, 빨강 = 상대 차)", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=70); print("saved", out, picks)
