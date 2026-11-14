"""한 자리에 몰린 벽 세게 닿음의 장면 (11-15 05:15: plan26r의 늘어난 벽 충돌이 오른쪽 아래 바깥 벽 한 곳에 몰림).
닿은 자리를 0.6 m 격자로 세어 가장 많은 칸 둘레 1.2 m 안의 충돌을 고름. 장면마다: 닿기 2 s 전부터 차 자취(점), 1.0·0.5 s 전과 닿는 순간 차체,
그 세션의 상자(갈색), 닿기 1.0 s 전·0.5 s 전 plan(검정·분홍), 같은 자리를 안 닿고 지난 다른 바퀴 자취(회색). 제목: 세션 경우, 속도, 근처 상자 수.
인자: 기록npz 출력png [장면 수 8]"""
import sys
import numpy as np
from rec_lib import Rec, plt

R = Rec(sys.argv[1]); out = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 8
st = R.st.astype(float); dt = R.dt
t_, b_ = np.nonzero(R.d[:, :, 3] > 0)
xy = st[t_, b_, :2]
cell = np.floor(xy / 0.6).astype(int)
keys, cnt = np.unique(cell, axis=0, return_counts=True)
c0 = (keys[cnt.argmax()] + 0.5) * 0.6
sel = np.nonzero(np.linalg.norm(xy - c0, axis=1) < 1.2)[0]
print(f"벽 세게 {len(t_)}번 중 가장 많은 자리 ({c0[0]:.1f}, {c0[1]:.1f}) 둘레 1.2 m: {len(sel)}번")
sel = sel[:N]
fig, axs = plt.subplots(2, (len(sel) + 1) // 2, figsize=(4.2 * ((len(sel) + 1) // 2), 8.4), squeeze=False)
pl = R.z["pl"] if "pl" in R.z.files else None
on = np.asarray(R.z["on"]).astype(bool)
for ax, i in zip(axs.ravel(), sel):
    t, b = int(t_[i]), int(b_[i]); k = int(R.tid[t, b])
    R.draw_map(ax, k)
    a = max(0, t - 20)
    # other passes of this session near the spot that did not hit
    near = np.nonzero(np.linalg.norm(st[:, b, :2] - c0, axis=1) < 1.0)[0]
    ax.plot(st[near, b, 0], st[near, b, 1], ".", color="0.7", ms=2)
    ax.scatter(st[a:t + 1, b, 0], st[a:t + 1, b, 1], c=np.arange(t + 1 - a), cmap="Blues", s=10)
    for dd, colr in ((10, "0.3"), (5, "m"), (0, "r")):
        s = max(0, t - dd); P = R.body(st[s, b, 0], st[s, b, 1], st[s, b, 2])
        ax.fill(P[:, 0], P[:, 1], color=colr, alpha=0.35)
    nb = 0
    for P in R.boxes(t, b):
        ax.fill(P[:, 0], P[:, 1], color="#a0522d")
        nb += int(np.linalg.norm(P.mean(0) - st[t, b, :2]) < 4.0)
    if pl is not None:
        for dd, colr in ((10, "k"), (5, "m")):
            s = max(0, t - dd); p = pl[s, b].astype(float).reshape(-1, 2) if pl[s, b].size % 2 == 0 else None
            if p is not None:
                ax.plot(p[:, 0], p[:, 1], "-", color=colr, lw=1)
    ax.set_xlim(c0[0] - 3.5, c0[0] + 3.5); ax.set_ylim(c0[1] - 3.5, c0[1] + 3.5); ax.set_aspect("equal")
    case = ("상자" if (R.sid[t, b] >= 0).any() else "빈") + (" · 상대 차" if on[t, b] else "")
    ax.set_title(f"세션 {b} {t * dt:.1f} s · {case}\n닿을 때 {abs(st[t, b, 3]):.1f} m/s, 1 s 전 {abs(st[max(0, t - 10), b, 3]):.1f} m/s · 4 m 안 상자 {nb}개", fontsize=8)
fig.suptitle(f"{R.name}: 벽 세게 닿음이 몰린 자리 ({len(sel)}개)", fontsize=10)
fig.savefig(out, dpi=95, bbox_inches="tight"); print("그림", out)
