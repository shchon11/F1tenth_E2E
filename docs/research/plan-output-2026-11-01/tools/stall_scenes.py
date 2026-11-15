"""(11-16) 선 구간 장면: 서기 1.5 s 전부터 차 자취, 서기 시작할 때 차체·plan(10점)·상자, 6 s 뒤 plan. 인자: 기록npz 출력png 세션,시작초 ..."""
import sys
import numpy as np
from rec_lib import Rec, plt

R = Rec(sys.argv[1]); out = sys.argv[2]; cases = [tuple(float(x) for x in a.split(",")) for a in sys.argv[3:]]
st = R.st.astype(float); pl = R.z["pl"].astype(float); dt = R.dt
fig, axs = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 5.2))
axs = np.atleast_1d(axs)
for ax, (b, t0s) in zip(axs, cases):
    b = int(b); t0 = int(round(t0s / dt)); k = int(R.tid[t0, b])
    g, ext = R.grid(k)
    ax.imshow((g <= 0).astype(float), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
    c = st[t0, b, :2]
    tr = st[max(0, t0 - 15):t0 + 1, b, :2]; ax.plot(tr[:, 0], tr[:, 1], ".", color="#1c7ed6", ms=3)
    for p in R.boxes(t0, b):
        ax.fill(p[:, 0], p[:, 1], color="#a0522d")
    body = R.body(*st[t0, b, :3]); ax.fill(body[:, 0], body[:, 1], color="#1c7ed6", alpha=0.6)
    ax.plot(np.r_[c[0], pl[t0, b, :, 0]], np.r_[c[1], pl[t0, b, :, 1]], "-", color="k", lw=1.5, label="서기 시작 plan")
    t6 = min(t0 + 60, R.T - 1)
    ax.plot(np.r_[st[t6, b, 0], pl[t6, b, :, 0]], np.r_[st[t6, b, 1], pl[t6, b, :, 1]], "-", color="#e8590c", lw=1.5, label="6 s 뒤 plan")
    ax.set_xlim(c[0] - 2.5, c[0] + 2.5); ax.set_ylim(c[1] - 2.5, c[1] + 2.5); ax.set_aspect("equal")
    ax.set_title(f"세션 {b} · {t0s:.1f} s에 섬 · 속도 {st[t0, b, 3]:.2f} m/s", fontsize=10); ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(out, dpi=100); print("그림", out)
