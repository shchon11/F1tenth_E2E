"""(11-17) frozen exam 기록 하나에서 상자에 세게 닿은 장면: 닿기 2 s 전부터 내 차 자취(파랑, 0.25 s 점), 닿기 1.0 s 전 plan(검정),
0.5 s 전 plan(분홍), 닿는 순간 plan(주황), 상대 차(빨강), 상자들(갈색, 닿은 상자 진하게). 인자: 기록npz 출력png [장면 수]"""
import sys
import numpy as np
sys.path.insert(0, ".")
from rec_lib import Rec, plt
import matplotlib.font_manager as fm
from matplotlib.patches import Polygon
for _f in ("NanumGothic", "Noto Sans CJK KR"):
    if any(_f == x.name for x in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
R = Rec(sys.argv[1]); z = R.z; N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
st, pl, sid, pose, op, on = R.st.astype(float), z["pl"].astype(float), z["sid"], z["pose"].astype(float), R.op.astype(float), R.on
fp = {int(k[3:]): z[k] for k in z.files if k.startswith("fp_")}
g, ext = R.grid(0)
ev = []
for t, b in zip(*np.nonzero(R.d[..., 4] > 0)):
    live = np.nonzero(sid[t, b] >= 0)[0]
    if len(live):
        dd = np.hypot(pose[t, b, live, 0] - st[t + 1, b, 0], pose[t, b, live, 1] - st[t + 1, b, 1])
        ev.append((t, b, live[dd.argmin()]))
ev = ev[:N]
cols = 4; rows = int(np.ceil(len(ev) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.5 * rows), squeeze=False)
for a in axs.flat: a.axis("off")
n10, n05 = int(round(1.0 / R.dt)), int(round(0.5 / R.dt))
for a, (t, b, j) in zip(axs.flat, ev):
    a.axis("on"); a.imshow((g <= 0).astype(float), origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1.6)
    for k in np.nonzero(sid[t, b] >= 0)[0]:
        x, y, yw = pose[t, b, k]; f = fp.get(int(sid[t, b, k]))
        if f is None: continue
        c, s = np.cos(yw), np.sin(yw)
        P = np.stack([x + f[:, 0] * c - f[:, 1] * s, y + f[:, 0] * s + f[:, 1] * c], 1)
        a.add_patch(Polygon(P, closed=True, color="#8b4513" if k == j else "#d2a679", alpha=0.95 if k == j else 0.6))
    w = slice(max(0, t - 20), t + 2)
    a.plot(st[w, b, 0], st[w, b, 1], color="#1971c2", lw=1.5); a.plot(st[w, b, 0][::2], st[w, b, 1][::2], ".", color="#1971c2", ms=5)
    for tt, col, lab in ((t - n10, "k", "1.0 s 전 plan"), (t - n05, "#e64980", "0.5 s 전"), (t, "#f08c00", "닿을 때")):
        if tt >= 0:
            a.plot(pl[tt, b, :, 0], pl[tt, b, :, 1], "o-", color=col, ms=2.5, lw=1, label=lab)
            a.plot(st[tt, b, 0], st[tt, b, 1], "s", color=col, ms=5)
    if on[t, b]:
        a.plot(op[w, b, 0], op[w, b, 1], color="#e03131", lw=1.2)
    c = st[t, b, :2]; a.set_xlim(c[0] - 3, c[0] + 3); a.set_ylim(c[1] - 3, c[1] + 3); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"세션 {b} t={t * R.dt:.1f}s · {st[max(0, t - n10), b, 3]:.1f} -> {st[t, b, 3]:.1f} m/s" + (" · 상대 차 있음" if on[t, b] else ""), fontsize=9)
axs.flat[0].legend(fontsize=7, loc="lower left")
fig.suptitle(f"{z['name']}: 상자에 세게 닿은 {len(ev)}건 (파랑 = 내 차 2 s 자취, 검정·분홍·주황 = 닿기 1.0 s·0.5 s 전·닿을 때 plan, 진한 갈색 = 닿은 상자)", fontsize=10)
fig.tight_layout(); fig.savefig(sys.argv[2], dpi=85); print(sys.argv[2], len(ev))
