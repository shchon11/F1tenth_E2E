"""(11-27, the user: "뭐가 달라진지도 모르겠네 체감이 하나도 안됨") Every hard box hit of checkpoint A in exam3, next to what checkpoint B
did at the same box of the same session (same seed = same layout and the same opponent-free world): A's last 2 s before the hit
(red, darker = later, star = the hit) and B's 2 s around its closest pass of that spot (blue), the boxes then, the walls.
usage: box_hit_compare.py A.npz B.npz OUT.png [label A] [label B]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Noto Sans CJK JP"

fa, fb, out = sys.argv[1:4]
la = sys.argv[4] if len(sys.argv) > 4 else "A"; lb = sys.argv[5] if len(sys.argv) > 5 else "B"
A, Bz = np.load(fa), np.load(fb)
dt = float(A["dt"]); n2 = int(2.0 / dt)
H, W = int(A["t_H"][0]), int(A["t_W"][0]); e = A["edt"][:H * W].reshape(H, W); res = float(A["t_res"][0]); org = A["t_origin"][0]
ext = [org[0], org[0] + W * res, org[1], org[1] + H * res]
fps = {int(k[3:]): A[k] for k in A.files if k.startswith("fp_")}
cnt = A["cnt"].astype(float); T, Bn = cnt.shape[:2]
d = np.clip(np.diff(np.concatenate([np.zeros((1, Bn, 6)), cnt], 0), axis=0), 0, None)
hits = np.argwhere(d[:, :, 4] > 0)
cb = Bz["cnt"].astype(float); db = np.clip(np.diff(np.concatenate([np.zeros((1, Bn, 6)), cb], 0), axis=0), 0, None)
n = len(hits); cols = 4; rows = max(1, (n + cols - 1) // cols)
fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.9 * rows)); axs = np.atleast_1d(axs).ravel()
clean = 0
for ax, (t, b) in zip(axs, hits):
    hx, hy = A["st"][t, b, :2]
    ax.imshow(e > 0, origin="lower", extent=ext, cmap="Greys_r", vmin=-0.5, vmax=1.5)
    for sl in range(A["sid"].shape[2]):                                   # the boxes at the hit
        s_ = int(A["sid"][t, b, sl])
        if s_ < 0 or s_ not in fps:
            continue
        x, y, yw = A["pose"][t, b, sl]; f = fps[s_]; c, s = np.cos(yw), np.sin(yw)
        P = np.stack([x + f[:, 0] * c - f[:, 1] * s, y + f[:, 0] * s + f[:, 1] * c], 1)
        ax.fill(P[:, 0], P[:, 1], color="#c98a10", alpha=0.9)
    k0 = max(0, t - n2)
    ax.plot(A["st"][k0:t + 1, b, 0], A["st"][k0:t + 1, b, 1], "-", color="#d62728", lw=2)
    ax.plot(hx, hy, "*", color="gold", ms=13, mec="k")
    # B: its passes of the spot in this session; the one nearest in time
    dist = np.hypot(Bz["st"][:, b, 0] - hx, Bz["st"][:, b, 1] - hy)
    near = np.nonzero(dist < 1.0)[0]
    if near.size:
        kb = near[np.argmin(np.abs(near - t))]
        while kb + 1 < len(dist) and dist[kb + 1] < dist[kb]:
            kb += 1
        while kb > 0 and dist[kb - 1] < dist[kb]:
            kb -= 1
        j0, j1 = max(0, kb - n2 // 2 - 5), min(len(dist), kb + n2 // 2)
        ax.plot(Bz["st"][j0:j1, b, 0], Bz["st"][j0:j1, b, 1], "-", color="#1f77b4", lw=2)
        hitb = db[max(0, j0):j1, b, 4].sum() > 0
        clean += not hitb
        tag = f"{lb}: {'박음' if hitb else '안 박음'}"
    else:
        tag = f"{lb}: 그 자리 안 지남"
    ax.set_xlim(hx - 2.5, hx + 2.5); ax.set_ylim(hy - 2.5, hy + 2.5); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"세션 {b} · {t * dt:.0f} s · {la} {A['st'][t, b, 3]:.1f} m/s 박음\n{tag}", fontsize=9)
for ax in axs[n:]:
    ax.axis("off")
fig.suptitle(f"{la}가 상자를 세게 박은 {n}번 (빨강) — 같은 세션 같은 상자에서 {lb} (파랑): {clean}번 안 박음", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=85)
print(out, n, clean)
