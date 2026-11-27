"""(11-27) Pictures of exam4 'closing rear-end' at-fault contacts (rear_end_why.py's set), 2 s before -> the contact: walls
(the record's distance field), our car's trail (blue, darker = later) and the other car's (red), our plan at 1 s before
(green) and at the contact (black), the contact point (star). usage: rear_end_scenes.py EXAM4.npz OUT.png [lap-bin lo hi]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fn, out = sys.argv[1], sys.argv[2]
lo, hi = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.0, 1.0)
z = np.load(fn); dt = float(z["dt"]); f = z["cfault"].astype(float); T, B = f.shape
st, op, pl = z["st"].astype(float), z["op"].astype(float), z["pl"].astype(float); L = float(z["lapL"][0])
H, W = int(z["t_H"][0]), int(z["t_W"][0]); e = z["edt"][:H * W].reshape(H, W); res = float(z["t_res"][0]); org = z["t_origin"][0]
pace = z["c_pace"][:B]
n1, n2 = int(1.0 / dt), int(2.0 / dt)
ev = []
for t, b in np.argwhere(f > 0):
    if t < n2:
        continue
    x, y, yw, v = st[t, b]
    dx = (op[t, b, 0] - x) * np.cos(yw) + (op[t, b, 1] - y) * np.sin(yw)
    sx = (op[t - n2:t, b, 0] - st[t - n2:t, b, 0]) * np.cos(st[t - n2:t, b, 2]) + (op[t - n2:t, b, 1] - st[t - n2:t, b, 1]) * np.sin(st[t - n2:t, b, 2])
    q = np.mod(z["rem"][t, b], L) / L
    if dx > 0.3 and not ((sx > 0) & (sx < 1.5)).all() and lo <= q < hi:
        ev.append((t, b, q))
ev = ev[:12]
n = len(ev); cols = 4; rows = max(1, (n + cols - 1) // cols)
fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axs = np.atleast_1d(axs).ravel()
for ax, (t, b, q) in zip(axs, ev):
    cx, cy = st[t, b, :2]; R = 3.0
    ext = [org[0], org[0] + W * res, org[1], org[1] + H * res]
    ax.imshow(e > 0, origin="lower", extent=ext, cmap="Greys_r", vmin=-0.5, vmax=1.5)
    for k in range(t - n2, t + 1):
        a = 0.25 + 0.75 * (k - t + n2) / n2
        ax.plot(*st[k, b, :2], "o", ms=3, color=(0.1, 0.3, 0.9, a)); ax.plot(*op[k, b, :2], "o", ms=3, color=(0.9, 0.1, 0.1, a))
    ax.plot(pl[t - n1, b, :, 0], pl[t - n1, b, :, 1], "-", color="green", lw=1.5)
    ax.plot(pl[t, b, :, 0], pl[t, b, :, 1], "-", color="black", lw=1.5)
    ax.plot(*st[t, b, :2], "*", ms=12, color="gold", mec="k")
    ax.set_xlim(cx - R, cx + R); ax.set_ylim(cy - R, cy + R); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"s{b} t{t * dt:.1f}s pace {pace[b]:.2f} lap {q:.2f}\nv {st[t - n1, b, 3]:.1f}->{st[t, b, 3]:.1f} opp {op[t - n1, b, 3]:.1f}->{op[t, b, 3]:.1f}", fontsize=8)
for ax in axs[n:]:
    ax.axis("off")
fig.tight_layout(); fig.savefig(out, dpi=90)
print(out, n)
