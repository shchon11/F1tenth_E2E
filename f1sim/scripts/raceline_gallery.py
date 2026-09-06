"""Render the teacher racelines of the training + held-out track set -> docs/racelines.png."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from f1sim.learn import common

names = [n for n in common.TRAIN_TRACKS + common.EVAL_TRACKS if not n.endswith("~rev")]
tracks, rls = common.load_tracks(names, racelines=True)
ncol = 5; nrow = int(np.ceil(len(names) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.9 * nrow), dpi=130); axes = axes.ravel()
for ax, n, t, rl in zip(axes, names, tracks, rls):
    H, W = t.occupancy.shape; ext = [t.origin[0], t.origin[0] + W * t.resolution, t.origin[1], t.origin[1] + H * t.resolution]
    img = np.full((H, W, 3), 0.92); img[t.occupancy] = (0.25, 0.25, 0.3)
    if t.duct is not None: img[t.duct] = (0.95, 0.55, 0.15)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    ax.plot(t.centerline[:, 0], t.centerline[:, 1], "c-", lw=0.4, alpha=0.7)
    sc = ax.scatter(rl.xy[:, 0], rl.xy[:, 1], c=rl.v, s=2, cmap="viridis", vmin=0, vmax=10)
    rr, cc = np.nonzero(~t.occupancy); m = 0.5
    ax.set_xlim(t.origin[0] + cc.min() * t.resolution - m, t.origin[0] + cc.max() * t.resolution + m)
    ax.set_ylim(t.origin[1] + rr.min() * t.resolution - m, t.origin[1] + rr.max() * t.resolution + m)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{n}\nmax|k| {np.abs(rl.kappa).max():.2f} 1/m, est. lap {rl.lap_time:.1f} s", fontsize=7)
for ax in axes[len(names):]: ax.axis("off")
fig.colorbar(sc, ax=axes.tolist(), shrink=0.4, label="raceline speed [m/s]")
fig.suptitle("min-curvature racelines (teacher), margin 0.40 m (less on narrow lanes), |kappa| <= ~1.1; cyan = centerline", fontsize=9)
out = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "racelines.png")
fig.savefig(out, bbox_inches="tight"); print("wrote", out)
