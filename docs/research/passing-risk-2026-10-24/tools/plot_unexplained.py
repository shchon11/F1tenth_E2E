"""Plot real_bag_unexplained.py's DUMP: the map, the car's path and the unexplained scan points (colour = time)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.load(sys.argv[1]); out = sys.argv[2]
occ, res, (ox, oy, _) = z["occ"], float(z["res"]), z["origin"]
H, W = occ.shape
fig, ax = plt.subplots(1, 2, figsize=(14, 9))
for k, a in enumerate(ax):
    a.imshow(np.where(occ, 0.0, 1.0), cmap="gray", origin="lower", extent=[ox, ox + W * res, oy, oy + H * res], vmin=0, vmax=1)
    a.plot(z["traj"][:, 1], z["traj"][:, 2], "-", color="tab:blue", lw=0.4, alpha=0.6)
    p = z["pts"]
    if k == 0:
        sc = a.scatter(p[:, 0], p[:, 1], c=p[:, 2] - p[0, 2], s=0.2, cmap="viridis")
        fig.colorbar(sc, ax=a, label="time [s]")
        a.set_title("unexplained points (> TOL from the map), every 5th scan")
    else:
        h, xe, ye = np.histogram2d(p[:, 0], p[:, 1], bins=[np.arange(ox, ox + W * res, 0.1), np.arange(oy, oy + H * res, 0.1)])
        a.imshow(np.log1p(h.T), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="magma", alpha=0.8)
        a.set_title("density (log)")
    a.set_aspect("equal")
plt.tight_layout(); plt.savefig(out, dpi=110)
print("saved", out)
