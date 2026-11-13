"""Gradient noise scale from --track-steps records of an --accum K run (11-14): how many decisions a step must average before
its signal is as large as its noise (McCandlish et al. 2018, "An Empirical Model of Large-Batch Training": B_noise = tr(Sigma)/|G|^2,
from the gradient norms at two batch sizes). With --accum the actor is fixed through the first epoch, so the minibatch gradients
and their sum are taken at the same point.

    python3 noise_scale.py RUN_DIR [decisions per minibatch, default 4096] [minibatches, default 16]
"""
import sys, glob, os
import numpy as np

run = sys.argv[1]; Bs = float(sys.argv[2]) if len(sys.argv) > 2 else 4096.0; K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
Bb = Bs * K
for f in sorted(glob.glob(os.path.join(run, "track", "t*.npz")), key=lambda f: int(os.path.basename(f)[1:-4])):
    z = np.load(f); row = []
    for t in ("policy", "vest", "memv", "tail"):
        if f"term_{t}_norm" not in z.files:
            continue
        n = z[f"term_{t}_norm"].astype(np.float64); S = z[f"term_{t}_sum"].astype(np.float64)
        gs2 = np.mean(n ** 2); gb2 = np.sum((S / K) ** 2)
        G2 = (Bb * gb2 - Bs * gs2) / (Bb - Bs); trS = (gs2 - gb2) / (1 / Bs - 1 / Bb)
        row.append(f"{t} {trS / G2 / 1000:.0f}k" if G2 > 0 else f"{t} 신호 0 (잡음뿐)")
    print(f"update {os.path.basename(f)[1:-4]}: " + " · ".join(row))
