"""Per-bag identification with the sim's own physics (what a 'bag -> profile' tool would have to do):
for each bag, grid (steer_gain, steer_bias, cmd_delay) on the 1 s yaw-rate error, then a mu profile at the best
point (mu_stiffness_exp 0: mu moves only the peak). Same venue -> the estimates should agree."""
import itertools
import json
import pickle
import time

import numpy as np
import torch

import models as M
from windows import WARM

torch.set_num_threads(3)
Wa = pickle.load(open("windows.pkl", "rb"))
H = 100


def cat(*Ws):
    keys = [k for k in Ws[0] if isinstance(Ws[0][k], np.ndarray)]
    return {k: np.concatenate([W[k] for W in Ws]) for k in keys}


W = cat(Wa["fit"], Wa["val"], Wa["pre"])
venue = {"20260826": "comp", "20260827": "comp", "20260714": "map17x18", "20260725": "map12x16", "20260727": "map16x07"}
bags = sorted(set(W["bag"]))
G = dict(steer_gain=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], steer_bias=[-0.03, -0.02, -0.01, 0.0, 0.01, 0.02],
         cmd_delay=[0.0, 0.03, 0.06, 0.09])
MU = [0.6, 0.75, 0.9, 1.05, 1.2, 1.5]
combos = list(itertools.product(*G.values()))
out = {}
t0 = time.time()
for b in bags:
    idx = np.flatnonzero(W["bag"] == b)
    if len(idx) < 30 or b.startswith("20260725-1522"):
        continue
    idx = idx[:: max(1, len(idx) // 160)]
    n = len(idx)
    rm = W["r"][idx, WARM + 1:WARM + H + 1]
    over = {k: np.repeat(np.array([c[i] for c in combos]), n) for i, k in enumerate(G)}
    Wt = {k: np.tile(v[idx], (len(combos),) + (1,) * (v.ndim - 1)) for k, v in W.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(W["r"])}
    pr = M.sim_rollout(Wt, over=over, hor=H)
    err = ((pr["r"][:, 1:H + 1] - np.tile(rm, (len(combos), 1))) ** 2).mean(1).reshape(len(combos), n).mean(1)
    k = int(np.argmin(err))
    best = dict(zip(G, combos[k]))
    # +3 % band of each parameter (others free): how sharply the bag pins it
    band = {}
    for i, name in enumerate(G):
        ok = [c[i] for c, e in zip(combos, err) if e <= 1.03 * err[k]]
        band[name] = (min(ok), max(ok))
    # mu profile at the best point
    over_mu = {kk: np.full(n * len(MU), v) for kk, v in best.items()}
    over_mu["mu"] = np.repeat(np.array(MU), n)
    Wm = {kk: np.tile(v[idx], (len(MU),) + (1,) * (v.ndim - 1)) for kk, v in W.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(W["r"])}
    prm = M.sim_rollout(Wm, over=over_mu, hor=H)
    e_mu = ((prm["r"][:, 1:H + 1] - np.tile(rm, (len(MU), 1))) ** 2).mean(1).reshape(len(MU), n).mean(1)
    ay_max = float(np.percentile(np.abs(W["ay"][idx, WARM:WARM + H]).max(1), 90))
    out[b] = dict(venue=venue[b[:8]], n=int(n), best=best, rmse=float(np.sqrt(err[k])), band=band,
                  mu_rmse=dict(zip(MU, np.sqrt(e_mu).round(4).tolist())), ay_p90_of_window_max=ay_max)
    print(b, json.dumps(out[b]), f"{time.time() - t0:.0f}s", flush=True)
json.dump(out, open("ident.json", "w"), indent=1)
