"""Fit the simple baselines (and the sim's steering gain/bias) on the fit split, evaluate everything on all splits."""
import itertools
import json
import pickle
import time

import numpy as np
import torch

import models as M
from windows import WARM

torch.set_num_threads(4)
Wa = pickle.load(open("windows.pkl", "rb"))
Wf = Wa["fit"]
H1 = 100


def mse_v(W, pr):
    return float(np.mean((pr["v"][:, 1:H1 + 1] - W["v"][:, WARM + 1:WARM + H1 + 1]) ** 2))


def mse_r(W, pr):
    return float(np.mean((pr["r"][:, 1:H1 + 1] - W["r"][:, WARM + 1:WARM + H1 + 1]) ** 2))


# ---- K1/K2: speed loop first (coordinate descent), then the lateral grid ----
t0 = time.time()
p = dict(tau=0.07, k=1.0, b=0.0, kus=0.0, tau_v=0.03, tau_m=0.2, a_a=5.0, a_b=5.0, v_s=7.0)
grids_v = dict(tau_m=[0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0], a_a=[2, 3, 4, 5, 6, 8], a_b=[2, 3, 4, 5, 6, 8],
               tau_v=[0.0, 0.03, 0.06, 0.1, 0.15], v_s=[5, 6, 7, 8, 10, 20])
for _ in range(3):
    for name, vals in grids_v.items():
        best = min(vals, key=lambda x: mse_v(Wf, M.kb_rollout(Wf, {**p, name: x})))
        p[name] = best
print("speed fit", {k: p[k] for k in grids_v}, f"{time.time() - t0:.0f}s", flush=True)
K1 = dict(p)
best = (1e9, None)
for k, tau, b in itertools.product(np.arange(0.40, 1.11, 0.05), np.arange(0.0, 0.161, 0.02), np.arange(-0.04, 0.041, 0.01)):
    e = mse_r(Wf, M.kb_rollout(Wf, {**K1, "k": k, "tau": tau, "b": b}))
    if e < best[0]:
        best = (e, (k, tau, b))
K1.update(k=float(best[1][0]), tau=float(best[1][1]), b=float(best[1][2]))
print("K1", K1, best[0], flush=True)
K2 = dict(K1)
best = (1e9, None)
for k, tau, b, kus in itertools.product(np.arange(0.6, 1.31, 0.05), np.arange(0.0, 0.161, 0.02), np.arange(-0.04, 0.041, 0.01),
                                        np.arange(0.0, 0.101, 0.01)):
    e = mse_r(Wf, M.kb_rollout(Wf, {**K2, "k": k, "tau": tau, "b": b, "kus": kus}))
    if e < best[0]:
        best = (e, (k, tau, b, kus))
K2.update(k=float(best[1][0]), tau=float(best[1][1]), b=float(best[1][2]), kus=float(best[1][3]))
print("K2", K2, best[0], f"{time.time() - t0:.0f}s", flush=True)

# ---- S1: the sim with its steering gain and trim fitted (everything else nominal) ----
sub = np.arange(0, len(Wf["r"]), 2)
Wsub = {k: (v[sub] if isinstance(v, np.ndarray) and len(v) == len(Wf["r"]) else v) for k, v in Wf.items()}
res = []
for g, b in itertools.product([0.8, 0.9, 1.0, 1.1, 1.2], [-0.03, -0.02, -0.01, 0.0, 0.01]):
    pr = M.sim_rollout(Wsub, over={"steer_gain": g, "steer_bias": b})
    res.append((mse_r(Wsub, pr), g, b))
    print("  S1 grid", g, b, round(res[-1][0], 4), flush=True)
e, g, b = min(res)
S1 = {"steer_gain": g, "steer_bias": b}
print("S1", S1, e, f"{time.time() - t0:.0f}s", flush=True)
json.dump(dict(K1=K1, K2=K2, S1=S1, S1_grid=res), open("fitted.json", "w"), indent=1)

# ---- evaluate ----
K0 = dict(tau=0.07, k=1.0, b=0.0, kus=0.0, tau_v=0.03, tau_m=0.2, a_a=5.0, a_b=5.0, v_s=7.0)
preds = {}
for split, W in Wa.items():
    preds[split] = {
        "P": M.persistence(W), "K0": M.kb_rollout(W, K0), "K1": M.kb_rollout(W, K1), "K2": M.kb_rollout(W, K2),
        "S0": M.sim_rollout(W), "S1": M.sim_rollout(W, over=S1),
    }
    print("evaluated", split, f"{time.time() - t0:.0f}s", flush=True)
pickle.dump(preds, open("preds.pkl", "wb"))
