"""(c2*): (c2) with the values the race bags cannot identify set to physically plausible ones instead of whatever the
optimiser drifted to -- one peak friction for both axles (mu_f_scale 1: no evidence for an asymmetry), mu at the
lower bound of its profile (the smallest grip consistent with the data under the C 1.3 shape) -- and the identified
values refitted. Usage: fit_sim3.py <mu> [comp]"""
import json, pickle, sys, time
import numpy as np, torch
from common import WORK, SPLIT
from f1sim.bag_profile import models as M, windows as WN, fit as F

torch.set_num_threads(1)
mu = float(sys.argv[1]); fscale = float(sys.argv[2]); car = sys.argv[3] if len(sys.argv) > 3 else "comp"
D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
W = WN.concat([D[k]["W"] for k in SPLIT[car]["train"] if k in D])
idx = np.arange(0, len(W["r"]), 2)
c2 = json.load(open(f"{WORK}/fit_sim2_{car}.json"))["over"]
F.BOUNDS.update({"steer_gain": (0.3, 2.5), "steer_bias": (-0.12, 0.12), "servo_tau": (0.005, 0.25),
                 "B_f": (0.2, 40.0), "B_r": (0.2, 40.0)})


def simmodel(W_, p, idx=None, steer_key="steer", hor=100):
    return M.sim_rollout(W_, over=dict(p), idx=idx, steer_key=steer_key, hor=hor)


t0 = time.time()
p = dict(c2, mu=mu, mu_ref=mu, mu_f_scale=fscale)
# keep the slopes mu * scale * B * C of (c2) as the start
p["B_f"] = c2["B_f"] * c2["mu"] * c2["mu_f_scale"] / (mu * fscale)
p["B_r"] = c2["B_r"] * c2["mu"] / mu
lat = ["steer_gain", "steer_bias", "servo_tau", "B_f", "B_r"]
print("start", round(F.objective(simmodel, W, p, idx=idx), 5), flush=True)
p, J, r = F.fit(simmodel, W, p, lat, idx=idx, max_nfev=30)
print("fit", {k: round(p[k], 4) for k in lat}, "J", round(J, 5), "nfev", r.nfev, f"{time.time() - t0:.0f}s", flush=True)
json.dump(dict(over=p, J=J, mu=mu, mu_f_scale=fscale), open(f"{WORK}/fit_sim3_{car}.json", "w"), indent=1)
print("done", flush=True)
