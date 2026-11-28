"""Task 1 step 1: fit the speed loop, then (a) lin, (b) st, (d) st+lt on the TRAIN split of one car; save params.
Usage: fit1.py comp|pre"""
import json
import pickle
import sys
import time

import numpy as np

from common import WORK, SPLIT
from f1sim.bag_profile import models as M, windows as WN, fit as F

car = sys.argv[1] if len(sys.argv) > 1 else "comp"
D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
W = WN.concat([D[k]["W"] for k in SPLIT[car]["train"] if k in D])
print(car, "train windows", len(W["r"]), flush=True)
t0 = time.time()
res = {}
# ---- longitudinal (shared by a, b, d) ----
sp, Jv, _ = F.fit(M.speed_rollout, W, dict(M.SPEED_DEFAULT), ["tau_m", "tau_v", "a_a", "a_b", "v_s"], target="v")
res["speed"] = dict(p={k: sp[k] for k in M.SPEED_DEFAULT}, J=Jv)
print("speed", res["speed"], f"{time.time() - t0:.0f}s", flush=True)
spd = res["speed"]["p"]
# ---- (a) ----
pa, Ja, _ = F.fit(M.lin_rollout, W, {**spd, **dict(k=1.15, tau=0.10, b=-0.03, kus=0.02)}, ["k", "tau", "b", "kus"])
res["lin"] = dict(p=pa, J=Ja)
print("lin", {k: round(pa[k], 4) for k in ("k", "tau", "b", "kus")}, Ja, f"{time.time() - t0:.0f}s", flush=True)
# ---- (b), (d): two starts each (soft tyres like the sim's, stiff tyres + low gain) ----
names = ["k", "b", "tau_d", "tau_s", "sf", "mf", "Cf", "sr", "mr", "Cr"]
starts = [dict(k=1.0, b=-0.02, tau_d=0.05, tau_s=0.04, sf=3.5, mf=1.0, Cf=1.3, sr=5.5, mr=1.1, Cr=1.3),
          dict(k=0.7, b=-0.015, tau_d=0.06, tau_s=0.04, sf=8.0, mf=1.2, Cf=1.5, sr=15.0, mr=1.2, Cr=1.5)]
for tag, fn in (("st", M.st_rollout), ("st_lt", M.st_lt_rollout)):
    best = None
    for i, s0 in enumerate(starts):
        p, J, r = F.fit(fn, W, {**M.ST_DEFAULT, **spd, **s0}, names, max_nfev=120)
        print(tag, "start", i, {k: round(p[k], 4) for k in names}, round(J, 5), "nfev", r.nfev, f"{time.time() - t0:.0f}s", flush=True)
        if best is None or J < best[1]:
            best = (p, J)
        res[f"{tag}_start{i}"] = dict(p=p, J=J)
    res[tag] = dict(p=best[0], J=best[1])
json.dump(res, open(f"{WORK}/fit1_{car}.json", "w"), indent=1)
print("done", f"{time.time() - t0:.0f}s", flush=True)
