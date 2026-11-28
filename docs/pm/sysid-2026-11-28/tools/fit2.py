"""Task 1 step 2: tyre-shape variants of (b) and the fixed (d) on the TRAIN split. Usage: fit2.py comp|pre"""
import json, pickle, sys, time
import numpy as np
from common import WORK, SPLIT
from f1sim.bag_profile import models as M, windows as WN, fit as F

car = sys.argv[1] if len(sys.argv) > 1 else "comp"
D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
W = WN.concat([D[k]["W"] for k in SPLIT[car]["train"] if k in D])
R = json.load(open(f"{WORK}/fit1_{car}.json"))
spd = R["speed"]["p"]
b0 = R["st"]["p"]
t0 = time.time()
out = {}
full = ["k", "b", "tau_d", "tau_s", "sf", "mf", "Cf", "sr", "mr", "Cr"]
for tag, fn, fixed, names in (
        ("st_C13", M.st_rollout, dict(Cf=1.3, Cr=1.3), [n for n in full if n not in ("Cf", "Cr")]),
        ("st_C10", M.st_rollout, dict(Cf=1.0, Cr=1.0), [n for n in full if n not in ("Cf", "Cr")]),
        ("st_lt", M.st_lt_rollout, {}, full)):
    best = None
    for s0 in (dict(b0), dict(b0, sf=8.0, sr=15.0, k=0.8)):
        base = {**M.ST_DEFAULT, **spd, **s0, **fixed}
        p, J, r = F.fit(fn, W, base, names, max_nfev=120)
        print(tag, {k: round(p[k], 4) for k in full}, round(J, 5), "nfev", r.nfev, f"{time.time() - t0:.0f}s", flush=True)
        if best is None or J < best[1]:
            best = (p, J)
    out[tag] = dict(p=best[0], J=best[1])
R.update(out)
json.dump(R, open(f"{WORK}/fit1_{car}.json", "w"), indent=1)
print("done", flush=True)
