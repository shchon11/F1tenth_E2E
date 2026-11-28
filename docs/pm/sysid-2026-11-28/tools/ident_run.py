"""Task 1.4: which value is identified from race bags? Profile likelihood (block-paired test) for every free value of
the selected structure (b' : C 1.3), plus a shared-mu variant and the sensitivity to the values held fixed (Iz/m,
lf/lr). Usage: ident_run.py comp|pre"""
import json, pickle, sys, time
import numpy as np
from common import WORK, SPLIT
from f1sim.bag_profile import models as M, windows as WN, fit as F, ident as I

car = sys.argv[1] if len(sys.argv) > 1 else "comp"
D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
W = WN.concat([D[k]["W"] for k in SPLIT[car]["train"] if k in D])
R = json.load(open(f"{WORK}/fit1_{car}.json"))
p = dict(R["st_C13"]["p"])
names = ["k", "b", "tau_d", "tau_s", "sf", "mf", "sr", "mr"]
G = dict(k=[0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8], b=[-0.05, -0.04, -0.03, -0.025, -0.02, -0.015, -0.01, 0.0],
         tau_d=[0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12], tau_s=[0.005, 0.015, 0.025, 0.035, 0.05, 0.07, 0.1, 0.14],
         sf=[1.5, 2.0, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0, 7.0], sr=[3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0, 15.0, 30.0],
         mf=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.2, 3.0], mr=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.2, 3.0])
out = {"base": p, "names": names}
t0 = time.time()
for which, grid in G.items():
    r = I.profile_verdict(M.st_rollout, W, p, names, which, grid, max_nfev=60)
    out[which] = r
    print(which, r["verdict"], r["interval"], "best", round(r["best"], 4),
          [(q["x"], round(q["dJ"] * 1e4, 2), round(q["se"] * 1e4, 2)) for q in r["points"]], f"{time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(f"{WORK}/ident_{car}.json", "w"), indent=1)
# shared friction: mf = mr = mu
def st_shared(W_, q, idx=None, steer_key="steer", hor=200):
    q = dict(q); q["mf"] = q["mr"] = q["mu"]
    return M.st_rollout(W_, q, idx=idx, steer_key=steer_key, hor=hor)
F.BOUNDS["mu"] = (0.3, 3.0)
names_s = ["k", "b", "tau_d", "tau_s", "sf", "sr", "mu"]
ps, Js, _ = F.fit(st_shared, W, dict(p, mu=1.3), names_s, max_nfev=60)
print("shared mu fit", {k: round(ps[k], 4) for k in names_s}, Js, flush=True)
r = I.profile_verdict(st_shared, W, ps, names_s, "mu", [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.7, 2.2, 3.0], max_nfev=60)
out["shared_mu"] = dict(fit=ps, J=Js, profile=r)
print("shared mu", r["verdict"], r["interval"], [(q["x"], round(q["dJ"] * 1e4, 2), round(q["se"] * 1e4, 2)) for q in r["points"]], flush=True)
# sensitivity to fixed geometry
sens = {}
for tag, ov in (("iz_m x0.6", dict(iz_m=M.IZ_M * 0.6)), ("iz_m x1.5", dict(iz_m=M.IZ_M * 1.5)),
                ("lr -0.02", dict(lr=M.LR - 0.02, lf=M.LF + 0.02)), ("lr +0.02", dict(lr=M.LR + 0.02, lf=M.LF - 0.02))):
    q, J, _ = F.fit(M.st_rollout, W, {**p, **ov}, names, max_nfev=40)
    sens[tag] = dict(J=J, dJ_rel=J / R["st_C13"]["J"] - 1, p={k: q[k] for k in names})
    print("sens", tag, round(J / R["st_C13"]["J"] - 1, 4), {k: round(q[k], 3) for k in names}, flush=True)
out["sensitivity"] = sens
json.dump(out, open(f"{WORK}/ident_{car}.json", "w"), indent=1)
print("done", f"{time.time() - t0:.0f}s", flush=True)
