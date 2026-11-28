"""(a) refit with the delay on a grid (whole samples, as the audit did) -- replaces the finite-difference fit, which
stalled in an interpolation dip next to its start value. Also checks the shared speed model's delay on a grid."""
import json, pickle
import numpy as np
from common import WORK, SPLIT
from f1sim.bag_profile import models as M, windows as WN, fit as F

D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
W = WN.concat([D[k]["W"] for k in SPLIT["comp"]["train"]])
R = json.load(open(f"{WORK}/fit1_comp.json"))
sp = R["speed"]["p"]
ps, Jv, curve_v = F.fit_grid(M.speed_rollout, W, dict(sp), ["tau_m", "a_a", "a_b"], "tau_v", np.arange(0.0, 0.205, 0.02), target="v")
print("speed grid", {k: round(ps[k], 3) for k in ("tau_m", "tau_v", "a_a", "a_b")}, round(Jv, 5), "(old, tau_v 0.10:", round(R["speed"]["J"], 5), ")")
print("  curve", [(g, round(j, 5)) for g, j in curve_v])
pa, Ja, curve = F.fit_grid(M.lin_rollout, W, {**sp, "k": 1.05, "b": -0.02, "kus": 0.015}, ["k", "b", "kus"], "tau", np.arange(0.0, 0.205, 0.01))
print("lin grid", {k: round(pa[k], 4) for k in ("k", "tau", "b", "kus")}, round(Ja, 5))
print("  curve", [(round(g, 2), round(j, 5)) for g, j in curve])
R["lin_fd_stalled"] = R["lin"]
R["lin"] = dict(p=pa, J=Ja, curve=curve)
R["speed_grid_check"] = dict(p=ps, J=Jv, curve=curve_v)
json.dump(R, open(f"{WORK}/fit1_comp.json", "w"), indent=1)
