"""PM 11-28 correction: default = competition-car values + pre-competition PLACE values. From the pre bags, with the
pre car's own car values (delay, trim) refitted, what place values can be pulled out, and do the three pre places
differ? (b') structure; stiffness and gain held at the competition fit (the transfer test found refitting them buys
nothing on held-out bags); grip mu (shared front/rear) profiled per place and pooled."""
import json, pickle
import numpy as np
from common import WORK, PLACE
from f1sim.bag_profile import models as M, windows as WN, fit as F, ident as I

D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
R = json.load(open(f"{WORK}/fit1_comp.json"))
src = dict(R["st_C13"]["p"])
F.BOUNDS["mu"] = (0.3, 3.0)


def st_shared(W_, q, idx=None, steer_key="steer", hor=200):
    q = dict(q); q["mf"] = q["mr"] = q["mu"]
    return M.st_rollout(W_, q, idx=idx, steer_key=steer_key, hor=hor)


Wof = lambda keys: WN.concat([D[k]["W"] for k in keys if k in D and len(D[k]["W"]["r"])])
groups = {"map12x16": PLACE["pre_map12x16"], "map17x18": PLACE["pre_map17x18"], "map16x07": PLACE["pre_map16x07"],
          "pre pooled": PLACE["pre_map12x16"] + PLACE["pre_map17x18"] + PLACE["pre_map16x07"],
          "comp (reference)": PLACE["comp_map08x23"][::2]}
grid = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.6, 2.0, 3.0]
out = {}
for g, keys in groups.items():
    W = Wof(keys)
    ay1 = np.abs(W["ay"][:, 50:151]).max(1)
    base = dict(src, mu=1.2)
    p, J, _ = F.fit(st_shared, W, base, ["b", "tau_d", "mu"], max_nfev=80)
    r = I.profile_verdict(st_shared, W, p, ["b", "tau_d", "mu"], "mu", grid, max_nfev=40)
    # stiffness refit as well: does the place move them? (interval by profile)
    p2, J2, _ = F.fit(st_shared, W, dict(p), ["b", "tau_d", "sf", "sr"], max_nfev=80)
    rs = I.profile_verdict(st_shared, W, p2, ["b", "tau_d", "sf", "sr"], "sr", [3, 4, 5, 6.14, 7, 8, 10, 15, 30], max_nfev=40)
    rf = I.profile_verdict(st_shared, W, p2, ["b", "tau_d", "sf", "sr"], "sf", [1.5, 2.0, 2.5, 2.97, 3.5, 4, 5, 7], max_nfev=40)
    out[g] = dict(n=int(len(W["r"])), ay_ge7=int((ay1 >= 7).sum()), b=p["b"], tau_d=p["tau_d"], mu_best=p["mu"],
                  mu_verdict=r["verdict"], mu_interval=r["interval"], sf=p2["sf"], sf_interval=rf["interval"], sf_verdict=rf["verdict"],
                  sr=p2["sr"], sr_interval=rs["interval"], sr_verdict=rs["verdict"],
                  mu_points=[(q["x"], q["dJ"], q["se"]) for q in r["points"]])
    print(f"{g:16s} n {len(W['r']):4d} (ay>=7: {(ay1 >= 7).sum():3d}) b {p['b']:+.4f} tau_d {p['tau_d']:.3f} | mu {p['mu']:.2f} "
          f"{r['verdict']} {r['interval']} | sf {p2['sf']:.2f} {rf['verdict']} {rf['interval']} | sr {p2['sr']:.2f} {rs['verdict']} {rs['interval']}", flush=True)
json.dump(out, open(f"{WORK}/pre_place.json", "w"), indent=1)
print("done", flush=True)
