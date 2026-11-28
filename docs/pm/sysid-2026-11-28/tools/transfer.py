"""PM 11-28 item 2: moving the adopted structure (b', single track C 1.3) to another place -- do only the place values
need refitting, or do the structure / car values move too? And how much driving does the refit need?
Refit sets (everything not listed stays at the source fit):
  none | trim b | b + delay tau_d | b + tau_d + grip mu | b + tau_d + stiffness sf, sr | b + tau_d + sf, sr + mu
  | all 8 values (reference: what a full refit at the new place buys)
Judged on held-out bags of the new place only."""
import json, pickle, sys, time
import numpy as np
from common import WORK, SPLIT, TRANSFER
from f1sim.bag_profile import models as M, windows as WN, fit as F
import evalcore as E

D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
R = json.load(open(f"{WORK}/fit1_comp.json"))
F.BOUNDS["mu"] = (0.3, 3.0)
ALL = ["k", "b", "tau_d", "tau_s", "sf", "mf", "sr", "mr"]


def st_shared(W_, q, idx=None, steer_key="steer", hor=200):
    q = dict(q)
    if "mu" in q:
        q["mf"] = q["mr"] = q["mu"]
    return M.st_rollout(W_, q, idx=idx, steer_key=steer_key, hor=hor)


SETS = {"없음": [], "트림 b": ["b"], "b + 지연 τd": ["b", "tau_d"], "b + τd + 마찰 μ": ["b", "tau_d", "mu"],
        "b + τd + 강성 sf, sr": ["b", "tau_d", "sf", "sr"], "b + τd + sf, sr + μ": ["b", "tau_d", "sf", "sr", "mu"],
        "b + τd + 이득 k": ["b", "tau_d", "k"], "전부 8개": ALL}
W = lambda keys: WN.concat([D[k]["W"] for k in keys if k in D and len(D[k]["W"]["r"])])
src_fit = {"comp": R["st_C13"]["p"]}
# map12x16 source: all values fitted on the map12x16 fit bags (same rule as comp)
W12 = W(TRANSFER["comp->map12x16"]["fit"])
p12, J12, _ = F.fit(M.st_rollout, W12, dict(R["st_C13"]["p"]), ALL, max_nfev=150)
src_fit["map12x16"] = p12
print("map12x16 full fit", {k: round(p12[k], 4) for k in ALL}, round(J12, 5), flush=True)
out = {"sources": src_fit}
for tname, T in TRANSFER.items():
    Wf, Wt = W(T["fit"]), W(T["test"])
    src = dict(src_fit[T["src"]])
    print(f"== {tname}: fit windows {len(Wf['r'])}, test windows {len(Wt['r'])}", flush=True)
    res = {}
    preds = {}
    for sname, names in SETS.items():
        base = dict(src)
        model = M.st_rollout
        if "mu" in names:
            base["mu"] = max(src["mf"], src["mr"]); model = st_shared
        if names:
            p, J, _ = F.fit(model, Wf, base, names, max_nfev=100)
        else:
            p, J = base, F.objective(model, Wf, base)
        pr = model(Wt, p)
        preds[sname] = pr
        res[sname] = dict(p={k: p.get(k) for k in ALL + ["mu"] if k in p}, J_fit=J,
                          shift={k: p[k] - src[k] for k in ALL if k in p and k in src})
        print(f"  {sname:20s} refit-J {J:.5f}  " + " ".join(f"{k} {p[k]:.3f}" for k in names if k in p), flush=True)
    T_ = E.table(Wt, preds)
    out[tname] = dict(refit=res, table=T_)
    print(E.md(T_, tname), flush=True)
# driving time needed: refit the minimal set on the first N s of the new place's fit bags (comp -> map12x16)
T = TRANSFER["comp->map12x16"]
Wf, Wt = W(T["fit"]), W(T["test"])
order = np.argsort([f"{b}:{t:08.2f}" for b, t in zip(Wf["bag"], Wf["t0"])])
curve = {}
for names_key in ("b + 지연 τd", "b + τd + 강성 sf, sr"):
    names = SETS[names_key]
    rows = []
    for nsec in (5, 10, 20, 40, 60, 1e9):
        nwin = int(min(len(order), nsec / 0.25))
        idx = np.sort(order[:nwin])
        p, J, _ = F.fit(M.st_rollout, Wf, dict(src_fit["comp"]), names, idx=idx, max_nfev=80)
        e = M.errors(Wt, M.st_rollout(Wt, p))
        rows.append(dict(seconds=float(nwin * 0.25), windows=nwin, yaw1=float(np.nanmedian(e[100]["yaw"])),
                         pos2=float(np.nanmedian(e[200]["pos"])), p={k: p[k] for k in names}))
        print("  curve", names_key, rows[-1], flush=True)
    curve[names_key] = rows
out["curve"] = curve
json.dump(out, open(f"{WORK}/transfer.json", "w"), indent=1, ensure_ascii=False)
print("done", flush=True)
