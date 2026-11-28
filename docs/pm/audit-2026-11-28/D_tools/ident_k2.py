"""Per-bag fit of the 4-parameter linear bicycle (k, tau, b, kus) and of the speed loop (tau_m, a_a, a_b):
what a simple 'bag -> profile' would return, and how much it moves between bags of one venue."""
import itertools, json, pickle, numpy as np
import models as M
from windows import WARM
Wa = pickle.load(open("windows.pkl","rb")); F = json.load(open("fitted.json"))
def cat(*Ws):
    keys = [k for k in Ws[0] if isinstance(Ws[0][k], np.ndarray)]
    return {k: np.concatenate([W[k] for W in Ws]) for k in keys}
W = cat(Wa["fit"], Wa["val"], Wa["pre"])
out = {}
for b in sorted(set(W["bag"])):
    idx = np.flatnonzero(W["bag"] == b)
    if len(idx) < 30: continue
    Wb = {k: v[idx] for k, v in W.items() if isinstance(v, np.ndarray) and len(v) == len(W["r"])}
    p = dict(F["K2"])
    # speed loop
    gv = dict(tau_m=[0.1,0.15,0.2,0.3,0.5,0.7], a_a=[2,3,4,5,6,8], a_b=[1,2,3,4,5,6,8], tau_v=[0,0.05,0.1,0.15])
    for _ in range(2):
        for name, vals in gv.items():
            p[name] = min(vals, key=lambda x: np.mean((M.kb_rollout(Wb, {**p, name: x})["v"][:,1:101]-Wb["v"][:,WARM+1:WARM+101])**2))
    best = (9, None); errs = {}
    for k, tau, bb, kus in itertools.product(np.arange(0.7,1.51,0.1), np.arange(0.0,0.201,0.04), np.arange(-0.04,0.041,0.01), [0,0.01,0.02,0.03,0.05,0.08]):
        e = np.mean((M.kb_rollout(Wb, {**p,"k":k,"tau":tau,"b":bb,"kus":kus})["r"][:,1:101]-Wb["r"][:,WARM+1:WARM+101])**2)
        errs[(round(k,2),round(tau,2),round(bb,2),round(kus,2))] = e
        if e < best[0]: best = (e, (k,tau,bb,kus))
    band = {}
    for i, nm in enumerate(["k","tau","b","kus"]):
        ok = [key[i] for key, e in errs.items() if e <= 1.05*best[0]]
        band[nm] = (min(ok), max(ok))
    out[b] = dict(n=int(len(idx)), k=round(best[1][0],2), tau=round(best[1][1],2), b=round(best[1][2],2), kus=round(best[1][3],2),
                  rmse=float(np.sqrt(best[0])), band5=band, speed={k: p[k] for k in gv})
    print(b, json.dumps(out[b]), flush=True)
json.dump(out, open("ident_k2.json","w"), indent=1)
