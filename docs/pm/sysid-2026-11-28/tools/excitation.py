"""PM 11-28 item 3: what an ONLINE estimator could pin from driving signals alone, and how fast.
(1) grip: the shared-mu profile on windows split by how hard the car cornered -- mu shows up only near the limit;
(2) trim / delay / yaw gain: per-chunk estimates from N s of driving (spread across chunks = how fast they settle)."""
import json, pickle
import numpy as np
from common import WORK, SPLIT, PLACE
from f1sim.bag_profile import models as M, windows as WN, fit as F, ident as I, measure as MS

D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
R = json.load(open(f"{WORK}/fit1_comp.json"))
p = dict(R["st_C13"]["p"])
F.BOUNDS["mu"] = (0.3, 3.0)
out = {}


def st_shared(W_, q, idx=None, steer_key="steer", hor=200):
    q = dict(q); q["mf"] = q["mr"] = q["mu"]
    return M.st_rollout(W_, q, idx=idx, steer_key=steer_key, hor=hor)


W = WN.concat([D[k]["W"] for k in SPLIT["comp"]["train"]])
ay1 = np.abs(W["ay"][:, 50:151]).max(1)
names = ["k", "b", "tau_d", "tau_s", "sf", "sr", "mu"]
grid = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2, 3.0]
for tag, m in (("ay<6", ay1 < 6), ("6<=ay<8", (ay1 >= 6) & (ay1 < 8)), ("ay>=8", ay1 >= 8)):
    Ws = WN.subset(W, np.flatnonzero(m))
    ps, Js, _ = F.fit(st_shared, Ws, dict(p, mu=1.4), names, max_nfev=60)
    r = I.profile_verdict(st_shared, Ws, ps, names, "mu", grid, max_nfev=30)
    out[f"mu_{tag}"] = dict(n=int(m.sum()), best=ps["mu"], verdict=r["verdict"], interval=r["interval"],
                            points=[(q["x"], q["dJ"], q["se"]) for q in r["points"]])
    print(tag, int(m.sum()), r["verdict"], r["interval"], flush=True)
# per-chunk estimates
grids = {k: np.load(f"{WORK}/grid_{k}.npz") for k in PLACE["comp_map08x23"] + PLACE["pre_map12x16"]}
chunks = {}
for L in (10, 20, 40):
    rows = []
    for key, Z in grids.items():
        S = {k: Z[k] for k in Z.files}
        t = S["t"]
        mv = S["v_wheel"] > 1.0
        if mv.sum() < L * 100:
            continue
        # chunks of L seconds of MOVING time
        idx = np.flatnonzero(mv)
        for c0 in range(0, len(idx) - L * 100, L * 100):
            sl = slice(idx[c0], idx[c0 + L * 100 - 1] + 1)
            Sc = {k: (v[sl] if isinstance(v, np.ndarray) and v.shape[:1] == t.shape else v) for k, v in S.items()}
            tr = MS.trim(Sc, 1.1); dl = MS.delay_xcorr(Sc)
            rows.append(dict(bag=key, place="comp" if key.startswith("202608") else "map12x16",
                             trim=tr.get("value"), delay=dl.get("value")))
    chunks[L] = rows
    for place in ("comp", "map12x16"):
        rr = [r for r in rows if r["place"] == place]
        tv = np.array([r["trim"] for r in rr], float); dv = np.array([r["delay"] for r in rr], float)
        print(f"chunk {L:3d} s {place:9s} n {len(rr):3d} trim median {np.nanmedian(tv):+.4f} sd {np.nanstd(tv):.4f} "
              f"(ok {np.isfinite(tv).mean():.2f}) | delay median {np.nanmedian(dv):.3f} sd {np.nanstd(dv):.3f}", flush=True)
out["chunks"] = {str(k): v for k, v in chunks.items()}
json.dump(out, open(f"{WORK}/excitation.json", "w"), indent=1, default=float)
print("done", flush=True)
