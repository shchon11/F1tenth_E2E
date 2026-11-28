"""Simulator before / after on the pre-competition places (held-out bags), and the default-profile grip choice.
The adopted (c2*) competition-car values; for the pre car its own delay and trim refitted in the simulator on the
map12x16 fit bags (cmd_delay on a grid -- the sim rounds it to 1 ms), its regen-limited braking (a_brake 2.0, the
pre car's -5 A setting); grip mu either the competition fit (1.35) or the pre-place estimate (1.0), stiffness slopes
mu*B*C kept."""
import json, pickle
import numpy as np, torch
from common import WORK, TRANSFER
from f1sim.bag_profile import models as M, windows as WN, fit as F
import evalcore as E

torch.set_num_threads(1)
D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
Wof = lambda keys: WN.concat([D[k]["W"] for k in keys if k in D and len(D[k]["W"]["r"])])
c3 = json.load(open(f"{WORK}/fit_sim3_comp.json"))["over"]
F.BOUNDS.update({"steer_bias": (-0.12, 0.12)})


def with_mu(ov, mu):
    o = dict(ov)
    o["B_f"] = ov["B_f"] * ov["mu"] / mu; o["B_r"] = ov["B_r"] * ov["mu"] / mu
    o["mu"] = o["mu_ref"] = mu
    return o


def simmodel(W_, p, idx=None, steer_key="steer", hor=100):
    return M.sim_rollout(W_, over=dict(p), idx=idx, steer_key=steer_key, hor=hor)


Wf = Wof(TRANSFER["comp->map12x16"]["fit"])
res = {}
for mu in (1.35, 1.0):
    base = dict(with_mu(c3, mu), a_brake=2.0)
    best = None
    for cd in (0.07, 0.085, 0.10, 0.115):
        p, J, _ = F.fit(simmodel, Wf, dict(base, cmd_delay=cd), ["steer_bias"], max_nfev=20)
        print("mu", mu, "cmd_delay", cd, "steer_bias", round(p["steer_bias"], 4), "J", round(J, 5), flush=True)
        if best is None or J < best[1]:
            best = (p, J)
    res[mu] = best[0]
json.dump({str(k): v for k, v in res.items()}, open(f"{WORK}/pre_sim_over.json", "w"), indent=1)
spin = "20260727-204907"
for tag, keys in (("map12x16 held-out 015428", ["20260725-015428"]), ("map17x18 held-out 223707", ["20260714-223707"]),
                  ("map16x07 spin bag", [spin])):
    W = Wof(keys)
    preds = {"시뮬 명목": M.sim_rollout(W),
             "(c2*) 대회 차 값 그대로": M.sim_rollout(W, over=c3),
             "(c2*) + 사전 차 지연·트림·제동, μ 1.35": M.sim_rollout(W, over=res[1.35]),
             "(c2*) + 사전 차 지연·트림·제동, μ 1.0": M.sim_rollout(W, over=res[1.0])}
    T = E.table(W, preds)
    print(E.md(T, tag), flush=True)
    json.dump(T, open(f"{WORK}/pre_sim_{keys[0]}.json", "w"), indent=1, ensure_ascii=False)
print("done", flush=True)
