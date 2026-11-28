"""(c2): the simulator's structure with longitudinal load transfer switched off through h = 0 (contact heights shifted
by the same 0.074 m so the contact roll / pitch moments, which depend on h - contact_z only, are unchanged), wheel model
on as in the simulator. Values refitted on TRAIN, starting from (b') mapped. Usage: fit_sim2.py comp|pre"""
import json, pickle, sys, time
import numpy as np, torch
from common import WORK, SPLIT
from f1sim.bag_profile import models as M, windows as WN, fit as F

torch.set_num_threads(1)
car = sys.argv[1] if len(sys.argv) > 1 else "comp"
D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
W = WN.concat([D[k]["W"] for k in SPLIT[car]["train"] if k in D])
idx = np.arange(0, len(W["r"]), 2)
R = json.load(open(f"{WORK}/fit1_{car}.json"))
b = R["st_C13"]["p"]
F.BOUNDS.update({"steer_gain": (0.3, 2.5), "steer_bias": (-0.12, 0.12), "servo_tau": (0.005, 0.25),
                 "B_f": (0.2, 40.0), "B_r": (0.2, 40.0), "mu": (0.5, 3.0), "mu_f_scale": (0.3, 3.0),
                 "motor_tau": (0.03, 1.0), "motor_delay": (0.0, 0.3), "a_max": (1.0, 15.0), "motor_ramp": (0.0, 20.0)})
H_SHIFT = dict(h=0.0, contact_z_end=0.031 - 0.074, contact_z_side=0.060 - 0.074)


def simmodel(W_, p, idx=None, steer_key="steer", hor=100):
    over = {k: v for k, v in p.items() if not k.startswith("_")}
    over["mu_ref"] = over.get("mu", 1.0489)
    over.update(H_SHIFT)
    return M.sim_rollout(W_, over=over, idx=idx, steer_key=steer_key, hor=hor)


t0 = time.time()
p = dict(steer_gain=b["k"], steer_bias=b["b"], cmd_delay=0.043, servo_tau=b["tau_s"], mu=b["mr"],
         mu_f_scale=b["mf"] / b["mr"], B_f=b["sf"] / (b["mf"] * 1.3), C_f=1.3, B_r=b["sr"] / (b["mr"] * 1.3), C_r=1.3,
         E_f=0.0, E_r=0.0, motor_tau=0.09, motor_delay=0.078, a_max=5.0, a_brake=5.0)
print("start", round(F.objective(simmodel, W, p, idx=idx), 5), flush=True)
lon = ["motor_tau", "motor_delay", "a_max"]           # a_brake stays the measured regen limit (5.0)
p, Jv, _ = F.fit(simmodel, W, p, lon, target="v", idx=idx, max_nfev=25)
print("lon", {k: round(p[k], 4) for k in lon}, "Jv", round(Jv, 5), f"{time.time() - t0:.0f}s", flush=True)
lat = ["steer_gain", "steer_bias", "servo_tau", "B_f", "B_r", "mu", "mu_f_scale"]
best = None
for cd in (0.043, 0.06):
    q, J, r = F.fit(simmodel, W, dict(p, cmd_delay=cd), lat, idx=idx, max_nfev=30)
    print("lat cmd_delay", cd, {k: round(q[k], 4) for k in lat}, "J", round(J, 5), "nfev", r.nfev, f"{time.time() - t0:.0f}s", flush=True)
    if best is None or J < best[1]:
        best = (q, J)
p, J = best
over = dict(p, mu_ref=p["mu"], **H_SHIFT)
json.dump(dict(over=over, J=J, Jv=Jv), open(f"{WORK}/fit_sim2_{car}.json", "w"), indent=1)
print("done", {k: round(v, 4) for k, v in over.items()}, f"{time.time() - t0:.0f}s", flush=True)
