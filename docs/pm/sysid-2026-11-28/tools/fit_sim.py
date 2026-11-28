"""(c): the simulator's own structure (f1sim.dynamics + actuators, wheel model on), values refitted on TRAIN.
Longitudinal first (speed error), then lateral (yaw-rate error), cmd_delay on a grid (the sim rounds it to 1 ms,
so it has no finite-difference gradient). Starts from (b)'s values mapped onto the sim's parameters.
Usage: fit_sim.py comp|pre"""
import json, pickle, sys, time
import numpy as np, torch
from common import WORK, SPLIT
from f1sim.bag_profile import models as M, windows as WN, fit as F

torch.set_num_threads(2)
car = sys.argv[1] if len(sys.argv) > 1 else "comp"
D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
W = WN.concat([D[k]["W"] for k in SPLIT[car]["train"] if k in D])
idx = np.arange(0, len(W["r"]), 2 if car == "comp" else 1)
R = json.load(open(f"{WORK}/fit1_{car}.json"))
b = R["st"]["p"]
F.BOUNDS.update({"steer_gain": (0.3, 2.5), "steer_bias": (-0.12, 0.12), "servo_tau": (0.005, 0.25),
                 "B_f": (0.2, 40.0), "C_f": (1.0, 2.2), "B_r": (0.2, 40.0), "C_r": (1.0, 2.2),
                 "mu": (0.5, 3.0), "mu_f_scale": (0.3, 3.0), "motor_tau": (0.03, 1.0), "motor_delay": (0.0, 0.3),
                 "a_max": (1.0, 15.0), "a_brake": (1.0, 15.0), "c_roll": (0.0, 1.0)})


def simmodel(W_, p, idx=None, steer_key="steer", hor=100):
    over = {k: v for k, v in p.items() if not k.startswith("_")}
    over["mu_ref"] = over.get("mu", 1.0489)            # B is the raw stiffness factor (lateral_B neutral)
    return M.sim_rollout(W_, over=over, idx=idx, steer_key=steer_key, hor=hor)


t0 = time.time()
# start: (b) mapped
p = dict(steer_gain=b["k"], steer_bias=b["b"], cmd_delay=round(b["tau_d"], 3), servo_tau=b["tau_s"],
         mu=b["mr"], mu_f_scale=b["mf"] / b["mr"], B_f=b["sf"] / (b["mf"] * b["Cf"]), C_f=b["Cf"],
         B_r=b["sr"] / (b["mr"] * b["Cr"]), C_r=b["Cr"], E_f=0.0, E_r=0.0,
         motor_tau=0.2, motor_delay=0.07, a_max=5.0, a_brake=5.0, c_roll=0.1)
J0 = F.objective(simmodel, W, p, idx=idx)
print("start (b mapped)", {k: round(v, 4) for k, v in p.items()}, "J", round(J0, 5), flush=True)
# longitudinal
lon = ["motor_tau", "motor_delay", "a_max", "a_brake"]
p, Jv, r = F.fit(simmodel, W, p, lon, target="v", idx=idx, max_nfev=30)
print("lon", {k: round(p[k], 4) for k in lon}, "Jv", round(Jv, 5), f"{time.time() - t0:.0f}s", flush=True)
lat = ["steer_gain", "steer_bias", "servo_tau", "B_f", "C_f", "B_r", "C_r", "mu", "mu_f_scale"]
best = None
for cd in (p["cmd_delay"], 0.035, 0.06):
    q = dict(p, cmd_delay=float(cd))
    q, J, r = F.fit(simmodel, W, q, lat, idx=idx, max_nfev=40)
    print("lat cmd_delay", cd, {k: round(q[k], 4) for k in lat}, "J", round(J, 5), "nfev", r.nfev, f"{time.time() - t0:.0f}s", flush=True)
    if best is None or J < best[1]:
        best = (q, J)
p, J = best
# longitudinal once more at the final lateral values
p, Jv, r = F.fit(simmodel, W, p, lon, target="v", idx=idx, max_nfev=20)
over = dict(p, mu_ref=p["mu"])
json.dump(dict(over=over, J=J, Jv=Jv, J_start=J0), open(f"{WORK}/fit_sim_{car}.json", "w"), indent=1)
print("done", {k: round(v, 4) for k, v in over.items()}, f"{time.time() - t0:.0f}s", flush=True)
