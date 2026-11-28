import json, pickle, numpy as np, torch
from common import WORK, SPLIT
from f1sim.bag_profile import models as M, windows as WN, fit as F
torch.set_num_threads(1)
D = pickle.load(open(f"{WORK}/windows.pkl", "rb"))
W = WN.concat([D[k]["W"] for k in SPLIT["comp"]["train"]])
idx = np.arange(0, len(W["r"]), 5)
R = json.load(open(f"{WORK}/fit1_comp.json")); b = R["st_C13"]["p"]
base = dict(steer_gain=b["k"], steer_bias=b["b"], cmd_delay=0.043, servo_tau=b["tau_s"], mu=b["mr"], mu_ref=b["mr"],
            mu_f_scale=b["mf"] / b["mr"], B_f=b["sf"] / (b["mf"] * 1.3), C_f=1.3, B_r=b["sr"] / (b["mr"] * 1.3), C_r=1.3,
            motor_tau=0.09, motor_delay=0.078)
def J(pr):
    e = pr["r"][:, 1:101] - W["r"][idx, 51:151]; return float(np.mean(e ** 2))
print("(b') itself", J(M.st_rollout(W, b, idx=idx, hor=100)))
for tag, ov, wheel in (("sim mapped", {}, None), ("wheel off", {}, False), ("h=0", dict(h=0.0), None),
                       ("h=0 wheel off", dict(h=0.0), False), ("h=0 wheel off blend .5-1.5", dict(h=0.0, v_blend_min=0.5, v_blend_max=1.5), False)):
    pr = M.sim_rollout(W, over={**base, **ov}, idx=idx, hor=100, wheel=wheel)
    print(tag, round(J(pr), 5), "spin", M.spin_flags(pr).mean().round(3), flush=True)
