"""map16x07 SPIN-AT-LIMIT (the only bag where the real car lost grip): which peak friction makes the adopted model
(b') lose grip where and how the car did? Windows every 0.1 s through the event, no impact exclusion (checked: no
contact -- the scan-matched rotation in the README), the pre-car values from the map12x16 transfer refit."""
import json, numpy as np
from common import WORK
from f1sim.bag_profile import models as M, windows as WN, signals as SG
key = "20260727-204907"
Z = np.load(f"{WORK}/grid_{key}.npz"); S = {k: Z[k] for k in Z.files}
t = S["t"]; v = S["v_wheel"]
print("a_imp max", np.max(SG.smooth(np.hypot(S["ax"], S["ay"]), 5)).round(1), "at", t[np.argmax(SG.smooth(np.hypot(S["ax"], S["ay"]), 5))].round(2))
i_ev = np.flatnonzero((t > 7.0) & (t < 10.0))
for i in i_ev[::20]:
    print(f"t {t[i]:5.2f} v {v[i]:4.2f} steer {S['steer'][i]:+.3f} r {S['r'][i]:+.2f} kin {v[i] * np.tan(S['steer'][i]) / 0.3302:+.2f} ay {S['ay_s'][i]:+.1f}")
tr = json.load(open(f"{WORK}/transfer.json"))
pp = dict(tr["comp->map12x16"]["refit"]["b + 지연 τd"]["p"])
spd = json.load(open(f"{WORK}/fit1_comp.json"))["speed"]["p"]
base = {**M.ST_DEFAULT, **spd, **pp, "Cf": 1.3, "Cr": 1.3}
# windows: t0 from 7.5 to 9.5 s every 0.1 s, horizon 2 s
W = {}
starts = np.flatnonzero((t >= 7.5) & (t <= 9.6))[::10]
span = np.arange(-50, 201)
G = starts[:, None] + span[None]
src = dict(steer=S["steer"], steer_cmd=S["steer_cmd"], speed_cmd=S["speed_cmd"], r=S["r"], v=v, ay=S["ay_s"], ax=S["ax_s"])
for k, a in src.items():
    W[k] = a[G]
for k in ("px", "py", "pyaw"):
    W[k] = S[k][G]
W["has_pf"] = np.ones(len(starts), bool); W["bag"] = np.array([key] * len(starts)); W["t0"] = t[starts]
real = np.degrees(np.abs(WN.real_sideslip_change(W)).max(1))
out = {}
for mf, mr in ((1.4, 1.35), (1.2, 1.2), (1.1, 1.1), (1.0, 1.0), (0.9, 0.9), (1.4, 1.1), (1.1, 1.35), (1.0, 1.35), (2.25, 1.35)):
    pr = M.st_rollout(W, {**base, "mf": mf, "mr": mr})
    e = pr["r"][:, 1:201] - W["r"][:, 51:251]
    yaw_rms = float(np.sqrt(np.mean(e ** 2)))
    beta = np.degrees(np.abs(pr["beta"]).max(1))
    out[f"{mf}/{mr}"] = dict(yaw_rms_2s=yaw_rms, model_beta_max=beta.tolist(), real_dbeta=real.tolist())
    print(f"mu_f {mf} mu_r {mr}: 2 s yaw RMS {yaw_rms:.3f} | model |beta|max p50 {np.median(beta):.0f} max {beta.max():.0f} deg | "
          f"windows with model beta>20: {(beta > 20).mean():.2f} (real dbeta>20: {(real > 20).mean():.2f})", flush=True)
json.dump(out, open(f"{WORK}/spin_bag.json", "w"), indent=1)
