"""Replay windows from the bags (audit). Every window: 0.5 s warm-up history + 2.0 s horizon at 100 Hz."""
import os

import numpy as np

import sig

GYRO_DEG = ("20260714-223333", "20260714-223707", "20260725-014941", "20260725-020031",
            "20260725-015428", "20260725-020831", "20260725-021025")
WARM, HOR = 50, 200            # samples at 100 Hz
LR = 0.17145


def bag_signals(f):
    Z = np.load(f)
    nm = os.path.basename(f)
    m = sig.mapping(Z)
    if not abs(m["servo_g"]) > 0.1:            # straight-only bag: no steering spread to fit, take the era's mapping
        m["servo_g"], m["servo_o"] = (0.6300, 0.3953) if "map08x23" in nm else (-0.5300, 0.6054)
    S = sig.signals(Z, m["servo_g"], m["servo_o"], m["erpm_k"],
                    gyro_scale=(np.pi / 180 if nm[:15] in GYRO_DEG else 1.0))
    # the steering the servo actually got (VESC clamps /commands/servo/position to its servo window)
    sp = Z["servo_pos"]
    S["steer_app"] = (sig.hold(sp[:, 0], sp[:, 1], S["t"]) - m["servo_o"]) / m["servo_g"]
    S["map"] = m
    return S


def build(fs, stride=25, v_min=1.0):
    rows = []
    for f in fs:
        nm = os.path.basename(f)
        S = bag_signals(f)
        n = len(S["t"])
        r_s = sig.smooth(S["gz"], 10); ay_s = sig.smooth(S["ay"], 10)
        a_imp = sig.smooth(np.hypot(S["ax"], S["ay"]), 5)
        imp = a_imp > 15.0
        has_pf = "px" in S
        if has_pf:                                 # particle filter must agree with the gyro, else it was lost
            pr = sig.deriv(sig.smooth(S["pyaw"], 5), 5); ok = (S["v_wheel"] > 1.0) & np.isfinite(pr)
            has_pf = ok.sum() > 100 and np.corrcoef(r_s[ok], pr[ok])[0, 1] > 0.7
        n_imp = 0
        for i0 in range(WARM + 1, n - HOR - 6, stride):
            sl = slice(i0 - WARM, i0 + HOR + 1)
            if S["v_wheel"][i0] < v_min or S["v_wheel"][i0:i0 + HOR + 1].min() < 0.3:
                continue
            if S["cmd_age"][sl].max() > 0.1:
                continue
            if imp[sl].any():
                n_imp += 1
                continue
            w = dict(bag=nm[:15], t0=S["t"][i0],
                     dcmd=S["steer_app"][sl], dcmd_raw=S["steer_cmd"][sl], vcmd=S["speed_cmd"][sl],
                     r=r_s[sl], v=S["v_wheel"][sl], ay=ay_s[sl], ax=sig.smooth(S["ax"], 10)[sl])
            if has_pf:
                w["px"], w["py"], w["pyaw"] = S["px"][sl], S["py"][sl], S["pyaw"][sl]
            rows.append(w)
        print(f"  {nm[:45]:45s} windows {sum(1 for r in rows if r['bag'] == nm[:15]):5d}  impact-excluded {n_imp}", flush=True)
    W = {k: np.stack([r[k] for r in rows]) for k in ("dcmd", "dcmd_raw", "vcmd", "r", "v", "ay", "ax")}
    W["has_pf"] = np.array(["px" in r for r in rows])
    for k in ("px", "py", "pyaw"):
        W[k] = np.stack([r.get(k, np.zeros(WARM + HOR + 1)) for r in rows])
    W["bag"] = np.array([r["bag"] for r in rows]); W["t0"] = np.array([r["t0"] for r in rows])
    return W


def splits():
    comp = sig.files("comp")
    fit = comp[0::2]; val = comp[1::2]          # the 11-09 split (fit / held-out)
    pre = [f for f in sig.files("pre") if "021025" not in f]
    return fit, val, pre
