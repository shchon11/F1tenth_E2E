"""Common 100 Hz signals from the audit npz files (own conventions, re-derived here)."""
import glob
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "npz")
HZ = 100.0
L = 0.3302


def files(group):
    fs = sorted(glob.glob(os.path.join(NPZ, "*.npz")))
    if group == "comp":
        return [f for f in fs if "map08x23" in f and "STATIONARY" not in f]
    if group == "pre":
        return [f for f in fs if "map08x23" not in f]
    return fs


def hold(ts, xs, t):
    i = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    return xs[i]


def fit_lin(x, y):
    A = np.stack([x, np.ones_like(x)], 1)
    g, o = np.linalg.lstsq(A, y, rcond=None)[0]
    return g, o


def mapping(Z):
    """servo = g*steer + o and erpm = k*speed, fitted from /drive (or /teleop merged) vs /commands/*.
    The command closest in time AFTER each /drive message (the mux forwards within a few ms)."""
    d = Z["drive"]; sc = Z["servo_cmd"]; ec = Z["erpm_cmd"]
    j = np.searchsorted(sc[:, 0], d[:, 0])
    ok = (j < len(sc)) & (np.abs(sc[np.clip(j, 0, len(sc) - 1), 0] - d[:, 0]) < 0.01)
    x, y = d[ok, 2], sc[j[ok], 1]
    g, o = fit_lin(x, y)
    res = y - (g * x + o)
    keep = np.abs(res) < 0.01
    g, o = fit_lin(x[keep], y[keep])
    j2 = np.searchsorted(ec[:, 0], d[:, 0])
    ok2 = (j2 < len(ec)) & (np.abs(ec[np.clip(j2, 0, len(ec) - 1), 0] - d[:, 0]) < 0.01)
    xs, ys = d[ok2, 3], ec[j2[ok2], 1]
    m = np.abs(xs) > 0.5
    k = np.median(ys[m] / xs[m]) if m.sum() > 10 else np.nan
    return dict(servo_g=g, servo_o=o, servo_frac=keep.mean(), erpm_k=k, n=len(x))


def signals(Z, servo_g, servo_o, erpm_k, gyro_scale=1.0):
    c = Z["core"] if "core" in Z else None
    im = Z["imu"]
    t_beg = max(im[0, 0], Z["servo_cmd"][0, 0], c[0, 0] if c is not None else 0)
    t_end = min(im[-1, 0], Z["servo_cmd"][-1, 0] + 0.5, c[-1, 0] if c is not None else 1e9)
    t = np.arange(t_beg, t_end, 1 / HZ)
    S = {"t": t}
    S["steer_cmd"] = (hold(Z["servo_cmd"][:, 0], Z["servo_cmd"][:, 1], t) - servo_o) / servo_g
    S["speed_cmd"] = hold(Z["erpm_cmd"][:, 0], Z["erpm_cmd"][:, 1], t) / erpm_k
    # commands stop being sent when nothing drives: mark stale (>0.2 s since last command)
    i = np.clip(np.searchsorted(Z["servo_cmd"][:, 0], t, side="right") - 1, 0, len(Z["servo_cmd"]) - 1)
    S["cmd_age"] = t - Z["servo_cmd"][i, 0]
    if c is not None:
        S["v_wheel"] = np.interp(t, c[:, 0], c[:, 2]) / erpm_k
        S["i_motor"] = np.interp(t, c[:, 0], c[:, 3])
    else:
        o = Z["odom"]
        S["v_wheel"] = np.interp(t, o[:, 0], o[:, 5])
        S["i_motor"] = np.full_like(t, np.nan)
    for j, k in enumerate(("gx", "gy", "gz")):
        S[k] = np.interp(t, im[:, 0], im[:, 2 + j]) * gyro_scale
    for j, k in enumerate(("ax", "ay", "az")):
        S[k] = np.interp(t, im[:, 0], im[:, 5 + j]) * 9.80665
    still = np.abs(S["v_wheel"]) < 0.02
    n = int(HZ)
    run = np.convolve(still.astype(float), np.ones(n), "same") >= n - 0.5
    S["still"] = run
    S["bias"] = {}
    if run.sum() > HZ:
        for k in ("gx", "gy", "gz", "ax", "ay"):
            b = float(np.median(S[k][run])); S["bias"][k] = b; S[k] = S[k] - b
    if "pf" in Z:
        p = Z["pf"]
        S["px"] = np.interp(t, p[:, 0], p[:, 2]); S["py"] = np.interp(t, p[:, 0], p[:, 3])
        S["pyaw"] = np.interp(t, p[:, 0], np.unwrap(p[:, 4]))
    o = Z["odom"]
    S["odom_v"] = np.interp(t, o[:, 0], o[:, 5]); S["odom_wz"] = np.interp(t, o[:, 0], o[:, 7])
    return S


def deriv(x, n=5):
    d = np.full_like(x, np.nan)
    d[n:-n] = (x[2 * n:] - x[:-2 * n]) * HZ / (2 * n)
    return d


def smooth(x, n):
    return np.convolve(x, np.ones(n) / n, "same") if n > 1 else x
