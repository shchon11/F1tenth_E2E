"""Steering calibration: how much does the car actually turn for a commanded angle?

Surface independent, so every recording can be pooled. Fitted only where the lateral acceleration
is low, because there the tyres are far from slipping and the kinematic relation holds:

    omega = v * tan(gain * delta_cmd + bias) / (L + k_us * v^2)

`gain` folds together linkage ratio and servo calibration; `k_us` is the understeer term the
simulator's teacher already models. Reading the ratio at high lateral acceleration instead would
confuse calibration with slip.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .bagread import BagData

WHEELBASE = 0.3302


def _resample(bag: BagData, rate: float = 50.0):
    t0 = max(bag.t[k][0] for k in ("/drive", "/odom") if bag.has(k))
    t1 = min(bag.t[k][-1] for k in ("/drive", "/odom") if bag.has(k))
    ts = np.arange(t0, t1, 1.0 / rate)
    return ts, bag.at("/drive", ts)[:, 0], bag.at("/odom", ts)


def command_delay(bag: BagData, gyro_ok: bool = True, max_lag: float = 0.30, rate: float = 50.0) -> float:
    """Lag [s] that best aligns the steering command with the yaw-rate response."""
    ts, delta, odom = _resample(bag, rate)
    w = bag.at("/sensors/imu/raw", ts)[:, 2] if (gyro_ok and bag.has("/sensors/imu/raw")) else odom[:, 5]
    v = odom[:, 3]
    use = v > 1.0
    if use.sum() < 100:
        return float("nan")
    a = delta * v                                     # command scaled by speed ~ yaw demand
    a = a - a[use].mean(); b = w - w[use].mean()
    a[~use] = 0.0; b[~use] = 0.0
    n = int(max_lag * rate)
    scores = [float(np.dot(a[:len(a) - k], b[k:])) if k else float(np.dot(a, b)) for k in range(n + 1)]
    return float(np.argmax(scores) / rate)


def fit(bag: BagData, gyro_ok: bool = True, a_lat_max: float = 3.0, rate: float = 50.0) -> dict:
    """Least-squares (gain, bias, k_us) on the low-lateral-acceleration samples."""
    ts, delta, odom = _resample(bag, rate)
    w = bag.at("/sensors/imu/raw", ts)[:, 2] if (gyro_ok and bag.has("/sensors/imu/raw")) else odom[:, 5]
    v = odom[:, 3]
    lag = command_delay(bag, gyro_ok, rate=rate)
    shift = int(round(lag * rate))
    if shift:
        delta, v, w = delta[:-shift], v[:-shift], w[shift:]
    keep = (v > 1.5) & (np.abs(v * w) < a_lat_max)
    if keep.sum() < 200:
        return {"n": int(keep.sum()), "lag_s": lag}
    d, vv, ww = delta[keep], v[keep], w[keep]

    def resid(p):
        gain, bias, k_us = p
        return vv * np.tan(gain * d + bias) / (WHEELBASE + k_us * vv ** 2) - ww

    sol = least_squares(resid, [1.0, 0.0, 0.003], bounds=([0.2, -0.1, 0.0], [3.0, 0.1, 0.05]))
    pred = resid(sol.x) + ww
    ss = 1 - np.sum((pred - ww) ** 2) / max(np.sum((ww - ww.mean()) ** 2), 1e-9)
    return {"n": int(keep.sum()), "lag_s": lag, "gain": float(sol.x[0]), "bias_rad": float(sol.x[1]),
            "k_us": float(sol.x[2]), "r2": float(ss), "delta_cmd_max": float(np.abs(delta).max())}
