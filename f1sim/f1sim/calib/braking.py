"""Braking authority from recorded decelerations.

Two things differ between the recording days and must not be pooled blindly:
  * the floor (tyre grip), which caps how much of the brake can be used;
  * the VESC regenerative current limit, which caps the brake itself
    (-30 to -49 A at the competition, -3 to -13 A before it).
Deceleration is therefore reported next to the motor current that produced it, so an actuator limit
can be told apart from a grip limit.
"""
from __future__ import annotations

import numpy as np

from .bagread import BagData


def smooth(x: np.ndarray, sec: float, dt: float) -> np.ndarray:
    k = max(1, int(sec / dt))
    return np.convolve(x, np.ones(k) / k, mode="same")


def braking(bag: BagData, gyro_ok: bool = True, dt: float = 0.02, win: float = 0.2) -> dict:
    """Sustained decelerations, from the IMU and from wheel speed, with the current that caused them."""
    keys = [k for k in ("/odom", "/sensors/imu/raw") if bag.has(k)]
    if len(keys) < 2:
        return {}
    t0 = max(bag.t[k][0] for k in keys); t1 = min(bag.t[k][-1] for k in keys)
    ts = np.arange(t0, t1, dt)
    v = bag.at("/odom", ts)[:, 3]
    ax_imu = smooth(bag.at("/sensors/imu/raw", ts)[:, 3], win, dt)
    ax_wheel = smooth(np.gradient(v, ts), win, dt)
    cur = smooth(bag.at("/sensors/core", ts)[:, 0], win, dt) if bag.has("/sensors/core") else None

    fast = v > 2.0                                        # braking from a speed worth braking from
    if fast.sum() < 25:
        return {"n": int(fast.sum())}
    braking_now = fast & (ax_wheel < -0.5)
    out = {
        "n": int(fast.sum()),
        "a_brake_imu_p1": float(np.percentile(ax_imu[fast], 1)),
        "a_brake_wheel_p1": float(np.percentile(ax_wheel[fast], 1)),
        "a_brake_wheel_min": float(ax_wheel[fast].min()),
        "v_at_brake": float(np.median(v[braking_now])) if braking_now.any() else float("nan"),
    }
    if cur is not None:
        out["current_min"] = float(np.percentile(cur, 1))
        out["current_at_hard_brake"] = float(np.median(cur[braking_now & (ax_wheel < np.percentile(ax_wheel[fast], 5))])) \
            if (braking_now & (ax_wheel < np.percentile(ax_wheel[fast], 5))).any() else float("nan")
    return out
