"""What acceleration does the real car actually achieve?

The raceline speed profile assumes a_lat = 6.0, a_acc = 4.0, a_brake = 3.0 m/s^2, and every speed
decision downstream rests on those three numbers. None of them had been measured.

Two independent estimates of lateral acceleration are reported because each is biased:
  * IMU a_y is the direct measurement but carries mounting tilt and vibration;
  * v * yaw_rate is the centripetal term only, so it overstates a_y through a turn-in where the
    lateral velocity derivative works against it.
Agreement between them is the check.
"""
from __future__ import annotations

import numpy as np

from .bagread import BagData

YAW_TOPIC_PREFERENCE = ("/sensors/imu/raw", "/odom", "/imu/filtered_angular_velocity")


def yaw_rate(bag: BagData, times: np.ndarray, gyro_ok: bool = True) -> tuple[np.ndarray, str]:
    """Yaw rate, preferring the raw gyro but falling back when it is the corrupted one."""
    if gyro_ok and bag.has("/sensors/imu/raw"):
        return bag.at("/sensors/imu/raw", times)[:, 2], "/sensors/imu/raw"
    if bag.has("/imu/filtered_angular_velocity"):
        return bag.at("/imu/filtered_angular_velocity", times), "/imu/filtered_angular_velocity"
    return bag.at("/odom", times)[:, 5], "/odom"


def envelope(bag: BagData, gyro_ok: bool = True, rate: float = 50.0, v_min: float = 1.0) -> dict:
    """Achieved acceleration envelope over the moving part of the recording."""
    t0 = max(bag.t[k][0] for k in ("/odom", "/sensors/imu/raw") if bag.has(k))
    t1 = min(bag.t[k][-1] for k in ("/odom", "/sensors/imu/raw") if bag.has(k))
    times = np.arange(t0, t1, 1.0 / rate)
    odom = bag.at("/odom", times)
    v = odom[:, 3]
    w, w_src = yaw_rate(bag, times, gyro_ok)
    imu = bag.at("/sensors/imu/raw", times) if bag.has("/sensors/imu/raw") else None

    moving = v > v_min
    a_lat_centripetal = np.abs(v * w)
    a_lat_imu = np.abs(imu[:, 4]) if imu is not None else None

    # longitudinal from wheel speed, median-filtered: the raw derivative carries ERPM spikes that
    # reach -12 to -15 m/s^2, which the tyre cannot produce
    dv = np.gradient(v, times)
    k = 5
    pad = np.pad(dv, (k // 2, k // 2), mode="edge")
    dv_med = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)

    def pct(x, q):
        return float(np.percentile(x[moving], q)) if moving.sum() else float("nan")

    return {
        "yaw_source": w_src,
        "duration_s": float(times[-1] - times[0]),
        "moving_fraction": float(moving.mean()),
        "v_max": float(v.max()),
        "a_lat_centripetal_p99": pct(a_lat_centripetal, 99),
        "a_lat_centripetal_max": float(a_lat_centripetal[moving].max()) if moving.any() else float("nan"),
        "a_lat_imu_p99": pct(a_lat_imu, 99) if a_lat_imu is not None else None,
        "a_lat_imu_max": float(a_lat_imu[moving].max()) if (a_lat_imu is not None and moving.any()) else None,
        "a_acc_p99": pct(dv_med, 99),
        "a_brake_p1": pct(dv_med, 1),
        "yaw_rate_p99": pct(np.abs(w), 99),
    }
