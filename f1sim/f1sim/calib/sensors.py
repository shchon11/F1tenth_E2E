"""LiDAR and IMU noise from the recordings, in the same terms `params.py` randomises."""
from __future__ import annotations

import numpy as np

from .bagread import BagData

G = 9.80665


def truly_still(bag: BagData, ts: np.ndarray, speed_max: float = 0.15, current_max: float = 2.0,
                gravity_tol: float = 0.5, win: float = 0.5, dt: float = 0.02) -> np.ndarray:
    """Samples where the car is actually at rest on its wheels.

    `/odom` alone is not enough: it is VESC wheel speed, so a car being carried, lifted onto a stand
    or shoved against a wall reads ~0 while the IMU sees plenty. Screening on wheel speed only, the
    "stationary" gyro spread of the competition recordings came out 20-40x that of the others -- all
    of it handling, not sensor noise. Three conditions together: wheels not turning, motor not
    pulling, and the accelerometer still seeing 1 g down.
    """
    still = bag.at("/odom", ts)[:, 3] < speed_max
    if bag.has("/sensors/core"):
        still &= np.abs(bag.at("/sensors/core", ts)[:, 0]) < current_max
    if bag.has("/sensors/imu/raw"):
        a = bag.at("/sensors/imu/raw", ts)[:, 3:]
        still &= np.abs(np.linalg.norm(a, axis=1) - G) < gravity_tol      # upright and unshaken
    k = max(1, int(win / dt))                                             # drop the edges of a
    if k > 1:                                                             # handling event, not just its peak
        still = np.convolve(still.astype(float), np.ones(k) / k, mode="same") > 0.999
    return still


def imu_noise(bag: BagData, still_speed: float = 0.15, dt: float = 0.02) -> dict:
    """Bias, white noise and vibration, split by whether the car is standing still."""
    if not bag.has("/odom", "/sensors/imu/raw"):
        return {}
    t0 = max(bag.t["/odom"][0], bag.t["/sensors/imu/raw"][0])
    t1 = min(bag.t["/odom"][-1], bag.t["/sensors/imu/raw"][-1])
    ts = np.arange(t0, t1, dt)
    v = bag.at("/odom", ts)[:, 3]
    imu = bag.at("/sensors/imu/raw", ts)
    still, moving = truly_still(bag, ts, speed_max=still_speed, dt=dt), v > 1.0
    out = {"still_s": float(still.sum() * dt), "moving_s": float(moving.sum() * dt)}
    if still.sum() > 50:
        g, a = imu[still][:, :3], imu[still][:, 3:]
        out.update(gyro_bias=g.mean(0).tolist(), gyro_noise=g.std(0).tolist(),
                   accel_bias_xy=a[:, :2].mean(0).tolist(), accel_noise=a.std(0).tolist(),
                   gravity_z=float(a[:, 2].mean()))
    if moving.sum() > 50:
        # vibration = what is left after removing the manoeuvre (0.2 s moving average)
        k = max(1, int(0.2 / dt))
        ker = np.ones(k) / k
        res_a = np.stack([imu[:, 3 + i] - np.convolve(imu[:, 3 + i], ker, "same") for i in range(3)], 1)
        res_g = np.stack([imu[:, i] - np.convolve(imu[:, i], ker, "same") for i in range(3)], 1)
        out.update(vib_accel=res_a[moving].std(0).tolist(), vib_gyro=res_g[moving].std(0).tolist())
    return out


def lidar_stats(bag: BagData, range_max: float = 10.0, sentinel: float = 65.0) -> dict:
    """Beam count, no-return rate and the range distribution of the valid returns.

    A miss is reported as 65.533 m (the 0xFFFF mm sentinel), which is finite and positive, so
    screening on `isfinite(r) & (r > 0)` counts every miss as a good return and reports a
    no-return rate of 0. That is how the sim ended up with `dropout_prob` fitted to nothing.
    """
    if not bag.has("/scan"):
        return {}
    r = bag.v["/scan"]
    miss = ~np.isfinite(r) | (r <= 0) | (r >= sentinel)
    good = ~miss
    return {"scans": int(r.shape[0]), "beams": int(r.shape[1]),
            "no_return_rate": float(miss.mean()),
            "sentinel_rate": float((np.isfinite(r) & (r >= sentinel)).mean()),
            "nonfinite_rate": float((~np.isfinite(r)).mean()),
            "beyond_range_max_rate": float((good & (r >= range_max)).mean()),
            "r_p01": float(np.percentile(r[good], 1)), "r_p50": float(np.percentile(r[good], 50)),
            "r_p99": float(np.percentile(r[good], 99)), "r_max": float(r[good].max())}


def lidar_range_noise(bag: BagData, still: np.ndarray, bins=(0, 1, 2, 3, 5, 8, 12, 20, 40),
                      sentinel: float = 65.0, min_scans: int = 30) -> dict:
    """Per-beam range noise against distance, over a window where the car is truly still.

    Reported per bin as the *median* MAD across beams, not the pooled spread: within one bin most
    beams see a flat surface and a handful straddle an edge or a passing person, and those few
    dominate any pooled statistic (the first pass at this read 1838 mm of "noise" that way).
    """
    if not bag.has("/scan") or still.sum() < min_scans:
        return {}
    r = np.asarray(bag.v["/scan"], dtype=float)[still]
    r[~np.isfinite(r) | (r <= 0) | (r >= sentinel)] = np.nan
    med = np.nanmedian(r, 0)                                          # (beams,) mean range per beam
    mad = 1.4826 * np.nanmedian(np.abs(r - med), 0)                   # (beams,) noise per beam
    ok = np.isfinite(med) & np.isfinite(mad) & (np.isfinite(r).sum(0) > 0.9 * r.shape[0])
    out = {"scans": int(still.sum()), "beams_used": int(ok.sum())}
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = ok & (med >= lo) & (med < hi)
        if sel.sum() < 5:
            continue
        rows.append((lo, hi, int(sel.sum()), float(np.median(med[sel])),
                     float(np.median(mad[sel])), float(np.percentile(mad[sel], 90))))
    out["bins"] = rows                                                # (lo, hi, n, r, mad_med, mad_p90)
    return out
