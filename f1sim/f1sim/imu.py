"""VESC built-in IMU model, integrated inside the physics substep loop (see sim.py).

Per substep (1 kHz):
  omega  = (roll_rate, pitch_rate, yaw_rate)                       body angular velocity
  a_s    = a_cog + alpha x r + omega x (omega x r)                  rigid-body accel at the sensor
  f      = a_s - g_body(roll, pitch)                                specific force (gravity leaks in)
  + tonal vibration at wheel / 2x wheel / motor frequency and broadband vibration, both ~ speed
  -> sensor misalignment -> 2nd-order low-pass at `bandwidth`
At the IMU sample instants (rate Hz): + bias (+ random walk) + white noise -> quantize.
The VESC attitude estimate integrates the gyro and slowly pulls roll/pitch toward the
accelerometer gravity direction (so it is fooled by sustained accelerations, like the real one).
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict

import torch

G = 9.81
TWO_PI = 2 * math.pi


def imu_state_init(B: int, device) -> torch.Tensor:
    """Loop-carried IMU state (B, 21): lp position (6), lp velocity (6), phases (3), gyro bias walk (3), ahrs (3)."""
    return torch.zeros(B, 21, device=device)


def sample_indices(rate: float, control_dt: float, dt: float, substeps: int):
    """Substep indices at which the IMU produces a sample within one control step (last sample at
    the end of the step, spacing 1/rate).

    Only correct when `rate * control_dt` is an integer: it assumes the same number of samples in
    every control step and that the last one lands exactly on the step boundary. 50 Hz on a 40 Hz
    loop -- what this project runs -- satisfies neither. `sample_schedule` is the general form;
    this is kept because it is still the right answer for the integer case.
    """
    K = max(1, int(math.floor(rate * control_dt + 1e-6)))
    idx = [substeps - 1 - int(round((K - 1 - k) / rate / dt)) for k in range(K)]
    return [max(0, i) for i in idx]


def sample_schedule(rate: float, control_dt: float, dt: float, substeps: int, max_period: int = 8):
    """Where a *free-running* IMU's samples fall inside each control step.

    The sensor ticks on its own clock at `rate`, which has no reason to divide the control period.
    At 50 Hz against a 40 Hz control loop the count per step is not constant: it cycles 1, 1, 1, 2
    and averages the declared rate exactly. Flooring to a fixed count instead ran the sensor at
    40 Hz and told its noise model 20 ms had passed when 25 ms had.

    Returns `(schedule, offsets, period)`:
      * `schedule[p]` -- substep indices at which cycle-step `p` emits, ascending in time
      * `offsets[p]`  -- how long before the *end* of that control step each sample falls [s], so a
                         consumer can timestamp it as `now - offset`
      * `period`      -- control steps per cycle; the layout repeats with this period

    One cycle spans a whole number of both clocks, so the sensor never drifts against the sim.
    `period` is also how many distinct layouts the substep loop can see, and each one is a separate
    compiled graph: hence `max_period`, which refuses a rate that would thrash the compile cache
    rather than silently accepting it.
    """
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError(f"imu_rate must be a positive, finite frequency; got {rate!r}")
    if rate > 1.0 / dt + 1e-9:
        raise ValueError(
            f"imu_rate {rate:g} Hz is faster than the {1 / dt:g} Hz physics tick, so two samples "
            f"would share one substep and the second would repeat the first. Lower the rate or "
            f"lower sim.physics_dt.")
    ratio = Fraction(rate * control_dt).limit_denominator(1000)    # samples per control step
    period, per_cycle = ratio.denominator, ratio.numerator
    if period > max_period:
        raise ValueError(
            f"imu_rate {rate:g} Hz against a {1 / control_dt:g} Hz control loop needs a "
            f"{period}-step cycle to stay phase-exact (limit {max_period}); pick a rate whose "
            f"ratio to the control rate is a simpler fraction")
    schedule: list[list[int]] = [[] for _ in range(period)]
    offsets: list[list[float]] = [[] for _ in range(period)]
    for j in range(per_cycle):
        t = (j + 1) / rate                                         # sensor tick, cycle-relative
        p = max(0, math.ceil(t / control_dt - 1e-9) - 1)            # the control step it lands in
        k = min(substeps - 1, max(0, int(round((t - p * control_dt) / dt)) - 1))
        schedule[p].append(k)
        offsets[p].append((p + 1) * control_dt - t)
    # A control step with no sample would make the environment drop the IMU keys from the
    # observation for that step and change the proprio width (gym_env._obs), so the observation
    # contract does not survive it. Refuse the rate instead of quietly producing a ragged one.
    if any(not idx for idx in schedule):
        empty = [p for p, idx in enumerate(schedule) if not idx]
        raise ValueError(
            f"imu_rate {rate:g} Hz is slower than the {1 / control_dt:g} Hz control loop, so "
            f"control step(s) {empty} of every {period} would carry no IMU sample. The observation "
            f"assumes at least one sample per step; supporting a slower sensor needs a held-value "
            f"or explicit-gap contract in gym_env._obs first.")
    if any(len(set(idx)) != len(idx) for idx in schedule):
        raise ValueError(
            f"imu_rate {rate:g} Hz puts two samples in the same physics substep; lower sim.physics_dt")
    return schedule, offsets, period


def specific_force(ax, ay, roll, pitch, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc, r_vec):
    """Accelerometer truth in the (tilted) body frame at sensor offset r_vec (B,3) from the CoG."""
    rx, ry, rz = r_vec[:, 0], r_vec[:, 1], r_vec[:, 2]
    wx, wy, wz = roll_rate, pitch_rate, yaw_rate
    # alpha x r
    a1x = pitch_acc * rz - yaw_acc * ry
    a1y = yaw_acc * rx - roll_acc * rz
    a1z = roll_acc * ry - pitch_acc * rx
    # omega x (omega x r) = omega (omega.r) - r |omega|^2
    wr = wx * rx + wy * ry + wz * rz
    w2 = wx * wx + wy * wy + wz * wz
    a2x = wx * wr - rx * w2
    a2y = wy * wr - ry * w2
    a2z = wz * wr - rz * w2
    sr, cr, sp, cp = torch.sin(roll), torch.cos(roll), torch.sin(pitch), torch.cos(pitch)
    # f = a - g_body with g_body = R^T (0,0,-g) = (g sin(pitch), -g cos(pitch) sin(roll), -g cos(pitch) cos(roll))
    fx = ax + a1x + a2x - G * sp
    fy = ay + a1y + a2y + G * cp * sr
    fz = a1z + a2z + G * cp * cr
    return fx, fy, fz


def misalign(vx, vy, vz, P: Dict[str, torch.Tensor]):
    """Small-angle rotation into the sensor frame: v' = v + v x theta."""
    tx, ty, tz = P["imu_roll"], P["imu_pitch"], P["imu_yaw"]
    return (vx + (vy * tz - vz * ty), vy + (vz * tx - vx * tz), vz + (vx * ty - vy * tx))


def substep(imu_state, ax, ay, vx, roll, pitch, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc,
            r_vec, P: Dict[str, torch.Tensor], dt: float):
    """Advance vibration phases and the sensor low-pass by one physics substep. Returns new state."""
    lp, lpv, ph = imu_state[:, 0:6], imu_state[:, 6:12], imu_state[:, 12:15]
    fx, fy, fz = specific_force(ax, ay, roll, pitch, roll_rate, pitch_rate, yaw_rate, roll_acc, pitch_acc, yaw_acc, r_vec)
    gx, gy, gz = roll_rate, pitch_rate, yaw_rate
    # vibration: wheel, 2x wheel, motor tones (phase accumulators) + broadband, amplitude ~ speed
    speed = vx.abs()
    f_w = speed / (math.pi * P["tire_d"])
    ph = ph + TWO_PI * dt * torch.stack([f_w, 2 * f_w, f_w * P["gear_ratio"]], 1)
    ph = torch.remainder(ph, TWO_PI)
    tone = torch.sin(ph[:, 0]) + 0.5 * torch.sin(ph[:, 1] + 1.0) + 0.7 * torch.sin(ph[:, 2] + 2.0)
    tone_y = torch.sin(ph[:, 0] + 0.7) + 0.5 * torch.sin(ph[:, 1] + 2.1) + 0.7 * torch.sin(ph[:, 2] + 0.3)
    tone_z = torch.sin(ph[:, 0] + 1.9) + 0.5 * torch.sin(ph[:, 1] + 0.4) + 0.7 * torch.sin(ph[:, 2] + 1.4)
    bb = P["vib_broadband"]
    # Measured over 22 recordings as the rms of the IMU residual above 5 Hz (vehicle motion lives
    # below that; a 0.2 s moving average, used first, leaves manoeuvre in the residual and made the
    # standstill bin 46x too noisy). Vibration is not proportional to speed and it is not constant
    # either -- it switches on the moment the wheels turn and then grows weakly:
    #   accel rms   0.018 m/s^2 truly still -> 1.42 at 0.5 m/s -> 3.3 at 7.5   (~80x at first motion)
    #   gyro  rms   0.0025 rad/s            -> 0.150           -> 0.41
    # `coefficient * speed` alone is silent exactly where the policy reads its proprio history to
    # infer grip and lag; a speed-independent floor instead leaves a stationary car buzzing at 80x
    # its real noise. So: a floor that ramps in over the first `vib_onset_v` of wheel speed, plus a
    # slope on top. Both coefficients are fitted to the *output* of this chain, not to the injection
    # -- the 40 Hz low-pass and 50 Hz sampling below remove a good part of what is injected here.
    # The floor is broadband, the slope carries the tones. Road texture and motor idle have no
    # reason to sit at the wheel frequency, and putting the floor on the tone makes it invisible
    # where it matters: at 0.5 m/s the wheel turns at 1.6 Hz, below the band this was fitted in,
    # so a tonal floor left the sim 30 % quiet there. (An earlier version multiplied the floor by
    # the tone *and* let it act at a standstill, where the phase is frozen -- that gives a constant
    # offset, not vibration, and read 0.6 rad/s of yaw on a parked car.)
    onset = (speed / P["vib_onset_v"]).clamp(max=1.0)
    va_f, va_s = P["vib_accel_floor"] * onset, P["vib_accel"] * speed
    vg_f, vg_s = P["vib_gyro_floor"] * onset, P["vib_gyro"] * speed
    n = torch.randn(vx.shape[0], 6, device=vx.device)
    m = torch.randn(vx.shape[0], 6, device=vx.device)
    fx = fx + va_s * (0.6 * (1 - bb) * tone + bb * n[:, 0]) + va_f * m[:, 0]
    fy = fy + va_s * (0.6 * (1 - bb) * tone_y + bb * n[:, 1]) + va_f * m[:, 1]
    fz = fz + va_s * (1.0 * (1 - bb) * tone_z + bb * n[:, 2]) + va_f * m[:, 2]
    gx = gx + vg_s * (0.8 * (1 - bb) * tone_y + bb * n[:, 3]) + vg_f * m[:, 3]
    gy = gy + vg_s * (0.8 * (1 - bb) * tone_z + bb * n[:, 4]) + vg_f * m[:, 4]
    gz = gz + vg_s * (0.5 * (1 - bb) * tone + bb * n[:, 5]) + vg_f * m[:, 5]
    gx, gy, gz = misalign(gx, gy, gz, P)
    fx, fy, fz = misalign(fx, fy, fz, P)
    u = torch.stack([gx, gy, gz, fx, fy, fz], 1)
    # 2nd-order low-pass (zeta 0.7) at the sensor bandwidth
    wc = (TWO_PI * P["bandwidth"])[:, None]
    lpa = wc * wc * (u - lp) - 2 * 0.7 * wc * lpv
    lpv = lpv + lpa * dt
    lp = lp + lpv * dt
    return torch.cat([lp, lpv, ph, imu_state[:, 15:21]], 1)


def sample(imu_state, P: Dict[str, torch.Tensor], ts: float):
    """Produce one IMU sample (B,6) = gyro xyz [rad/s], accel xyz [m/s^2] from the filtered truth,
    and update the gyro bias walk + attitude estimate. ts: sample period."""
    lp = imu_state[:, 0:6]
    bias_walk = imu_state[:, 15:18]
    ahrs = imu_state[:, 18:21]
    B = lp.shape[0]
    bias_walk = bias_walk + torch.randn(B, 3, device=lp.device) * (P["gyro_bias_walk"] * math.sqrt(ts))[:, None]
    gbias = torch.stack([P["gyro_bias_x"], P["gyro_bias_y"], P["gyro_bias_z"]], 1) + bias_walk
    abias = torch.stack([P["accel_bias_x"], P["accel_bias_y"], P["accel_bias_z"]], 1)
    noise = torch.randn(B, 6, device=lp.device)
    gyro = lp[:, 0:3] + gbias + noise[:, 0:3] * P["gyro_noise"][:, None]
    acc = lp[:, 3:6] + abias + noise[:, 3:6] * P["accel_noise"][:, None]
    qg, qa = P["quant_gyro"][:, None], P["quant_accel"][:, None]
    gyro = torch.round(gyro / qg) * qg
    acc = torch.round(acc / qa) * qa
    # VESC attitude filter: integrate gyro, pull roll/pitch toward the accelerometer tilt
    roll_e, pitch_e, yaw_e = ahrs[:, 0], ahrs[:, 1], ahrs[:, 2]
    roll_e = roll_e + gyro[:, 0] * ts
    pitch_e = pitch_e + gyro[:, 1] * ts
    yaw_e = yaw_e + gyro[:, 2] * ts
    roll_acc = torch.atan2(acc[:, 1], acc[:, 2])
    pitch_acc = torch.atan2(-acc[:, 0], torch.sqrt(acc[:, 1] ** 2 + acc[:, 2] ** 2))
    conf = (1.0 - (torch.linalg.norm(acc, dim=1) / G - 1.0).abs() * P["ahrs_accel_decay"]).clamp(0.0, 1.0)
    k = (ts / P["ahrs_tau"]).clamp(max=1.0) * conf
    roll_e = roll_e + k * (roll_acc - roll_e)
    pitch_e = pitch_e + k * (pitch_acc - pitch_e)
    yaw_e = torch.remainder(yaw_e + math.pi, TWO_PI) - math.pi
    new_state = torch.cat([imu_state[:, 0:15], bias_walk, torch.stack([roll_e, pitch_e, yaw_e], 1)], 1)
    return torch.cat([gyro, acc], 1), new_state
