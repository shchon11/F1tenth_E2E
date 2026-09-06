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
from typing import Dict

import torch

G = 9.81
TWO_PI = 2 * math.pi


def imu_state_init(B: int, device) -> torch.Tensor:
    """Loop-carried IMU state (B, 21): lp position (6), lp velocity (6), phases (3), gyro bias walk (3), ahrs (3)."""
    return torch.zeros(B, 21, device=device)


def sample_indices(rate: float, control_dt: float, dt: float, substeps: int):
    """Substep indices at which the IMU produces a sample within one control step (last sample at
    the end of the step, spacing 1/rate)."""
    K = max(1, int(math.floor(rate * control_dt + 1e-6)))
    idx = [substeps - 1 - int(round((K - 1 - k) / rate / dt)) for k in range(K)]
    return [max(0, i) for i in idx]


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
    va = P["vib_accel"] * speed
    vg = P["vib_gyro"] * speed
    n = torch.randn(vx.shape[0], 6, device=vx.device)
    fx = fx + va * (0.6 * (1 - bb) * tone + bb * n[:, 0])
    fy = fy + va * (0.6 * (1 - bb) * tone_y + bb * n[:, 1])
    fz = fz + va * (1.0 * (1 - bb) * tone_z + bb * n[:, 2])
    gx = gx + vg * (0.8 * (1 - bb) * tone_y + bb * n[:, 3])
    gy = gy + vg * (0.8 * (1 - bb) * tone_z + bb * n[:, 4])
    gz = gz + vg * (0.5 * (1 - bb) * tone + bb * n[:, 5])
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
