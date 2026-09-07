"""VESC-style odometry (vesc_to_odom). The IMU model lives in imu.py."""
from __future__ import annotations

from typing import Dict

import torch

from .dynamics import wrap_angle


class VescOdom:
    """Dead-reckoned pose from measured speed (ERPM) and *commanded* steering angle,
    exactly like the f1tenth vesc_to_odom node. State (B, 5): x, y, yaw, v, yaw_rate."""

    def __init__(self, num_envs: int, device):
        self.state = torch.zeros(num_envs, 5, device=device)

    def reset(self, env_ids: torch.Tensor, pose: torch.Tensor):
        self.state[env_ids, :3] = pose
        self.state[env_ids, 3:] = 0.0

    def update(self, vx: torch.Tensor, steer_cmd: torch.Tensor, P: Dict[str, torch.Tensor], dt: float):
        self.state = self.update_pure(self.state, vx, steer_cmd, P, dt)
        return self.state

    @staticmethod
    def update_pure(state: torch.Tensor, vx: torch.Tensor, steer_cmd: torch.Tensor, P: Dict[str, torch.Tensor], dt: float):
        """One dead-reckoning step as a pure function of the previous odometry state (compilable)."""
        v_meas = vx * (1.0 + P["speed_scale_err"]) + torch.randn_like(vx) * P["speed_noise_std"]
        # the operator calibrates steering_angle_to_servo_offset/_gain against the physical servo, so
        # odom knows steer_bias and steer_gain up to residuals (steer_offset, steer_gain_err)
        steer_est = (steer_cmd * P["steer_gain"] + P["steer_bias"]) * (1.0 + P["steer_gain_err"]) + P["steer_offset"]
        L = P["lf"] + P["lr"]
        yaw_rate = v_meas * torch.tan(steer_est) / L + torch.randn_like(vx) * P["yaw_rate_noise_std"]
        yaw = wrap_angle(state[:, 2] + yaw_rate * dt)
        return torch.stack([state[:, 0] + v_meas * torch.cos(yaw) * dt, state[:, 1] + v_meas * torch.sin(yaw) * dt, yaw, v_meas, yaw_rate], 1)
