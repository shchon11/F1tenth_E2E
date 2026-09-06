"""Actuator models: servo command mapping and VESC speed loop (latency lives in sim.py)."""
from __future__ import annotations

from typing import Dict

import torch


def servo_target(steer_cmd: torch.Tensor, P: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Commanded steering angle -> servo target with calibration error and trim bias."""
    return (steer_cmd * P["steer_gain"] + P["steer_bias"]).clamp(-P["s_max"], P["s_max"])


def vesc_accel(speed_cmd: torch.Tensor, vx: torch.Tensor, P: Dict[str, torch.Tensor]) -> torch.Tensor:
    """VESC speed control loop emulated as first-order tracking; limits applied in dynamics."""
    v_tgt = (speed_cmd * P["speed_gain"]).clamp(P["v_min"], P["v_max"])
    return (v_tgt - vx) / P["motor_tau"]
