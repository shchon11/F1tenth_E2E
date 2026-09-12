"""Actuator models: servo command mapping and VESC speed loop (latency lives in sim.py)."""
from __future__ import annotations

from typing import Dict

import torch


def servo_target(steer_cmd: torch.Tensor, P: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Commanded steering angle -> servo target with calibration error and trim bias."""
    return (steer_cmd * P["steer_gain"] + P["steer_bias"]).clamp(-P["s_max"], P["s_max"])


def vesc_accel(speed_cmd: torch.Tensor, v_fb: torch.Tensor, P: Dict[str, torch.Tensor]) -> torch.Tensor:
    """VESC speed control loop emulated as first-order tracking; limits applied in dynamics.

    `v_fb` is the speed the loop closes on. On the car that is ERPM -- the *wheel* surface speed
    `omega_r * r_w` -- which is what `sim.py` passes with `vehicle.wheel_model` on; with it off it
    passes the body speed `vx`, as this did before 2026-09-13.

    Which one it is decides whether a wheel can slip at all. A spinning wheel reads fast, so the
    loop backs off; a locked one reads slow, so it keeps driving -- and that second half is visible
    in the recordings, where the motor current is *positive* through the hardest brake locks
    (20260826-173704 t=46.5: wheel -143 m/s^2 at +53 A). Closed on the body speed instead, the loop
    fights every slip back to zero and the wheel state has nothing to do.
    """
    v_tgt = (speed_cmd * P["speed_gain"]).clamp(P["v_min"], P["v_max"])
    return (v_tgt - v_fb) / P["motor_tau"]
