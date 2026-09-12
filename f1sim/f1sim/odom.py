"""VESC-style odometry (vesc_to_odom). The IMU model lives in imu.py."""
from __future__ import annotations

from typing import Dict

import torch

from .dynamics import wrap_angle


class VescOdom:
    """Dead-reckoned pose from measured speed (ERPM) and *commanded* steering angle,
    exactly like the f1tenth vesc_to_odom node. State (B, 5): x, y, yaw, v, yaw_rate.

    The speed it reports is the **wheel** speed, not the ground speed: `vesc_to_odom` divides an
    ERPM count by a calibration gain and publishes the result as `twist.twist.linear.x`. With
    `vehicle.wheel_model` on, `sim.py` feeds it `omega_r * r_w` and the two differ whenever the
    rear axle slips -- which is the whole reason `f1sim_ros/traction.py` can work at all. With the
    switch off it is fed the body speed, as it was before 2026-09-13.

    Two artefacts of that channel are modelled alongside it (`stamp`, `erpm_quantum`), because the
    guard's entire input distribution is made of them; both are measured in `params.OdomParams`.
    """

    def __init__(self, num_envs: int, device):
        self.state = torch.zeros(num_envs, 5, device=device)

    def reset(self, env_ids: torch.Tensor, pose: torch.Tensor):
        self.state[env_ids, :3] = pose
        self.state[env_ids, 3:] = 0.0

    def update(self, v_wheel: torch.Tensor, steer_cmd: torch.Tensor, P: Dict[str, torch.Tensor],
               dt: float, quantise: bool = False):
        self.state = self.update_pure(self.state, v_wheel, steer_cmd, P, dt, quantise)
        return self.state

    @staticmethod
    def update_pure(state: torch.Tensor, v_wheel: torch.Tensor, steer_cmd: torch.Tensor,
                    P: Dict[str, torch.Tensor], dt: float, quantise: bool = False):
        """One dead-reckoning step as a pure function of the previous odometry state (compilable).

        `quantise` puts the reported speed on the ERPM lattice. It is applied *before* the pose is
        integrated, because the real node integrates the same quantised number it publishes.
        """
        v_meas = v_wheel * (1.0 + P["speed_scale_err"]) + torch.randn_like(v_wheel) * P["speed_noise_std"]
        if quantise:
            q = P["erpm_quantum"]
            v_meas = torch.round(v_meas / q) * q
        # the operator calibrates steering_angle_to_servo_offset/_gain against the physical servo, so
        # odom knows steer_bias and steer_gain up to residuals (steer_offset, steer_gain_err)
        steer_est = (steer_cmd * P["steer_gain"] + P["steer_bias"]) * (1.0 + P["steer_gain_err"]) + P["steer_offset"]
        L = P["lf"] + P["lr"]
        yaw_rate = v_meas * torch.tan(steer_est) / L + torch.randn_like(v_wheel) * P["yaw_rate_noise_std"]
        yaw = wrap_angle(state[:, 2] + yaw_rate * dt)
        return torch.stack([state[:, 0] + v_meas * torch.cos(yaw) * dt, state[:, 1] + v_meas * torch.sin(yaw) * dt, yaw, v_meas, yaw_rate], 1)

    @staticmethod
    def stamp(t: float, prev: torch.Tensor, P: Dict[str, torch.Tensor], period: float) -> torch.Tensor:
        """When this step's /odom sample claims to have been taken, per env [s].

        The *value* is a sample of the wheel speed one control period after the last one; the
        *timestamp* is when the message was published, and on the car those two are not the same
        clock. 7.2 % of the recordings' 88 975 /odom steps are shorter than 15 ms and 0.12 % are
        shorter than 5 ms, down to 0.057 ms -- and a speed difference worth a whole period divided
        by a 0.3 ms step is how `np.gradient` reads 110 m/s^2 of wheel acceleration that the car
        never saw (real-car REPORT.md section 5). That artefact is the reason `TractionParams`
        carries `min_diff_dt` at all, so a simulator that published a clean 40 Hz grid would train
        a detector with no reason to have it.

        Two components, both measured (`OdomParams`), and each matches a different part of the
        recordings' step distribution:

        * a Gaussian publish offset, sd 2.4 ms. Its *difference* between consecutive samples is the
          step's jitter (sd 3.39 ms), which reproduces the measured p5 / p95 of 14.375 / 25.570 ms
          on the car's 20 ms period.
        * a **catch-up publish**: with probability `stamp_jitter_burst` a sample goes out
          immediately after the previous one instead of on the grid. This is the mechanism, not a
          fitted tail -- a message that was held then released lands microseconds after its
          predecessor while carrying a whole period of new wheel speed, which is exactly the
          `20260827-111616` t=5.541240 / 5.541531 pair (0.291 ms apart, dv = -0.032 m/s) that reads
          as -110 m/s^2. A Gaussian offset bounded by half a period cannot produce that at all, and
          without it a simulator at 40 Hz has no sub-5 ms steps whatsoever.

        `prev` is the previous step's stamp per env. The sequence stays strictly increasing: the
        Gaussian branch is confined to half a period either side and the catch-up branch is
        measured forward from `prev`.
        """
        half = 0.5 * period
        g = (t + torch.randn_like(prev) * P["stamp_jitter_std"]).clamp(t - half, t + half)
        # 0.05-3 ms after the previous sample: the span the recordings' back-to-back pairs occupy
        catch = prev + 5e-5 + torch.rand_like(prev) * 2.95e-3
        burst = (torch.rand_like(prev) < P["stamp_jitter_burst"]) & (catch < g)
        return torch.where(burst, catch, g)
