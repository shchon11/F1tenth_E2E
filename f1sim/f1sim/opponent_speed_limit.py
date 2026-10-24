"""A speed limit for a car ahead, read off the v2 policy's own estimate of the nearest car (2026-10-25).

When another car is ahead of us -- in our lane now, or, at its current sideways speed relative to us, in our lane
within `horizon` seconds -- the speed command is limited to the speed from which we can still slow to that car's
speed before the gap between the bodies closes:

    limit = v_opp + sqrt(2 brake max(0, gap - gap_min - max(0, v_me - v_opp) lag))

with gap = longitudinal distance - car length, v_opp = v_me - closing speed. A car counts only while it is wholly ahead
(gap > 0.1 m, within `reach`), and not when we would already be a car length past it by the time it reaches our lane.
Everything is in our own frame (x forward, y left): computed along a raceline or the centreline the same rule did
nothing on the Korea track (docs/research/passing-risk-2026-10-24, section 6).

The other car comes from the v2 actor's nearest-car output (`learn/aux_v2.NearestCarHead`: [exists logit, dx / 5,
dy / 2, closing / 5, v_lat / 2] of the nearest car the LiDAR sees, relative to the LiDAR). That output does not feed the
action, so a checkpoint whose nearest-car output was retrained on its frozen features drives exactly as before
(`checkpoints/korea26_v2_ppo_s2421_u512_nearcar.pt`). Nothing here reads the simulator's truth: the same code runs on
the car (`f1sim_ros` controller node).

Measured on the Korea entry (finals race cells, fresh seeds 91-92): hard (>= 1.5 m/s) car contacts 1.22 -> 0.49 /km
with props on the line and 1.34 -> 0.54 with props off, all hard contacts 2.32 -> 1.63 and 2.12 -> 1.51, race speed
-1..-2 %, passes unchanged; in solo laps it never limited the speed. 30 % of its limits are ones the true state would
not set, and they precede most of the extra wall hits (+0.1-0.2 /km).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

#: the nearest-car output's columns and scales (`aux_v2.NearestCarHead`, `F1VecEnv.nearest_visible_car_v2`)
READING_DIM = 5
NO_CAR_LOGIT = -10.0


@dataclass(frozen=True)
class OpponentSpeedLimit:
    horizon: float = 1.0           # [s] how far ahead a sideways move into our lane is projected
    brake: float = 3.0             # [m/s^2] the deceleration the limit assumes
    lane_half_width: float = 0.35  # [m] |lateral offset| under which a car is in our lane (bodies are 0.3 m wide)
    lag: float = 0.2               # [s] drive lag spent at the current closing speed before braking bites
    gap_min: float = 0.2           # [m] gap left between the bodies when the speeds match
    car_length: float = 0.5        # [m]
    reach: float = 6.0             # [m] the farthest car considered


DEFAULT = OpponentSpeedLimit()


def no_car(batch: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(batch, 5) readings that see no car (the limit is off until a policy writes real ones)."""
    r = torch.zeros(batch, READING_DIM, device=device, dtype=dtype)
    r[:, 0] = NO_CAR_LOGIT
    return r


def relative(reading: torch.Tensor, mount_x, mount_y):
    """(exists, dx, dy, closing, v_lat) from the nearest-car output (..., 5): positions from the LiDAR moved to the car's
    reference point by the mount offset [m]."""
    r = reading.float()
    return r[..., 0] > 0, r[..., 1] * 5.0 + mount_x, r[..., 2] * 2.0 + mount_y, r[..., 3] * 5.0, r[..., 4] * 2.0


def speed_limit(reading: torch.Tensor, v_me: torch.Tensor, mount_x=0.297, mount_y=0.0,
                p: OpponentSpeedLimit = DEFAULT) -> torch.Tensor:
    """(...,) [m/s] the speed limit for the car `reading` describes, +inf where there is none (no car seen, not wholly
    ahead, beyond `reach`, not in our lane now nor within `horizon`, or passed by then). `v_me`: our forward speed."""
    ex, dx, dy, closing, v_lat = relative(reading, mount_x, mount_y)
    gap = dx - p.car_length
    v_opp = (v_me.float() - closing).clamp_min(0.0)
    room = (gap - p.gap_min - closing.clamp_min(0.0) * p.lag).clamp_min(0.0)
    lim = v_opp + torch.sqrt(2.0 * p.brake * room)
    in_lane = dy.abs() < p.lane_half_width
    t_in = torch.where(in_lane, torch.zeros_like(dy),
                       torch.where((dy * v_lat) < 0, (dy.abs() - p.lane_half_width) / v_lat.abs().clamp_min(1e-6),
                                   torch.full_like(dy, float("inf"))))
    passed = (dx - closing * t_in.clamp(max=p.horizon)) < -(p.car_length + 0.1)
    applies = ex & (gap > 0.1) & (dx < p.reach) & (t_in <= p.horizon) & ~passed
    return torch.where(applies, lim, torch.full_like(lim, float("inf")))
