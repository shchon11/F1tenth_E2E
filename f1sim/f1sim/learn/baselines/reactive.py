"""The reactive classics as baselines: ForzaETH's Follow-the-Gap and the Disparity Extender.

Neither needs a map, a line or a localisation, so neither carries a privileged input: each reads the
car's own 270 deg scan (and FTG the odometry speed) and commands a steering angle and a speed, the
same contract as TinyLidarNet and End2Race (`BaselineDriver`). The user asked for the published
stacks as baselines (2026-10-17); these two are the classics every F1TENTH benchmark carries.

**Follow-the-Gap** is the classic reactive method of the RoboRacer (F1TENTH) course. The implementation here
is the one in ForzaETH's stack (`ForzaETH/race_stack` @ e7d07f7f4ffe0d4c30b90340b0f49a76af2877c0,
`controller/ftg/ftg.py`, MIT), ported line for line with the ROS plumbing and the RViz markers
removed, and with the stack's competition parameters (`stack_master/config/state_machine_params.yaml`:
safety_radius 40 beams, max_lidar_dist 9 m, range_offset 180, track_width 2.6 m, max_speed 6.0) as
defaults. In that stack it is the fallback controller; its top speed (`max_speed * 0.6`) is tuned
for that role, so `max_speed` is the one knob a fair comparison sweeps, as it does every baseline's
pace. The vendored copy lives in `external/baselines/forzaeth_ftg/` (read-only, never staged).

**Disparity Extender** (UNC, CPSWeek 2019 winner) has no official code; this follows the authors'
description (nathanotterness.com, 2019-04) step by step. The post gives no numbers, so the ones below
are this car's (`VehicleParams`: width 0.31 m, steering stop 0.4189 rad) plus declared choices:
disparity threshold 0.2 m (the post: "0.1 or 0.2"), 0.1 m tolerance beyond half the width, the
target among bearings within +-90 deg, a side check of half the width + 0.1 m, and a speed that rises
in two linear pieces from `v_min` at 0.5 m of free road ahead to `0.5 top_speed` at 2 m and `top_speed`
at 6 m. `top_speed` is the knob that is swept (named apart from the adapter's `v_max`, the action encoding).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .common import BaselineDriver, BaselineError, ScanContract

#: This car's scanner, which both algorithms read as-is (ROS order: index 0 is -135 deg, right).
N_BEAMS, FOV, RANGE_MAX, RATE = 1081, 1.5 * math.pi, 10.0, 40.0


class _Params:
    """The `backend` a `BaselineDriver` reports: no network, just the algorithm and its numbers."""

    def __init__(self, name: str, params: dict, source: str):
        self.name, self.params, self.source = name, dict(params), source

    def describe(self) -> dict:
        return {"backend": "numpy", "algorithm": self.name, "params": self.params, "source": self.source}


# --------------------------------------------------------------------------------- Follow-the-Gap
class FollowTheGap(BaselineDriver):
    KIND = "ftg"
    needs_speed = True          # the bubble radius grows with the car's speed (`set_vel`)

    PREPROCESS_CONV_SIZE = 3
    SCALE = 0.6                 # ".575 is max", as upstream
    STRAIGHTS_STEERING_ANGLE = np.pi / 18
    MILD_CURVE_ANGLE = np.pi / 6
    ULTRASTRAIGHTS_ANGLE = np.pi / 60

    def __init__(self, *, max_speed: float = 6.0, safety_radius: int = 40, max_lidar_dist: float = 9.0,
                 range_offset: int = 180, track_width: float = 2.6):
        p = dict(max_speed=float(max_speed), safety_radius=int(safety_radius),
                 max_lidar_dist=float(max_lidar_dist), range_offset=int(range_offset),
                 track_width=float(track_width))
        super().__init__(ScanContract(n_beams=N_BEAMS, fov=FOV, range_max=RANGE_MAX, control_rate=RATE),
                         _Params("follow_the_gap", p, "ForzaETH/race_stack@e7d07f7 controller/ftg/ftg.py (MIT)"),
                         note="ForzaETH Follow-the-Gap, line-for-line port; max_speed is the tuned knob")
        self.MAX_SPEED = p["max_speed"]
        self.SAFETY_RADIUS = p["safety_radius"]
        self.MAX_LIDAR_DIST = p["max_lidar_dist"]
        self.range_offset = p["range_offset"]
        self.track_width = p["track_width"]
        s = self.MAX_SPEED * self.SCALE
        self.CORNERS_SPEED, self.MILD_CORNERS_SPEED = 0.3 * s, 0.45 * s
        self.STRAIGHTS_SPEED, self.ULTRASTRAIGHTS_SPEED = 0.8 * s, s
        self.radians_per_elem = None

    # -- upstream, one row ---------------------------------------------------------------------
    def _preprocess_lidar(self, ranges: np.ndarray) -> np.ndarray:
        self.radians_per_elem = (1.5 * np.pi) / len(ranges)
        proc = np.array(ranges[self.range_offset:-self.range_offset])
        proc = np.convolve(proc, np.ones(self.PREPROCESS_CONV_SIZE) / self.PREPROCESS_CONV_SIZE, "valid")
        proc = np.clip(proc, 0, self.MAX_LIDAR_DIST)
        return proc[::-1]

    def _safety_border(self, ranges: np.ndarray) -> np.ndarray:
        filtered = list(ranges)
        n = len(ranges)
        i = 0
        while i < n - 1:
            if ranges[i + 1] - ranges[i] > 0.5:
                for j in range(self.SAFETY_RADIUS):
                    if i + j < n:
                        filtered[i + j] = ranges[i]
                i += self.SAFETY_RADIUS - 2
            i += 1
        i = n - 1
        while i > 0:
            if ranges[i - 1] - ranges[i] > 0.5:
                for j in range(self.SAFETY_RADIUS):
                    if i - j >= 0:
                        filtered[i - j] = ranges[i]
                i = i - self.SAFETY_RADIUS + 2
            i -= 1
        return np.array(filtered)

    def _find_largest_gap(self, ranges: np.ndarray, radius: float):
        bin_ranges = np.where(ranges >= radius, 1, 0)
        bin_diffs = np.abs(np.diff(bin_ranges))
        bin_diffs[0] = 1
        bin_diffs[-1] = 1
        diff_idxs = bin_diffs.nonzero()[0]
        high_gaps = []
        for i in range(len(diff_idxs) - 1):
            low, high = diff_idxs[i], diff_idxs[i + 1]
            high_gaps.append(np.mean(bin_ranges[low:high]) > 0.5)
        widths = np.array(high_gaps) * np.diff(diff_idxs)
        gap_left = diff_idxs[np.argmax(widths)]
        gap_width = np.max(widths)
        return gap_left, gap_left + gap_width

    def _get_radius(self, velocity: float) -> float:
        return min(5., self.track_width / 2 + 2 * (velocity / self.MAX_SPEED))

    def _row(self, ranges: np.ndarray, velocity: float):
        proc = self._safety_border(self._preprocess_lidar(ranges))
        radius = self._get_radius(velocity)
        gap_left, gap_right = self._find_largest_gap(proc, radius)
        gap_left += self.range_offset - 180
        gap_right += self.range_offset - 180
        gap_middle = int((gap_right + gap_left) / 2)
        best_y = np.cos(gap_middle * self.radians_per_elem) * radius
        best_x = np.sin(gap_middle * self.radians_per_elem) * radius
        steering_angle = float(np.clip(math.atan2(best_y, best_x), -0.4, 0.4))
        a = abs(steering_angle)
        if a > self.MILD_CURVE_ANGLE:
            speed = self.CORNERS_SPEED
        elif a > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.MILD_CORNERS_SPEED
        elif a > self.ULTRASTRAIGHTS_ANGLE:
            speed = self.STRAIGHTS_SPEED
        else:
            speed = self.ULTRASTRAIGHTS_SPEED
        return steering_angle, speed

    def _forward(self, ranges_m: np.ndarray, speed_mps: Optional[np.ndarray]) -> np.ndarray:
        out = np.empty((ranges_m.shape[0], 2), dtype=np.float32)
        for b in range(ranges_m.shape[0]):
            out[b] = self._row(ranges_m[b].astype(np.float64), float(speed_mps[b]))
        return out


# ------------------------------------------------------------------------------ Disparity Extender
class DisparityExtender(BaselineDriver):
    KIND = "disparity"
    needs_speed = False

    def __init__(self, *, top_speed: float = 6.0, v_min: float = 0.5, car_width: float = 0.31,
                 tolerance: float = 0.1, disparity_threshold: float = 0.2, side_margin: float = 0.1,
                 max_steer: float = 0.4189, d_near: float = 0.5, d_mid: float = 2.0, d_far: float = 6.0,
                 forward_halfwidth_deg: float = 5.0):
        p = dict(top_speed=float(top_speed), v_min=float(v_min), car_width=float(car_width), tolerance=float(tolerance),
                 disparity_threshold=float(disparity_threshold), side_margin=float(side_margin),
                 max_steer=float(max_steer), d_near=float(d_near), d_mid=float(d_mid), d_far=float(d_far),
                 forward_halfwidth_deg=float(forward_halfwidth_deg))
        if not (p["d_near"] < p["d_mid"] < p["d_far"]):
            raise BaselineError(f"disparity: need d_near < d_mid < d_far, got {p}")
        super().__init__(ScanContract(n_beams=N_BEAMS, fov=FOV, range_max=RANGE_MAX, control_rate=RATE),
                         _Params("disparity_extender", p, "N. Otterness, 'The Disparity Extender', 2019-04 (spec; no code)"),
                         note="Disparity Extender from the authors' description; numbers are this car's, top_speed is the tuned knob")
        self.p = p
        self.angles = np.linspace(-FOV / 2, FOV / 2, N_BEAMS)
        self.inc = FOV / (N_BEAMS - 1)

    def _extend(self, r: np.ndarray) -> np.ndarray:
        """Steps 2-3: at every disparity, overwrite the farther side with the closer distance for as
        many samples as half the car's width plus the tolerance spans at that distance, never
        overwriting a sample that is already closer."""
        out = r.copy()
        half = self.p["car_width"] / 2 + self.p["tolerance"]
        d = np.diff(r)
        for i in np.nonzero(np.abs(d) > self.p["disparity_threshold"])[0]:
            near_i, far_dir = (i, +1) if r[i] < r[i + 1] else (i + 1, -1)
            near = r[near_i]
            n = int(math.ceil(half / max(near * self.inc, 1e-6)))
            j0 = near_i + far_dir
            j1 = j0 + far_dir * n
            sl = slice(j0, min(j1, len(r))) if far_dir > 0 else slice(max(j1 + 1, 0), j0 + 1)
            out[sl] = np.minimum(out[sl], near)
        return out

    def _row(self, r: np.ndarray):
        p = self.p
        f = self._extend(r)
        front = np.abs(self.angles) <= math.pi / 2
        idx = np.nonzero(front)[0]
        best = f[idx].max()                                          # step 4: the farthest bearing ahead;
        tied = idx[f[idx] >= best - 1e-6]                            # a saturated scan ties many beams, and
        k = tied[len(tied) // 2]                                     # the middle of them is the open road's
        target = float(self.angles[k])
        steer = float(np.clip(target, -p["max_steer"], p["max_steer"]))   # step 5: as sharp as allowed
        # step 6: a wall beside the car on the side it wants to turn into -> go straight
        side = (self.angles > math.pi / 2) if steer > 0 else (self.angles < -math.pi / 2)
        if steer != 0.0 and np.any(r[side] * np.abs(np.sin(self.angles[side])) < p["car_width"] / 2 + p["side_margin"]):
            steer = 0.0
        # step 7: speed from the free road straight ahead, two linear pieces
        fw = np.abs(self.angles) <= math.radians(p["forward_halfwidth_deg"])
        dist = float(np.min(f[fw]))
        if dist <= p["d_near"]:
            v = p["v_min"]
        elif dist <= p["d_mid"]:
            v = p["v_min"] + (0.5 * p["top_speed"] - p["v_min"]) * (dist - p["d_near"]) / (p["d_mid"] - p["d_near"])
        elif dist <= p["d_far"]:
            v = 0.5 * p["top_speed"] + 0.5 * p["top_speed"] * (dist - p["d_mid"]) / (p["d_far"] - p["d_mid"])
        else:
            v = p["top_speed"]
        return steer, v

    def _forward(self, ranges_m: np.ndarray, speed_mps: Optional[np.ndarray]) -> np.ndarray:
        out = np.empty((ranges_m.shape[0], 2), dtype=np.float32)
        for b in range(ranges_m.shape[0]):
            out[b] = self._row(ranges_m[b].astype(np.float64))
        return out


def load_ftg(weights: str = "", **opts) -> FollowTheGap:
    """`weights` names no file (the algorithm has none); options are FollowTheGap's keywords."""
    return FollowTheGap(**{k: v for k, v in opts.items() if k in (
        "max_speed", "safety_radius", "max_lidar_dist", "range_offset", "track_width")})


def load_disparity(weights: str = "", **opts) -> DisparityExtender:
    return DisparityExtender(**{k: v for k, v in opts.items() if k in (
        "top_speed", "v_min", "car_width", "tolerance", "disparity_threshold", "side_margin", "max_steer",
        "d_near", "d_mid", "d_far", "forward_halfwidth_deg")})
