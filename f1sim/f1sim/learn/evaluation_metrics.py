"""First-attempt trial accounting, independent of simulator autoresets."""
from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class TrialMetrics(TypedDict):
    initial_trials: int
    completions: int
    collisions: int
    timeouts: int
    completion_rate: float | None
    collision_rate: float | None
    timeout_rate: float | None
    active_vehicle_seconds: float
    distance_m: float
    progress_m: float
    collisions_per_km: float | None
    progress_rate_mps: float | None
    mean_speed: float | None
    completion_times_s: list[float]
    completion_time_mean_s: float | None
    completion_time_median_s: float | None
    completion_time_p90_s: float | None
    lap_time_s: float | None
    spin_events: int
    wrong_way_fraction: float | None
    high_yaw_rate_fraction: float | None
    large_slip_fraction: float | None
    max_abs_yaw_rate: float | None


class TrialAccumulator:
    """Mutable counters for exactly one attempt per initial learner."""

    def __init__(self, lengths: NDArray[np.float64], step_dt: float):
        self.lengths = lengths.copy()
        self.step_dt = step_dt
        self.active = np.ones(lengths.shape, dtype=bool)
        self.progress = np.zeros(lengths.shape)
        self.elapsed = np.zeros(lengths.shape)
        self.distance = np.zeros(lengths.shape)
        self.completed = np.zeros(lengths.shape, dtype=bool)
        self.collided = np.zeros(lengths.shape, dtype=bool)
        self.timed_out = np.zeros(lengths.shape, dtype=bool)
        self.was_wrong_way = np.zeros(lengths.shape, dtype=bool)
        self.spin_events = 0
        self.wrong_way_seconds = 0.0
        self.high_yaw_rate_seconds = 0.0
        self.large_slip_seconds = 0.0
        self.max_abs_yaw_rate = 0.0

    def update(self, progress: NDArray[np.float64], speed: NDArray[np.float64],
               collision: NDArray[np.bool_], truncated: NDArray[np.bool_], *,
               yaw_rate: NDArray[np.float64] | None = None, heading_error: NDArray[np.float64] | None = None,
               longitudinal_speed: NDArray[np.float64] | None = None, lateral_speed: NDArray[np.float64] | None = None) -> None:
        """Consume one transition; terminal flags and exposure share the active mask.

        Progress/speed are transition measurements; collision/truncation are its
        competing outcomes. Keeping them together prevents reset-state accounting.
        """
        active = self.active.copy()
        self.progress[active] += progress[active]
        self.elapsed[active] += self.step_dt
        self.distance[active] += np.abs(speed[active]) * self.step_dt
        if yaw_rate is not None and heading_error is not None and longitudinal_speed is not None and lateral_speed is not None:
            wrong_way = np.cos(heading_error) < 0
            high_yaw_rate = np.abs(yaw_rate) > 5.0
            slip = np.abs(np.arctan2(lateral_speed, np.abs(longitudinal_speed).clip(min=1e-6))) > np.deg2rad(20)
            moving = speed > 1.0
            self.spin_events += int((active & wrong_way & ~self.was_wrong_way).sum())
            self.wrong_way_seconds += float((active & wrong_way).sum()) * self.step_dt
            self.high_yaw_rate_seconds += float((active & high_yaw_rate).sum()) * self.step_dt
            self.large_slip_seconds += float((active & moving & slip).sum()) * self.step_dt
            self.max_abs_yaw_rate = max(self.max_abs_yaw_rate, float(np.abs(yaw_rate[active]).max(initial=0.0)))
            self.was_wrong_way[active] = wrong_way[active]
        crash = self.active & collision
        success = self.active & ~crash & (self.progress >= self.lengths)
        timeout = self.active & ~crash & ~success & truncated
        self.collided |= crash
        self.completed |= success
        self.timed_out |= timeout
        self.active &= ~(crash | success | timeout)

    def finish(self) -> None:
        """The requested time budget ends every still-active first attempt."""
        self.timed_out |= self.active
        self.active[:] = False

    def report(self) -> TrialMetrics:
        """Return strict-JSON compatible rates and successful completion times."""
        n = int(self.lengths.size)
        completions, collisions, timeouts = (int(x.sum()) for x in (self.completed, self.collided, self.timed_out))
        exposure, distance, progress = (float(x.sum()) for x in (self.elapsed, self.distance, self.progress))
        times = self.elapsed[self.completed]
        mean = float(times.mean()) if times.size else None
        return {
            'initial_trials': n, 'completions': completions, 'collisions': collisions, 'timeouts': timeouts,
            'completion_rate': completions / n if n else None,
            'collision_rate': collisions / n if n else None, 'timeout_rate': timeouts / n if n else None,
            'active_vehicle_seconds': exposure, 'distance_m': distance, 'progress_m': progress,
            'collisions_per_km': collisions * 1000 / distance if distance else None,
            'progress_rate_mps': progress / exposure if exposure else None,
            'mean_speed': distance / exposure if exposure else None,
            'completion_times_s': times.tolist(), 'completion_time_mean_s': mean,
            'completion_time_median_s': float(np.median(times)) if times.size else None,
            'completion_time_p90_s': float(np.percentile(times, 90)) if times.size else None,
            'lap_time_s': mean,
            'spin_events': self.spin_events,
            'wrong_way_fraction': self.wrong_way_seconds / exposure if exposure else None,
            'high_yaw_rate_fraction': self.high_yaw_rate_seconds / exposure if exposure else None,
            'large_slip_fraction': self.large_slip_seconds / exposure if exposure else None,
            'max_abs_yaw_rate': self.max_abs_yaw_rate if exposure else None,
        }
