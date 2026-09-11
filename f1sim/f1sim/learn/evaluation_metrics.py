"""First-attempt trial accounting, independent of simulator autoresets.

Primary metric is `collisions_per_km`: a hazard rate per metre driven, which is comparable
across tracks and across evaluation budgets. `completion_rate` is a *derived* quantity --
P(complete) ~ exp(-lambda * L) -- so it mixes policy quality with track length and with the
time budget, and must never be compared across tracks of different lengths on its own.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """95 % Wilson score interval for a binomial rate (sane at 0 and n successes)."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def poisson_rate_interval(count: int, exposure: float, scale: float = 1.0) -> tuple[float, float] | None:
    """Exact 95 % interval for a Poisson rate `count / exposure * scale` (chi-square method)."""
    if exposure <= 0:
        return None
    from scipy.stats import chi2
    lo = 0.0 if count == 0 else float(chi2.ppf(0.025, 2 * count) / 2)
    hi = float(chi2.ppf(0.975, 2 * (count + 1)) / 2)
    return (lo / exposure * scale, hi / exposure * scale)


class TrialMetrics(TypedDict):
    initial_trials: int
    completions: int
    collisions: int
    timeouts: int
    completion_rate: float | None
    completion_rate_ci95: tuple[float, float] | None
    collision_rate: float | None
    timeout_rate: float | None
    active_vehicle_seconds: float
    distance_m: float
    progress_m: float
    collisions_per_km: float | None
    collisions_per_km_ci95: tuple[float, float] | None
    hazard_predicted_completion_rate: float | None
    budget_laps_available: float | None
    budget_feasible: bool | None
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

    def __init__(self, lengths: NDArray[np.float64], step_dt: float, *,
                 time_budget_s: float | None = None, speed_cap: float | None = None):
        """time_budget_s / speed_cap: recorded so the report can say whether a trial could have
        finished at all. A budget shorter than length / speed_cap makes every timeout structural,
        and the resulting completion rate carries no information about the policy."""
        self.lengths = lengths.copy()
        self.step_dt = step_dt
        self.time_budget_s = time_budget_s
        self.speed_cap = speed_cap
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
        hazard = collisions * 1000 / distance if distance else None
        median_length = float(np.median(self.lengths)) if n else None
        laps_available = None
        if self.time_budget_s and self.speed_cap and median_length:
            laps_available = self.time_budget_s * self.speed_cap / median_length
        return {
            'initial_trials': n, 'completions': completions, 'collisions': collisions, 'timeouts': timeouts,
            'completion_rate': completions / n if n else None,
            'completion_rate_ci95': wilson_interval(completions, n),
            'collision_rate': collisions / n if n else None, 'timeout_rate': timeouts / n if n else None,
            'active_vehicle_seconds': exposure, 'distance_m': distance, 'progress_m': progress,
            'collisions_per_km': hazard,
            'collisions_per_km_ci95': poisson_rate_interval(collisions, distance, 1000.0),
            # exp(-lambda L): what the completion rate should be if crashes were a uniform hazard.
            # Close agreement means completion rate adds nothing beyond `collisions_per_km` and length.
            'hazard_predicted_completion_rate': (float(np.exp(-hazard / 1000 * median_length))
                                                 if hazard is not None and median_length else None),
            'budget_laps_available': laps_available,
            'budget_feasible': None if laps_available is None else bool(laps_available >= 1.05),
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
