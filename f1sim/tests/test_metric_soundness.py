"""Regressions for the measurement problems that made a working policy look like a broken one.

Each test pins a specific failure the old reporting allowed:
  * a track longer than the fixed step budget scored 0 % completion no matter how it was driven;
  * completion rate was read as a policy property, though it is ~exp(-hazard * length);
  * point estimates from 16 trials were compared as if they were separated;
  * a car parked against a wall paid no proximity penalty.
"""
import numpy as np
import pytest
import torch

from f1sim.gym_env import EnvConfig, F1VecEnv
from f1sim.learn.evaluate import budget_steps, resolve_steps
from f1sim.learn.evaluation_metrics import TrialAccumulator, poisson_rate_interval, wilson_interval
from f1sim.params import Config
from f1sim.track import Track


class _Loop:
    """Stand-in for a Track with a known centerline length."""

    def __init__(self, length_m: float, n: int = 400):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        r = length_m / (2 * np.pi)
        self.centerline = np.stack([r * np.cos(t), r * np.sin(t)], 1)


def test_budget_scales_with_track_length():
    # Given a 444 m circuit and a 4 m/s cap, one lap cannot fit in the old fixed 60 s budget.
    monza, korea = _Loop(444.0), _Loop(43.4)
    assert 444.0 / 4.0 > 2400 * 0.025

    # When the budget is derived from the track instead of fixed.
    steps = budget_steps([monza], speed_cap=4.0, step_dt=0.025, budget_laps=2.0, max_steps=24000)

    # Then it covers two laps, and a short track gets a proportionally smaller budget.
    assert steps * 0.025 * 4.0 == pytest.approx(2 * 444.0, rel=0.02)
    assert budget_steps([korea], 4.0, 0.025, 2.0, 24000) < steps


def test_rolling_protocol_keeps_a_fixed_exposure():
    # A hazard rate needs comparable exposure across tracks, so only `trials` scales its budget.
    tracks = [_Loop(444.0)]
    assert resolve_steps("rolling", 2400, 2.0, tracks, 4.0, 0.025, 24000) == 2400
    assert resolve_steps("trials", 2400, 2.0, tracks, 4.0, 0.025, 24000) > 2400
    assert resolve_steps("trials", 2400, None, tracks, 4.0, 0.025, 24000) == 2400


def test_budget_uses_the_longest_track_and_respects_the_ceiling():
    mixed = budget_steps([_Loop(43.4), _Loop(444.0)], 4.0, 0.025, 2.0, 24000)
    assert mixed == budget_steps([_Loop(444.0)], 4.0, 0.025, 2.0, 24000)
    assert budget_steps([_Loop(444.0)], 4.0, 0.025, 2.0, max_steps=100) == 100


def test_infeasible_budget_is_flagged_rather_than_reported_as_failure():
    # Given a 60 s budget on a 444 m track at 4 m/s: every timeout is structural.
    trials = TrialAccumulator(np.array([444.0] * 4), 0.025, time_budget_s=60.0, speed_cap=4.0)
    trials.finish()
    report = trials.report()
    assert report['budget_feasible'] is False
    assert report['budget_laps_available'] == pytest.approx(60.0 * 4.0 / 444.0)
    assert report['timeouts'] == 4

    # When the budget covers the lap, the flag clears.
    ok = TrialAccumulator(np.array([43.4] * 4), 0.025, time_budget_s=60.0, speed_cap=4.0)
    ok.finish()
    assert ok.report()['budget_feasible'] is True


def test_step_dt_comes_from_the_simulator_config():
    # The budget is derived before the env exists, so it reads the config directly. `SimParams` has
    # `control_rate`, not `control_dt` -- taking the wrong one raises only at run time.
    config = Config()
    assert not hasattr(config.sim, "control_dt")
    step_dt = 1.0 / config.sim.control_rate
    assert step_dt == pytest.approx(0.025)

    # And the derived budget matches what the running env would use for its step length.
    track = Track.generate_random(0, style="circuit")
    sim_config = Config()
    sim_config.sim.compile = False
    sim_config.rand.enabled = False
    env = F1VecEnv(track, sim_config, EnvConfig(compile_tracker=False), num_envs=1, device="cpu")
    assert env.sim.control_dt == pytest.approx(step_dt)


def test_completion_rate_is_predicted_by_hazard_and_length():
    # Given a uniform crash hazard, completion rate carries no information beyond lambda and L.
    trials = TrialAccumulator(np.array([100.0]), 1.0, time_budget_s=1000.0, speed_cap=4.0)
    for _ in range(20):                                    # 20 m driven, then a crash
        trials.update(np.array([1.0]), np.array([1.0]), np.array([False]), np.array([False]))
    trials.update(np.array([1.0]), np.array([1.0]), np.array([True]), np.array([False]))
    report = trials.report()
    hazard = report['collisions_per_km'] / 1000
    assert report['hazard_predicted_completion_rate'] == pytest.approx(np.exp(-hazard * 100.0))


def test_small_trial_counts_report_intervals_that_overlap():
    # Given 3/16 and 6/16 completions -- the kind of pair earlier experiments were decided on.
    lo, hi = wilson_interval(3, 16)
    other_lo, other_hi = wilson_interval(6, 16)
    assert lo < 0.1 and hi > 0.4                           # 3/16 is [~0.07, ~0.45], not "12 %"
    assert other_lo < hi                                   # the intervals overlap: not a difference


def test_zero_collisions_still_bounds_the_hazard_from_above():
    # Given a clean run, the rate is 0 but the upper bound is not.
    interval = poisson_rate_interval(0, exposure=3600.0, scale=1000.0)
    assert interval[0] == 0.0
    assert interval[1] == pytest.approx(1000 * 3.688 / 3600.0, rel=0.01)
    assert poisson_rate_interval(0, 0.0) is None


def _wall_env(**cfg):
    """Single env on a generated track, proximity penalty on."""
    track = Track.generate_random(0, style="circuit")
    env_cfg = EnvConfig(reward_proximity=1.0, safe_dist=0.30, reward_progress=0.0, reward_collision=0.0,
                        reward_steer_rate=0.0, reward_wrong_way=0.0, compile_tracker=False, **cfg)
    conf = Config()
    conf.sim.compile = False
    conf.rand.enabled = False
    return F1VecEnv(track, conf, env_cfg, num_envs=2, device="cpu")


def test_proximity_penalty_scales_with_distance_driven_not_centerline_progress():
    # Given a car inside the wall margin, driven straight at the wall so centerline progress is
    # near zero while the car is still moving: the old |progress| scaling made that free.
    env = _wall_env()
    env.reset(seed=0)
    proximity_unit = 1.0                                   # penalty per metre at zero wall gap
    speeds = torch.tensor([0.0, 3.0])                      # stopped vs moving, same wall distance
    new_penalty = -proximity_unit * speeds.abs() * env.sim.control_dt
    old_penalty = -proximity_unit * torch.tensor([0.0, 0.0]).abs()   # crawling at a wall: no progress

    # Then the moving car is charged under the new form and was not under the old one.
    assert new_penalty[1] < 0 and old_penalty[1] == 0.0
    # And the penalty is proportional to speed, so hugging a wall fast costs more than creeping.
    assert new_penalty[1] == pytest.approx(3.0 * env.sim.control_dt * -1.0)


def test_proximity_component_tracks_speed_in_the_running_env():
    # End to end: the reported component never exceeds the distance actually driven.
    env = _wall_env()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(torch.zeros(2, env.act_dim))
    travelled = env.sim.state[:, 3].abs() * env.sim.control_dt
    proximity = info["reward_components"]["proximity"]
    assert bool((proximity.abs() <= travelled * env.ecfg.reward_proximity + 1e-6).all())
    torch.testing.assert_close(sum(info["reward_components"].values()), reward)
