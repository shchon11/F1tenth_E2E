"""First-attempt accounting regression scenarios."""
import json

import numpy as np
import pytest

from f1sim.learn.evaluation_metrics import TrialAccumulator


def test_collision_masks_later_success_and_retry_exposure():
    # Given two initial trials, the first collides immediately.
    trials = TrialAccumulator(np.array([10., 10.]), 0.5)
    # When later autoreset attempts complete a lap.
    trials.update(np.array([1., 2.]), np.array([3., 4.]), np.array([True, False]), np.array([False, False]))
    trials.update(np.array([100., 8.]), np.array([100., 4.]), np.array([False, False]), np.array([False, False]))
    # Then only the second first attempt completes; retry exposure is ignored.
    result = trials.report()
    assert (result['initial_trials'], result['completions'], result['collisions'], result['timeouts']) == (2, 1, 1, 0)
    assert result['active_vehicle_seconds'] == 1.5
    assert result['distance_m'] == 5.5
    assert result['collisions_per_km'] == pytest.approx(1000 / 5.5)
    assert result['collision_rate'] == result['completion_rate'] == 0.5
    assert result['completion_times_s'] == [1.0]


@pytest.mark.parametrize('collision,expected', [(False, (1, 0, 0)), (True, (0, 1, 0))])
def test_collision_then_success_then_timeout_precedence(collision, expected):
    # Given a full lap ending exactly at timeout.
    trials = TrialAccumulator(np.array([10.]), 0.5)
    # When all terminal conditions coincide.
    trials.update(np.array([10.]), np.array([20.]), np.array([collision]), np.array([True]))
    # Then collision wins over success, which wins over timeout.
    result = trials.report()
    assert (result['completions'], result['collisions'], result['timeouts']) == expected


def test_short_finish_line_crossing_is_not_full_lap():
    # Given a random spawn one metre before the finish line.
    trials = TrialAccumulator(np.array([100.]), 0.5)
    # When the car crosses it and the budget ends.
    trials.update(np.array([1.]), np.array([2.]), np.array([False]), np.array([False]))
    trials.finish()
    # Then the initial attempt timed out and undefined statistics are JSON null.
    result = trials.report()
    assert result['completions'] == 0
    assert result['timeouts'] == 1
    assert result['completion_time_mean_s'] is None
    assert 'NaN' not in json.dumps(result, allow_nan=False)


def test_signed_progress_and_truncation_mask_late_retries():
    # Given a car that goes forward and then backward before truncation.
    trials = TrialAccumulator(np.array([10.]), 0.5)
    # When it truncates before net lap distance, then completes on a retry.
    trials.update(np.array([9.]), np.array([18.]), np.array([False]), np.array([False]))
    trials.update(np.array([-8.]), np.array([16.]), np.array([False]), np.array([True]))
    trials.update(np.array([20.]), np.array([40.]), np.array([False]), np.array([False]))
    # Then the retry changes no first-trial metrics.
    result = trials.report()
    assert result['timeouts'] == 1
    assert result['progress_m'] == 1
    assert result['distance_m'] == 17


def test_instability_metrics_count_wrong_way_entries_and_lateral_slip() -> None:
    trials = TrialAccumulator(np.array([100.]), 0.5)
    trials.update(np.array([1.]), np.array([2.]), np.array([False]), np.array([False]),
                  yaw_rate=np.array([2.]), heading_error=np.array([0.]), longitudinal_speed=np.array([2.]), lateral_speed=np.array([0.]))
    trials.update(np.array([1.]), np.array([2.]), np.array([False]), np.array([False]),
                  yaw_rate=np.array([6.]), heading_error=np.array([2.]), longitudinal_speed=np.array([1.]), lateral_speed=np.array([1.]))
    trials.update(np.array([1.]), np.array([2.]), np.array([False]), np.array([True]),
                  yaw_rate=np.array([6.]), heading_error=np.array([2.]), longitudinal_speed=np.array([1.]), lateral_speed=np.array([1.]))
    result = trials.report()
    assert result['spin_events'] == 1
    assert result['wrong_way_fraction'] == 2 / 3
    assert result['high_yaw_rate_fraction'] == 2 / 3
    assert result['large_slip_fraction'] == 2 / 3
    assert result['max_abs_yaw_rate'] == 6


def test_real_teacher_report_has_resolved_configuration():
    # Given a tiny eager CPU evaluation on a cached procedural track.
    from f1sim.learn.evaluate import evaluate
    from f1sim.params import Config
    cfg = Config()
    cfg.sim.compile = False
    cfg.lidar.n_beams = 64
    # When two first attempts exhaust a two-step budget.
    result = evaluate('', ['gen:competition:0'], 2, 2, 4., 'cpu', cfg=cfg,
                      teacher=True, protocol='trials', seed=17)
    # Then counts and reproducibility settings survive strict JSON serialization.
    decoded = json.loads(json.dumps(result, allow_nan=False))
    assert sum(decoded[key] for key in ('completions', 'collisions', 'timeouts')) == 2
    assert decoded['metadata']['env_config']['max_steps'] == 2
    assert decoded['metadata']['seeds']['reset'] == 17
    assert decoded['metadata']['config']['sim']['compile'] is False
    assert len(decoded['initial_track_lengths_m']) == 2


def test_per_track_sweeps_always_write_complete_report(monkeypatch, tmp_path):
    # Given a CLI report run whose numerical evaluator is separately exercised above.
    import sys
    from f1sim.learn import evaluate as module
    def evaluation(ckpt, names, envs, steps, cap, device, **kwargs):
        return {'initial_trials': envs, 'tracks': names, 'seed': kwargs['seed']}
    output = tmp_path / 'report.json'
    monkeypatch.setattr(module, 'evaluate', evaluation)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--teacher', '--tracks', 'a,b', '--per-track',
                                   '--sweep', '--seed', '27', '--output', str(output)])
    # When both nominal tracks and all robustness sweeps are requested.
    module.main()
    # Then all 18 results are machine readable, including seeds and explicit counts.
    report = json.loads(output.read_text())
    assert len(report) == 18
    assert {'a', 'b', 'mu=0.7/a', 'lidar_z=0.18/b'} <= report.keys()
    assert all(result['seed'] == 27 and result['initial_trials'] == 256 for result in report.values())


def test_rolling_report_does_not_claim_first_attempt_distance_accounting():
    from f1sim.learn.evaluate import evaluate
    from f1sim.params import Config
    cfg = Config()
    cfg.sim.compile = False
    cfg.lidar.n_beams = 64
    result = evaluate('', ['gen:competition:0'], 2, 2, 4., 'cpu', cfg=cfg,
                      teacher=True, protocol='rolling', seed=17)
    assert result['metadata']['trial_success'] is None
    assert 'distance not reported' in result['metadata']['distance_convention']
