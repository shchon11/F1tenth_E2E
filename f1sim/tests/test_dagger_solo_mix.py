"""Empty-track DAgger is a separate simulator and counts learner labels, not car rows."""
import json
import sys

import pytest
import torch

from f1sim import tracks
from f1sim.learn import common, dagger


def test_step_budget_counts_learner_samples():
    assert dagger.collection_steps(100, .3) == (30, 70)
    assert dagger.collection_steps(7, .3) == (2, 5)
    assert dagger.collection_steps(7, 0) == (0, 7)
    for steps, fraction in [(0, .3), (1, .3), (10, 1), (10, -.1), (10, float('nan'))]:
        with pytest.raises(ValueError):
            dagger.collection_steps(steps, fraction)


def test_bare_maps_remove_assets_keep_direction_and_deduplicate():
    names = dagger.bare_track_names(['real/korea26@rev#line:4', 'real/korea26@rev#hard:8'])
    assert len(names) == 1
    sc = tracks.parse(names[0])
    assert sc.reverse and sc.bare and not sc.obstacle and not sc.asset


def test_real_mixed_collection_and_recurrent_training(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'RUNS_DIR', str(tmp_path))
    real_collect = dagger.collect
    observed = []

    def capture(env, model, teacher, steps, beta, device, buf, **kw):
        out = real_collect(env, model, teacher, steps, beta, device, buf, **kw)
        observed.append((env.M, env.B, steps, teacher.__class__.__name__,
                         tuple(t.props for t in env.sim.tracks), out))
        return out

    monkeypatch.setattr(dagger, 'collect', capture)
    monkeypatch.setattr(sys, 'argv', ['dagger', '--name', 'mixed', '--tracks', 'gen:control:1400',
        '--envs', '6', '--race-size', '3', '--opponent', 'teacher', '--action-mode', 'plan',
        '--teacher', 'interactive', '--teacher-offsets', '0', '--teacher-speeds', '1',
        '--teacher-cand-iters', '0', '--teacher-horizon', '0.1',
        '--solo-fraction', '0.3', '--steps', '10', '--iters', '2', '--keep-iters', '1', '--epochs', '1',
        '--batch', '8', '--scan-stack', '2', '--hist-len', '2', '--memory', 'gru',
        '--memory-hidden', '16', '--chunk-length', '2', '--eval-steps', '2',
        '--device', 'cpu', '--eager', '--wandb', 'disabled'])
    dagger.main()
    assert [(x[0], x[1], x[2]) for x in observed] == [(3, 6, 7), (1, 2, 3)] * 2
    assert observed[0][3] == 'InteractiveTeacher'
    assert observed[1][3] == 'RacelineTeacher'
    assert not any(observed[1][4])
    traffic, solo = observed[2][-1], observed[3][-1]
    assert (traffic.B, solo.B, len(traffic), len(solo)) == (2, 2, 14, 6)
    for buf in (traffic, solo):
        assert buf.N[0].all()  # independent simulator/chunk reset, never a cross-cohort sequence
        assert torch.isfinite(buf.L).all()
        _, _, labels, keep = buf.sample_chunks(4, 2, 'cpu')
        assert labels.shape[:2] == keep.shape == (2, 4)
    saved = torch.load(tmp_path / 'mixed/student_latest.pt', weights_only=True)
    extra = saved['extra']
    assert extra['solo_samples'] == 6 and extra['traffic_samples'] == 14
    assert extra['collection_mix']['solo_fraction'] == .3
    assert extra['samples'] == 20
    row = json.loads((tmp_path / 'mixed/progress.jsonl').read_text().splitlines()[-1])
    assert row['solo_samples'] == 6 and row['traffic_samples'] == 14


@pytest.mark.parametrize('flags,reason', [
    (['--solo-tracks', 'gen:control:1400'], 'positive --solo-fraction'),
    (['--solo-fraction', '.3'], 'race-size >= 2'),
    (['--solo-fraction', '.3', '--race-size', '3', '--teacher', 'interactive',
      '--opp-token', 'future'], 'opp-token off'),
])
def test_incompatible_cohorts_rejected_before_loading_maps(monkeypatch, flags, reason):
    monkeypatch.setattr(sys, 'argv', ['dagger', *flags])
    with pytest.raises(SystemExit, match=reason):
        dagger.main()
