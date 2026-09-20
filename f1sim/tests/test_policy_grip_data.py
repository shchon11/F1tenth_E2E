"""Durability, causal input and episode/split identity for policy-domain collection."""
from types import SimpleNamespace

import pytest
import torch

from f1sim.learn.adaptive_grip import adaptive_feature_spec
from f1sim.learn.adaptive_grip_train import auroc
from f1sim.learn.policy_grip_data import (DatasetStore, WindowStream, calibrate_admitted,
    fit_dataset, load_split, observation_for_estimator, score_final, split_tracks,
    terminal_observation, policy_rows)


def obs(speed=0.2, batch=2):
    return {"speed": torch.full((batch, 1), speed), "imu": torch.zeros(batch, 6),
            "imu_att": torch.zeros(batch, 2)}


def test_terminal_old_observation_old_mu_and_reset_command_boundary():
    spec = adaptive_feature_spec()
    stream = WindowStream(2, "cpu", spec, 7)
    stream.begin(obs())
    issued = torch.tensor([[.1, 4.], [.2, 5.]])
    old_mu = torch.tensor([.4, 1.2])
    target = (old_mu, torch.tensor([True, True]), torch.tensor([False, False]),
              torch.tensor([.2, .3]), torch.ones(2))
    for _ in range(8):
        stream.append(obs(.3), issued, target, torch.zeros(2, dtype=torch.bool), obs())
    post_reset = obs(.9)
    info = {"final": {"ids": torch.tensor([0])}, "final_obs": obs(.4, batch=1)}
    terminal = terminal_observation(post_reset, info)
    assert terminal["speed"].flatten().tolist() == pytest.approx([.4, .9])
    rows = stream.append(terminal, issued, target, torch.tensor([True, False]), post_reset)
    old_mu[0] = 1.3  # simulator parameters mutate in place on reset; persisted target is a clone
    assert rows["mu"].tolist() == pytest.approx([.4, 1.2])
    assert rows["features"][0, 0, 0] == pytest.approx(.4)
    assert rows["features"][0, 0, 10] == pytest.approx(.4)
    assert stream.history.valid.sum(1).tolist() == [1, 10]
    assert stream.history.features[0, 0, 0] == pytest.approx(.9)
    assert stream.history.features[0, 0, 9:].tolist() == [0., 0.]
    assert stream.episode[0] == rows["episode"][0] + 2
    assert stream.step.tolist() == [0, 9]
    assert not stream.positive[0].any()


def test_observation_normalization_uses_only_measured_channels():
    x = obs(.5)
    x["imu"][:] = 2
    x["imu_att"][:] = .2
    x["mu"] = torch.full((2,), 100.)
    x["body_velocity"] = torch.full((2, 3), 999.)
    ecfg = SimpleNamespace(v_max_policy=20., imu_gyro_scale=10., imu_accel_scale=20.)
    result = observation_for_estimator(x, ecfg, adaptive_feature_spec())
    assert set(result) == {"speed", "imu", "imu_att"}
    assert torch.equal(result["speed"], torch.ones(2, 1))
    assert torch.equal(result["imu"], torch.full((2, 6), 4.))
    x["body_velocity"] *= 10
    assert all(torch.equal(v, observation_for_estimator(x, ecfg, adaptive_feature_spec())[k]) for k, v in result.items())


def test_split_is_train_only_and_base_map_disjoint():
    from f1sim.learn import common
    names = common.TRAIN_TRACKS
    split = split_tracks(names, [100, 200, 300, 400])
    bases = [{common.base_map(n) for n in group} for group in split.values()]
    assert all(bases)
    for i, group in enumerate(bases):
        assert all(not group.intersection(other) for other in bases[i + 1:])
    with pytest.raises(ValueError, match="TRAIN"):
        split_tracks([common.HELDOUT_TRACKS[0]] + names, [100, 200, 300, 400])
    with pytest.raises(ValueError, match="distinct"):
        split_tracks(names, [1, 1, 2, 3])


def test_resume_durable_chunks_detect_mismatch_and_orphan(tmp_path):
    spec = {"jobs": [{"id": 0, "split": "train"}], "source": "frozen"}
    store = DatasetStore(tmp_path, spec)
    data = {"mu": torch.tensor([.4, .5]), "episode": torch.tensor([1, 2])}
    store.write(0, 0, data)
    resumed = DatasetStore(tmp_path, spec)
    resumed.write(0, 0, data)
    with pytest.raises(ValueError, match="non-deterministic"):
        resumed.write(0, 0, {**data, "mu": torch.ones(2)})
    torch.save(data, tmp_path / "job00000_chunk00001.pt")
    resumed.write(0, 1, data)
    resumed.complete(0)
    assert len(load_split(tmp_path, "train")["mu"]) == 4
    with pytest.raises(ValueError, match="differs"):
        DatasetStore(tmp_path, {**spec, "source": "modified"})
    (tmp_path / "job00000_chunk00000.pt").write_bytes(b"bad")
    with pytest.raises(ValueError, match="digest"):
        load_split(tmp_path, "train")


def test_calibration_includes_false_positive_admissions():
    q = torch.tensor([[.8, 1., 1.2], [.3, .5, .7], [2., 2.2, 2.4]])
    data = {"mu": torch.tensor([.4, .5, .3]), "episode": torch.tensor([1, 2, 3]),
            "informative": torch.tensor([False, True, True])}
    cal = calibrate_admitted(q, torch.tensor([.9, .9, .1]), data)
    assert cal["cal_delta"] == pytest.approx(.4)
    assert cal["admitted_windows"] == 2
    with pytest.raises(ValueError, match="no admitted"):
        calibrate_admitted(q, torch.zeros(3), data)


def test_auroc_exact_ties_without_pairwise_matrix():
    score = torch.tensor([.1, .1, .4, .9, .5])
    label = torch.tensor([True, False, True, False, True])
    a, b = score[label], score[~label]
    expected = ((a[:, None] > b).float() + .5 * (a[:, None] == b).float()).mean()
    assert auroc(score, label) == pytest.approx(float(expected))


def test_tiny_fit_never_opens_final_and_final_is_single_use(tmp_path):
    torch.set_num_threads(1)
    spec = adaptive_feature_spec()
    root = tmp_path / "data"
    jobs = [{"id": i, "split": name} for i, name in enumerate(("train", "cal", "dev", "final"))]
    store = DatasetStore(root, {"feature_spec": spec.to_meta(), "jobs": jobs})
    n = 48
    for job in jobs:
        data = {"features": torch.zeros(n, 40, 11), "valid": torch.ones(n, 40, dtype=torch.bool),
                "mu": torch.full((n,), .7), "informative": torch.ones(n, dtype=torch.bool),
                "low_excitation": torch.zeros(n, dtype=torch.bool), "lower_bound": torch.full((n,), .4),
                "episode": torch.arange(n) + job["id"] * n,
                "job": torch.full((n,), job["id"], dtype=torch.long)}
        store.write(job["id"], 0, data)
        store.complete(job["id"])
    fit = tmp_path / "fit"
    report = fit_dataset(root, fit, seed=17, epochs=8, batch_size=16, hidden=4, learning_rate=.05)
    assert report["final_opened"] is False
    assert not (root / "final_opened.json").exists()
    assert (fit / "candidate.pt").exists()
    score_final(root, fit / "candidate.pt", tmp_path / "final.json")
    with pytest.raises(FileExistsError):
        score_final(root, fit / "candidate.pt", tmp_path / "another-final.json")


def test_calibration_population_excludes_scripted_excitation():
    manifest = {"specification": {"jobs": [
        {"id": 0, "mode": "policy_solo"}, {"id": 1, "mode": "policy_traffic"},
        {"id": 2, "mode": "scripted_solo"}]}}
    data = {"job": torch.tensor([0, 1, 2]), "mu": torch.tensor([.4, .8, 1.2])}
    assert policy_rows(data, manifest)["mu"].tolist() == pytest.approx([.4, .8])


def test_collector_pins_graph_and_observation_sources_without_extending_v2_pilot():
    from f1sim.learn.adaptive_grip_train import _source_hashes
    from f1sim.learn.policy_grip_data import collector_source_hashes
    old, current = _source_hashes(), collector_source_hashes()
    required = {"learn/policy_grip_data.py", "learn/common.py", "learn/obs.py",
                "learn/grip_runtime.py", "learn/grip_control.py", "mpc.py",
                "learn/graph_runtime.py", "learn/model.py", "teacher.py", "viewer/graph_fastpath.py"}
    assert required.issubset(current)
    assert "learn/policy_grip_data.py" not in old
    assert all(current[k] == v for k, v in old.items())


def _small_policy_checkpoint(path, missing_actor_key=False, saved_spec=True, **spec_overrides):
    from dataclasses import asdict
    from f1sim.learn.model import ActorCritic
    from f1sim.learn.obs import ObsSpec
    spec = ObsSpec(n_beams=32, scan_stack=1, act_dim=8, **spec_overrides)
    model = ActorCritic(1, 32, spec.proprio_dim, 3, act_dim=8)
    state = model.state_dict()
    if missing_actor_key:
        del state[next(key for key in state if key.startswith("actor."))]
    torch.save({"meta": model.meta, "state_dict": state,
                "extra": {"spec": asdict(spec)} if saved_spec else {}}, path)
    return asdict(spec)


def test_collector_rejects_missing_actor_tensor_and_missing_observation_spec(tmp_path):
    from f1sim.learn.policy_grip_data import load_collection_policy
    from f1sim.params import Config
    bad_actor, no_spec = tmp_path / "bad_actor.pt", tmp_path / "no_spec.pt"
    _small_policy_checkpoint(bad_actor, missing_actor_key=True)
    _small_policy_checkpoint(no_spec, saved_spec=False)
    with pytest.raises(ValueError, match="strict load failed"):
        load_collection_policy(bad_actor, "cpu", Config())
    with pytest.raises(ValueError, match="no observation specification"):
        load_collection_policy(no_spec, "cpu", Config())


def test_collector_restores_lidar_contract_and_rejects_attitude_scale(tmp_path):
    from f1sim.learn.policy_grip_data import load_collection_policy
    from f1sim.params import Config
    path = tmp_path / "policy.pt"
    _small_policy_checkpoint(path, range_max=17., gyro_scale=7., accel_scale=14.)
    cfg = Config()
    model, saved = load_collection_policy(path, "cpu", cfg)
    assert cfg.lidar.n_beams == 32 and cfg.lidar.range_max == 17.
    assert saved["gyro_scale"] == 7. and saved["accel_scale"] == 14.
    assert all(not p.requires_grad for p in model.parameters())
    _small_policy_checkpoint(path, att_scale=.5)
    with pytest.raises(ValueError, match="attitude scale"):
        load_collection_policy(path, "cpu", cfg)


def test_policy_normalizer_mismatch_is_rejected_before_collection(monkeypatch):
    from dataclasses import asdict
    from f1sim.learn import common
    from f1sim.learn.obs import ObsSpec
    expected = ObsSpec(n_beams=32, act_dim=8, range_max=17., v_max=12.)
    actual = ObsSpec(n_beams=32, act_dim=8, range_max=10., v_max=12.)
    monkeypatch.setattr(common, "obs_spec", lambda env: actual)
    with pytest.raises(ValueError, match="range_max"):
        common.validate_policy_observation(asdict(expected), object())
    actual.range_max = 17.
    actual.gyro_scale = 9.
    with pytest.raises(ValueError, match="gyro_scale"):
        common.validate_policy_observation(asdict(expected), object())


@pytest.mark.parametrize("bad_name", ["../outside.pt", "/tmp/outside.pt", "job0000_chunk00000.pt", "job000000_chunk00000.pt"])
def test_manifest_rejects_noncanonical_or_escaping_shards(tmp_path, bad_name):
    import json
    store = DatasetStore(tmp_path, {"jobs": [{"id": 0, "split": "train"}]})
    store.write(0, 0, {"mu": torch.tensor([.7])})
    store.complete(0)
    manifest = store.manifest
    manifest["shards"][bad_name] = manifest["shards"].pop("job00000_chunk00000.pt")
    store.path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="noncanonical"):
        load_split(tmp_path, "train")
    with pytest.raises(ValueError, match="noncanonical"):
        DatasetStore(tmp_path, manifest["specification"])


def test_manifest_schema_and_symlink_files_are_rejected(tmp_path):
    import json
    root = tmp_path / "data"
    specification = {"jobs": [{"id": 0, "split": "train"}]}
    store = DatasetStore(root, specification)
    store.write(0, 0, {"mu": torch.tensor([.7])})
    store.complete(0)
    shard = root / "job00000_chunk00000.pt"
    outside = tmp_path / "outside.pt"
    shard.replace(outside)
    shard.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        load_split(root, "train")
    with pytest.raises(ValueError, match="symlink"):
        store.write(0, 0, {"mu": torch.tensor([.7])})
    shard.unlink()
    outside.replace(shard)
    original = json.loads(store.path.read_text())
    for bad in ({**original, "format": "unknown"}, {**original, "completed_jobs": [999]},
                {**original, "specification": {"jobs": "bad"}}):
        store.path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_split(root, "train")
    external_manifest = tmp_path / "external.json"
    external_manifest.write_text(json.dumps(original))
    store.path.unlink()
    store.path.symlink_to(external_manifest)
    with pytest.raises(ValueError, match="symlink"):
        DatasetStore(root, specification)


def test_atomic_writers_ignore_predictable_temp_symlinks(tmp_path):
    store = DatasetStore(tmp_path / "data", {"jobs": [{"id": 0, "split": "train"}]})
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    (store.root / "manifest.json.tmp").symlink_to(victim)
    (store.root / "job00000_chunk00000.pt.tmp").symlink_to(victim)
    store.write(0, 0, {"mu": torch.tensor([.7])})
    store.complete(0)
    assert victim.read_text() == "untouched"
    assert load_split(store.root, "train")["mu"].item() == pytest.approx(.7)
    assert not list(store.root.glob(".*.tmp"))


def _asset_spec(tmp_path):
    from f1sim.learn.grip_estimator import file_sha256
    policy, estimator = tmp_path / "policy.pt", tmp_path / "estimator.pt"
    policy.write_bytes(b"policy bytes")
    estimator.write_bytes(b"estimator bytes")
    return {"jobs": [{"id": 0, "split": "train", "seed": 123, "cohort": "randomized"}],
            "device": "cpu", "policy": str(policy), "estimator": str(estimator),
            "policy_sha256": file_sha256(policy), "estimator_sha256": file_sha256(estimator)}


def test_asset_replacement_rejected_before_any_shard_commit(tmp_path):
    from pathlib import Path
    from f1sim.learn.policy_grip_data import verify_asset_pins
    specification = _asset_spec(tmp_path)
    store = DatasetStore(tmp_path / "data", specification)
    verify_asset_pins(specification)
    Path(specification["policy"]).write_bytes(b"replacement policy")
    with pytest.raises(ValueError, match="policy asset SHA changed"):
        store.write(0, 0, {"mu": torch.tensor([.7])})
    assert not list(store.root.glob("*.pt"))


def test_asset_replacement_during_policy_load_fails_before_environment(tmp_path, monkeypatch):
    from pathlib import Path
    from f1sim.learn import policy_grip_data as module
    specification = _asset_spec(tmp_path)
    store = DatasetStore(tmp_path / "data", specification)
    def replaced_while_loading(*args):
        Path(specification["estimator"]).write_bytes(b"replacement estimator")
        return object(), {}
    monkeypatch.setattr(module, "load_collection_policy", replaced_while_loading)
    with pytest.raises(ValueError, match="estimator asset SHA changed"):
        module.collect_job(store, specification["jobs"][0])
    assert not list(store.root.glob("*.pt"))


def _tiny_fit_store(root, rows=9):
    spec = adaptive_feature_spec()
    store = DatasetStore(root, {"feature_spec": spec.to_meta(),
                               "jobs": [{"id": i, "split": s} for i, s in enumerate(("train", "cal", "dev"))]})
    base = {"features": torch.zeros(rows, 40, 11), "valid": torch.ones(rows, 40, dtype=torch.bool),
            "mu": torch.linspace(.5, .8, rows), "informative": torch.ones(rows, dtype=torch.bool),
            "low_excitation": torch.zeros(rows, dtype=torch.bool), "lower_bound": torch.full((rows,), .3),
            "episode": torch.arange(rows)}
    for job in store.manifest["specification"]["jobs"]:
        store.write(job["id"], 0, {**base, "job": torch.full((rows,), job["id"], dtype=torch.long)})
        store.complete(job["id"])
    return base


def test_default_scratch_fit_preserves_original_numerical_training(tmp_path, monkeypatch):
    import numpy as np
    from f1sim.learn import policy_grip_data as module
    from f1sim.learn.adaptive_grip import AdaptiveGripNet
    from f1sim.learn.adaptive_grip_train import adaptive_loss
    train = _tiny_fit_store(tmp_path / "data")
    torch.manual_seed(1234)
    expected = AdaptiveGripNet(adaptive_feature_spec(), hidden=4)
    optimizer = torch.optim.AdamW(expected.parameters(), lr=.002)
    losses = []
    for _ in range(2):
        per_batch = []
        for ix in torch.randperm(len(train['mu'])).split(4):
            batch = {k: v[ix] for k, v in train.items()}
            q, logit = expected.distribution(batch['features'], batch['valid'])
            loss = adaptive_loss(q, logit, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), 2.)
            optimizer.step()
            per_batch.append(float(loss.detach()))
        losses.append(float(np.mean(per_batch)))
    monkeypatch.setattr(module, 'calibrate_admitted', lambda *args: {'cal_delta': 0.})
    result = module.fit_dataset(tmp_path/'data', tmp_path/'fit', 1234, 2, batch_size=4, hidden=4)
    ck = torch.load(tmp_path/'fit/candidate.pt', weights_only=True)
    assert result['training_loss'] == losses
    assert all(torch.equal(v, ck['state_dict'][k]) for k, v in expected.state_dict().items())
    assert 'fit_assets' not in ck['meta']['training']


def test_replay_seeded_mixture_matches_sealed_equal_half_recipe():
    from f1sim.learn.policy_grip_data import epoch_training_batches
    def data(n, offset):
        return {k: torch.arange(n) + offset for k in ('features', 'valid', 'mu', 'informative', 'lower_bound')}
    train, replay = data(9, 0), data(20, 100)
    torch.manual_seed(71)
    current = torch.randperm(9)
    current = torch.cat([current, torch.randint(9, (1,))])
    old = torch.randperm(20)[:10]
    expected = [torch.cat([ci, replay['mu'][ri]]) for ci, ri in zip(current.split(2), old.split(2))]
    torch.manual_seed(71)
    actual = list(epoch_training_batches(train, 4, replay, .5))
    assert len(actual) == 5
    assert all(torch.equal(batch['mu'], want) for batch, want in zip(actual, expected))
    torch.manual_seed(71)
    again = list(epoch_training_batches(train, 4, replay, .5))
    assert all(torch.equal(a['mu'], b['mu']) for a, b in zip(actual, again))


def test_warm_replay_pins_inputs_and_keeps_calibration_current_only(tmp_path, monkeypatch):
    from f1sim.learn import policy_grip_data as module
    from f1sim.learn.adaptive_grip import AdaptiveGripNet, save_adaptive_grip_estimator
    from f1sim.learn.grip_estimator import file_sha256
    base = _tiny_fit_store(tmp_path/'data')
    replay = {k: v.clone() for k, v in base.items()}
    replay['mu'][:] = 1.2
    torch.save(replay, tmp_path/'replay-train.pt')
    spec = adaptive_feature_spec()
    init = AdaptiveGripNet(spec, hidden=4)
    with torch.no_grad():
        init.head[-1].bias[-1] = 5.
    save_adaptive_grip_estimator(tmp_path/'init.pt', init, spec, {'cal_delta': .123}, {'deployment_approved': False})
    observed = []
    original = module.calibrate_admitted
    def cal(q, c, data, threshold):
        observed.append(data['mu'].clone())
        return original(q, c, data, threshold)
    monkeypatch.setattr(module, 'calibrate_admitted', cal)
    module.fit_dataset(tmp_path/'data', tmp_path/'fit', 17, 1, 4, 4, .0003,
                       init_estimator=tmp_path/'init.pt', replay_data=tmp_path/'replay-train.pt')
    ck = torch.load(tmp_path/'fit/candidate.pt', weights_only=True)
    declaration = ck['meta']['training']
    assert declaration['fit_assets']['init_estimator']['sha256'] == file_sha256(tmp_path/'init.pt')
    assert declaration['fit_assets']['replay_data']['sha256'] == file_sha256(tmp_path/'replay-train.pt')
    assert declaration['replay']['effective_fraction'] == .5
    assert len(observed) == 1 and torch.equal(observed[0], base['mu'])
    assert ck['meta']['deployment_approved'] is False
    assert not (tmp_path/'data/final_opened.json').exists()


@pytest.mark.parametrize('failure', ['features', 'valid', 'nonfinite'])
def test_replay_incompatible_features_mask_or_values_rejected(tmp_path, failure):
    from f1sim.learn.policy_grip_data import validate_training_tensors
    data = _tiny_fit_store(tmp_path/'data')
    if failure == 'features':
        data['features'] = data['features'][:, :, :-1]
    elif failure == 'valid':
        data['valid'][0, 0] = False
    else:
        data['features'][0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        validate_training_tensors(data, adaptive_feature_spec(), 'replay TRAIN')


def test_incompatible_init_and_changed_replay_rejected(tmp_path, monkeypatch):
    from f1sim.learn import policy_grip_data as module
    from f1sim.learn.adaptive_grip import AdaptiveGripNet, save_adaptive_grip_estimator
    data = _tiny_fit_store(tmp_path/'data')
    spec = adaptive_feature_spec()
    save_adaptive_grip_estimator(tmp_path/'init.pt', AdaptiveGripNet(spec, hidden=8), spec, {}, {})
    with pytest.raises(ValueError, match='architecture/hidden'):
        module.fit_dataset(tmp_path/'data', tmp_path/'wrong-hidden', 17, 1, hidden=4,
                           init_estimator=tmp_path/'init.pt')
    torch.save(data, tmp_path/'replay.pt')
    original = module.adaptive_loss
    def replaced_after_load(*args):
        (tmp_path/'replay.pt').write_bytes(b'replaced')
        return original(*args)
    monkeypatch.setattr(module, 'adaptive_loss', replaced_after_load)
    with pytest.raises(ValueError, match='replay_data fit asset changed'):
        module.fit_dataset(tmp_path/'data', tmp_path/'changed', 17, 1, 4, 4,
                           replay_data=tmp_path/'replay.pt')
    assert not (tmp_path/'changed/candidate.pt').exists()


def test_fit_cli_flags_and_replay_requirement(monkeypatch, tmp_path):
    from f1sim.learn import policy_grip_data as module
    calls = []
    monkeypatch.setattr(module, 'fit_dataset', lambda *args, **kwargs: calls.append((args, kwargs)))
    base = ['fit', '--data', str(tmp_path/'data'), '--out', str(tmp_path/'out'), '--seed', '17', '--epochs', '15']
    assert module.main(base + ['--init-estimator', str(tmp_path/'init.pt'), '--replay-data', str(tmp_path/'train.pt'), '--replay-fraction', '.5']) == 0
    assert calls[0][1]['init_estimator'] == tmp_path/'init.pt'
    assert calls[0][1]['replay_data'] == tmp_path/'train.pt'
    with pytest.raises(SystemExit):
        module.main(base + ['--replay-fraction', '.25'])
    with pytest.raises(SystemExit):
        module.main(base + ['--replay-data', str(tmp_path/'train.pt'), '--replay-fraction', '1'])
    assert len(calls) == 1


def test_warm_start_feature_spec_mismatch_rejected(tmp_path):
    from dataclasses import replace
    from f1sim.learn.policy_grip_data import fit_dataset
    from f1sim.learn.adaptive_grip import AdaptiveGripNet, save_adaptive_grip_estimator
    _tiny_fit_store(tmp_path/'data')
    incompatible = replace(adaptive_feature_spec(), warm_frames=9)
    save_adaptive_grip_estimator(tmp_path/'init.pt', AdaptiveGripNet(incompatible, hidden=4), incompatible, {}, {})
    with pytest.raises(ValueError, match='feature spec'):
        fit_dataset(tmp_path/'data', tmp_path/'fit', 17, 1, hidden=4, init_estimator=tmp_path/'init.pt')
    assert not (tmp_path/'fit/predeclared.json').exists()
