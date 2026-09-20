"""Checkpoint-normalizer and exact-actor seams used outside training."""
import time
from types import SimpleNamespace

import pytest
import torch

from f1sim.learn import common, evaluate
from f1sim.learn.obs import ObsSpec


def test_evaluator_restores_checkpoint_normalizers_before_build(tmp_path, monkeypatch):
    spec = vars(ObsSpec(n_beams=541, scan_stack=6, action_history=2, act_dim=8,
                        v_max=8., gyro_scale=3., accel_scale=12., range_max=12.)).copy()
    path = tmp_path / "policy.pt"
    torch.save({"meta": {}, "state_dict": {}, "extra": {"spec": spec}}, path)
    model = SimpleNamespace(meta={"act_dim": 8}, eval=lambda: None)
    monkeypatch.setattr(evaluate, "load_checkpoint", lambda *a, **k: (model, {"spec": spec}))
    monkeypatch.setattr(common, "load_tracks", lambda *a, **k: ([object()], None))

    class Built(Exception):
        pass

    def make_env(tracks, envs, device, ecfg, *, cfg, **kw):
        assert ecfg.v_max_policy == 8.
        assert ecfg.imu_gyro_scale == 3.
        assert ecfg.imu_accel_scale == 12.
        assert ecfg.scan_stack == 6 and ecfg.action_history == 2
        assert cfg.lidar.range_max == 12. and cfg.lidar.n_beams == 541
        raise Built

    monkeypatch.setattr(common, "make_env", make_env)
    with pytest.raises(Built):
        evaluate.evaluate(str(path), ["unused"], 1, 2, 8., "cpu")


def test_observation_contract_rejects_scalar_drift_and_nonfinite(monkeypatch):
    expected = ObsSpec(v_max=8.)
    monkeypatch.setattr(common, "obs_spec", lambda env: ObsSpec(v_max=10.))
    with pytest.raises(ValueError, match="v_max"):
        common.validate_policy_observation(vars(expected), object())
    monkeypatch.setattr(common, "obs_spec", lambda env: ObsSpec(v_max=float("nan")))
    with pytest.raises(ValueError, match="v_max"):
        common.validate_policy_observation(vars(expected), object())


def test_auto_hot_reload_calls_exact_actor_guard(tmp_path, monkeypatch):
    from f1sim.viewer.sim_worker import SimWorker
    from f1sim.learn import grip_runtime, model as models, policy_adaptation
    path = tmp_path / "updated.pt"
    torch.save({"state_dict": {}, "extra": {}}, path)
    old_model = object()
    session = {"last_reload": time.time() - 10, "mtime": 0., "ckpt_path": str(path),
               "device": torch.device("cpu"), "controller": object(), "model": old_model}
    monkeypatch.setattr(models, "load_checkpoint", lambda *a, **k: (object(), {}))
    monkeypatch.setattr(grip_runtime, "validate_runtime_checkpoint", lambda *a, **k: {"arm": "auto"})
    calls = []

    def exact_actor(model, checkpoint):
        calls.append(checkpoint)
        raise RuntimeError("missing actor key: refusing reload")

    monkeypatch.setattr(policy_adaptation, "require_exact_actor", exact_actor)
    worker = SimWorker.__new__(SimWorker)
    worker.gen = 1
    messages = []
    worker.say = lambda kind, **kw: messages.append(kw)
    worker._reload_if_changed(session)
    assert calls and session["model"] is old_model
    assert any("missing actor key" in str(message) for message in messages)
