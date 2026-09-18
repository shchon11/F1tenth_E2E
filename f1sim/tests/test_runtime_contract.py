"""A policy resumes/evaluates with exactly its recorded automatic control pair."""
from copy import deepcopy

import pytest
import torch

from f1sim.learn.grip_runtime import (ControllerRuntime, validate_runtime_checkpoint,
                                     load_embedded_estimator, DEFAULT_AUTO_PROFILE,
                                     LOCAL_AUTO_PROFILE, AUTO_RUNTIME_VERSION, LOCAL_AUTO_RUNTIME_VERSION)
from f1sim.learn.grip_estimator import FeatureSpec, QuantileGripNet, save_grip_estimator
from test_automatic_grip import Env, sensors


def make_record(tmp_path, auto_profile=None):
    path = tmp_path / "observer.pt"
    torch.manual_seed(31)
    spec = FeatureSpec()
    save_grip_estimator(path, QuantileGripNet(spec, width=4, hidden=8), spec,
                        {"cal_delta": 0.}, {})
    rt = ControllerRuntime(Env(), "auto", str(path), auto_profile=auto_profile).install()
    try:
        record = rt.checkpoint_meta()
    finally:
        rt.release()
    return path, record, {"extra": {"experiment": {"controller": record}}}


def test_embedded_pair_survives_external_file_removal(tmp_path):
    path, record, ck = make_record(tmp_path)
    path.unlink()
    rt = ControllerRuntime(Env(), "auto", checkpoint_meta=record).install()
    try:
        assert validate_runtime_checkpoint(ck, rt) is record
        rt.begin(sensors(3))
        assert torch.isfinite(rt.pre_action(sensors(3))).all()
    finally:
        rt.release()


def test_modified_embedded_weight_is_refused(tmp_path):
    _, record, _ = make_record(tmp_path)
    record = deepcopy(record)
    next(iter(record["estimator_state"].values())).add_(1.)
    with pytest.raises(ValueError, match="content hash"):
        load_embedded_estimator(record)


def test_approval_metadata_is_bound_to_embedded_content(tmp_path):
    _, record, _ = make_record(tmp_path)
    record = deepcopy(record)
    record["estimator"]["meta"]["deployment_approved"] = True
    with pytest.raises(ValueError, match="content hash"):
        load_embedded_estimator(record)


def test_runtime_version_and_physical_spec_are_enforced(tmp_path):
    path, record, ck = make_record(tmp_path)
    wrong = deepcopy(ck)
    wrong["extra"]["experiment"]["controller"]["runtime_version"] = "different"
    with pytest.raises(ValueError, match="runtime version"):
        validate_runtime_checkpoint(wrong, "auto")
    rt = ControllerRuntime(Env(), "auto", str(path)).install()
    try:
        wrong = deepcopy(ck)
        wrong["extra"]["experiment"]["controller"]["grip_spec"]["a_brake"] = 1.
        with pytest.raises(ValueError, match="physical controller"):
            validate_runtime_checkpoint(wrong, rt)
    finally:
        rt.release()


def test_explicit_observer_cannot_silently_override_embedded_pair(tmp_path):
    path, record, _ = make_record(tmp_path)
    spec = FeatureSpec()
    torch.manual_seed(99)
    save_grip_estimator(path, QuantileGripNet(spec, width=4, hidden=8), spec,
                        {"cal_delta": 0.}, {})
    rt = ControllerRuntime(Env(), "auto", str(path), checkpoint_meta=record)
    with pytest.raises(ValueError, match="conflicts"):
        rt.install()


def test_default_auto_restores_historical_profile_without_observer_bypass(tmp_path):
    path, record, _ = make_record(tmp_path)
    rt = ControllerRuntime(Env(), "auto", str(path))
    assert rt.gspec.profile_version == DEFAULT_AUTO_PROFILE
    assert rt.gspec.drive_split_r == .5
    assert not rt.research_estimator
    assert record["runtime_version"] == AUTO_RUNTIME_VERSION
    assert record["auto_profile_source"] == "default"


def test_fresh_adaptive_selection_is_explicit_and_independent_of_estimator_permission(tmp_path):
    path, _, _ = make_record(tmp_path)
    rt = ControllerRuntime(Env(), "auto", str(path), auto_profile=LOCAL_AUTO_PROFILE).install()
    try:
        assert rt.gspec.profile_version == LOCAL_AUTO_PROFILE
        assert not rt.research_estimator
        assert rt.grip.tracker._command_hook is not None
        meta = rt.checkpoint_meta()
        assert meta["runtime_version"] == LOCAL_AUTO_RUNTIME_VERSION
        assert meta["auto_profile_source"] == "explicit"
    finally:
        rt.release()


@pytest.mark.parametrize("profile", [DEFAULT_AUTO_PROFILE, LOCAL_AUTO_PROFILE])
def test_research_profile_does_not_enable_unapproved_observer_loading(tmp_path, profile):
    from f1sim.learn.adaptive_grip import AdaptiveGripNet, save_adaptive_grip_estimator
    path = tmp_path / "unapproved.pt"
    net = AdaptiveGripNet(hidden=4)
    save_adaptive_grip_estimator(path, net, net.spec, {"cal_delta": 0.}, {"deployment_approved": False})
    rt = ControllerRuntime(Env(), "auto", str(path), research_profile=profile)
    assert rt.gspec.profile_version == profile
    assert not rt.research_estimator
    with pytest.raises(ValueError, match="deployment"):
        rt.install()


def test_saved_local_pair_reloads_its_bound_profile_even_after_default_changes(tmp_path):
    path, record, ck = make_record(tmp_path, LOCAL_AUTO_PROFILE)
    path.unlink()
    rt = ControllerRuntime(Env(), "auto", checkpoint_meta=record).install()
    try:
        assert rt.gspec.profile_version == LOCAL_AUTO_PROFILE
        assert rt.runtime_version == LOCAL_AUTO_RUNTIME_VERSION
        assert rt.auto_profile_source == "checkpoint"
        assert validate_runtime_checkpoint(ck, rt) is record
    finally:
        rt.release()


@pytest.mark.parametrize("saved,requested", [(LOCAL_AUTO_PROFILE, DEFAULT_AUTO_PROFILE),
                                             (DEFAULT_AUTO_PROFILE, LOCAL_AUTO_PROFILE)])
def test_explicit_profile_cannot_override_saved_pair(tmp_path, saved, requested):
    _, record, _ = make_record(tmp_path, saved)
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        ControllerRuntime(Env(), "auto", checkpoint_meta=record, auto_profile=requested)
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        ControllerRuntime(Env(), "auto", checkpoint_meta=record, research_profile=requested,
                          research_estimator=True)


def test_old_historical_research_version_remains_reproducible_without_relabeling(tmp_path):
    _, record, _ = make_record(tmp_path)
    record["runtime_version"] = LOCAL_AUTO_RUNTIME_VERSION
    rt = ControllerRuntime(Env(), "auto", checkpoint_meta=record).install()
    try:
        assert rt.gspec.profile_version == DEFAULT_AUTO_PROFILE
        assert rt.checkpoint_meta()["runtime_version"] == LOCAL_AUTO_RUNTIME_VERSION
    finally:
        rt.release()


def test_local_profile_cannot_use_historical_runtime_version(tmp_path):
    _, record, ck = make_record(tmp_path, LOCAL_AUTO_PROFILE)
    record["runtime_version"] = AUTO_RUNTIME_VERSION
    with pytest.raises(ValueError, match="runtime version"):
        validate_runtime_checkpoint(ck, "auto")
    with pytest.raises(ValueError, match="runtime version"):
        ControllerRuntime(Env(), "auto", checkpoint_meta=record)


def test_legacy_actor_can_enter_new_runtime_but_trained_actor_cannot_change_arm(tmp_path):
    assert validate_runtime_checkpoint({}, "auto") == {}
    _, _, ck = make_record(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        validate_runtime_checkpoint(ck, "legacy")
