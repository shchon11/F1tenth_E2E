"""Automatic friction runtime: causal inputs, nominal drivetrain, and historical-arm parity."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from f1sim import mpc
from f1sim.params import Config
from f1sim.gym_env import EnvConfig
from f1sim.learn import grip_control as gc
from f1sim.learn import grip_runtime as gr
from f1sim.learn.grip_estimator import FeatureSpec, QuantileGripNet, save_grip_estimator


class NoPlantTruth:
    @property
    def P(self):
        raise AssertionError("automatic runtime must not read simulator truth")


class Env:
    def __init__(self, batch=3):
        self.B, self.device, self.cfg = batch, torch.device("cpu"), Config()
        self.ecfg = EnvConfig()
        self.sim = NoPlantTruth()
        self.tracker = SimpleNamespace(spec=mpc.PlanSpec(), wb=0.3302, s_max=0.4189,
                                       v_max=10.0, _solver=None)
        self.last_cmd = torch.zeros(batch, 2)

    def _reset_envs(self, ids):
        self.last_cmd[ids] = 0


@pytest.fixture
def estimator_path(tmp_path):
    path = tmp_path / "estimator.pt"
    spec = FeatureSpec()
    torch.manual_seed(31)
    save_grip_estimator(path, QuantileGripNet(spec, width=4, hidden=8), spec,
                        {"cal_delta": 0.0}, {"test": True})
    return str(path)


def sensors(batch):
    return {"speed": torch.full((batch, 1), 0.3), "imu": torch.zeros(batch, 6),
            "imu_att": torch.zeros(batch, 2), "priv": torch.full((batch, 21), float("nan")),
            "mu": torch.full((batch,), float("nan"))}


def test_auto_is_one_estimated_path_and_requires_estimator():
    assert gr.split_arm("auto") == gr.ArmParts("estimated", False, False)
    assert gr.arm_to_grip_mode("auto") == "estimated"
    for spelling in ("auto+clearance", "auto+tcs", "oracle+auto"):
        with pytest.raises(ValueError, match="arm must be one of"):
            gr.split_arm(spelling)
    with pytest.raises(ValueError, match="needs --estimator"):
        gr.ControllerRuntime(Env(), "auto")
    with pytest.raises(ValueError, match="nominal vehicle"):
        gr.ControllerRuntime(Env(), "auto", "unused", gspec=gc.GripSpec(mode="estimated"))


def test_adaptive_evidence_hold_is_not_a_fallback_or_old_fixed_floor():
    rt = gr.ControllerRuntime(Env(batch=2), "auto", "unused")
    rt.estimator = SimpleNamespace(meta={"format": "adaptive_grip_v2"},
                                   consumption=SimpleNamespace(physical_floor=.05))
    diag = {"warm": torch.ones(2, dtype=torch.bool), "fault": torch.zeros(2, dtype=torch.bool),
            "fallback_reason": torch.full((2,), 3), "q_gap": torch.zeros(2),
            "excitation_pass": torch.zeros(2, dtype=torch.bool)}
    rt._record(torch.full((2,), gc.MU_FIXED_LOW), diag)
    sums = rt.acc.read()
    assert sums["fallback"] == sums["warm_fallback"] == sums["at_floor"] == 0
    diag["fault"][0] = True
    rt._record(torch.tensor([.05, .8]), diag)
    sums = rt.acc.read()
    assert sums["fallback"] == sums["at_floor"] == 1


def test_auto_uses_nominal_4wd_split_and_actuator_budgets():
    env = Env()
    spec = gr.automatic_grip_spec(env)
    assert spec.profile_version == gr.DEFAULT_AUTO_PROFILE == "historical-global-v1"
    assert gr.automatic_grip_spec(env, profile_version=gr.LOCAL_AUTO_PROFILE).profile_version == "local-v2"
    assert spec.drive_split_r == 0.5
    _, acc, brk = gc.budgets(torch.tensor([gc.MU_FIXED_LOW]), torch.zeros(1), spec)
    assert acc.item() == pytest.approx(7.0)
    assert brk.item() == pytest.approx(5.0)
    env.cfg.vehicle.a_brake = 4.2
    _, _, brk = gc.budgets(torch.tensor([gc.MU_FIXED_LOW]), torch.zeros(1),
                           gr.automatic_grip_spec(env))
    assert brk.item() == pytest.approx(4.2)
    env.cfg.vehicle.wheel_model = False
    assert gr.automatic_grip_spec(env).drive_split_r == 1.0


def test_historical_specs_keep_bit_identical_rear_only_budgets():
    for mode in gc.MODES:
        spec = gc.GripSpec(mode=mode)
        old = asdict(spec)
        old.pop("drive_split_r")
        assert gc.GripSpec(**old).drive_split_r == 1.0
        mu = torch.tensor([0.73423, 0.94401, 1.15379])
        rho = torch.tensor([0.0, 0.4, 0.85])
        q = mu * spec.mu_r_scale * torch.sqrt((1 - rho.clamp(0, 1) ** 2).clamp_min(0))
        acc_old = (q * gc.G * spec.lf / (spec.L - q * spec.h).clamp_min(1e-3)).clamp(max=spec.a_max)
        brk_old = (q * gc.G * spec.lf / (spec.L + q * spec.h)).clamp(max=spec.a_brake)
        _, acc, brk = gc.budgets(mu, rho, spec)
        assert torch.equal(acc, acc_old)
        assert torch.equal(brk, brk_old)


@pytest.mark.parametrize("share", [0.0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_drive_split_is_rejected(share):
    with pytest.raises(ValueError, match="drive_split_r"):
        gc.GripSpec(drive_split_r=share).validate()


def test_batched_auto_loads_frozen_estimator_and_resets_each_car(estimator_path):
    env = Env(batch=3)
    rt = gr.ControllerRuntime(env, "auto", estimator_path).install()
    obs = sensors(env.B)
    done = torch.zeros(env.B, dtype=torch.bool)
    try:
        rt.begin(obs)
        first = rt.pre_action(obs)
        assert first.shape == (env.B,)
        assert torch.equal(first, torch.full((env.B,), rt.estimator.spec.mu_min))
        assert not rt.last_estimate["warm"].any()
        assert torch.equal(rt.last_estimate["used_mu"], first)
        rt.post_step(done, done)
        for _ in range(40):
            used = rt.pre_action(obs)
            rt.post_step(done, done)
        assert rt.history.is_warm().all()
        assert torch.equal(used, rt.grip.mu)
        assert not any(p.requires_grad for p in rt.net.parameters())
        assert torch.isfinite(used).all()
        rt.post_step(torch.tensor([False, True, False]), done)
        used = rt.pre_action(obs)
        assert rt.history.valid_count().tolist() == [40, 1, 40]
        assert used[1].item() == pytest.approx(rt.estimator.spec.mu_min)
        assert rt.last_estimate["warm"].tolist() == [True, False, True]
        meta = rt.checkpoint_meta()
        assert meta["arm"] == "auto"
        assert meta["runtime_version"] == gr.AUTO_RUNTIME_VERSION
        assert meta["grip_spec"]["drive_split_r"] == 0.5
        assert meta["estimator"]["sha256"] == rt.estimator.sha
        assert meta["estimator_state"]
        with pytest.raises(RuntimeError, match="must not read P"):
            rt._truth()
        metrics, _ = rt.collect_metrics()
        assert "controller/warm_frac" in metrics
        assert "controller/fallback_frac" in metrics
        assert "controller/coverage_q10_q90" not in metrics
    finally:
        rt.release()
    assert env.tracker._solver is None


def test_auto_rescales_policy_sensors_and_normalizes_commands_only_once(estimator_path):
    env = Env(batch=2)
    env.ecfg.v_max_policy = 20.0
    env.ecfg.imu_gyro_scale = 2.5
    env.ecfg.imu_accel_scale = 20.0
    rt = gr.ControllerRuntime(env, "auto", estimator_path).install()
    obs = sensors(env.B)
    obs["speed"].fill_(0.2)                       # 4 m/s, estimator expects 0.4
    obs["imu"][:] = torch.tensor([0.4, 0.8, 1.2, 0.1, 0.2, 0.3])
    obs["imu_att"][:] = torch.tensor([0.1, -0.2])
    before = {key: value.clone() for key, value in obs.items()}
    done = torch.zeros(env.B, dtype=torch.bool)
    try:
        rt.begin(obs)
        rt.pre_action(obs)
        env.last_cmd[:] = torch.tensor([0.20945, 5.0])
        rt.post_step(done, done)
        rt.pre_action(obs)
        expected = torch.tensor([0.4, 0.2, 0.4, 0.6, 0.2, 0.4, 0.6, 0.1, -0.2, 0.5, 0.5])
        assert torch.allclose(rt.history.features[:, 0], expected.expand(env.B, -1))
        for key in ("speed", "imu", "imu_att"):
            assert torch.equal(obs[key], before[key]), "the actor's observation was mutated"
        adapter = rt.checkpoint_meta()["observation_adapter"]
        assert adapter["sensor_multipliers"] == [2.0, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 1.0, 1.0]
        assert adapter["control_dt"] == 0.025
        obs["imu"] = torch.zeros(env.B, 1)
        with pytest.raises(ValueError, match="exactly"):
            rt.pre_action(obs)
    finally:
        rt.release()


def test_historical_estimated_keeps_its_pass_through_normalization(estimator_path):
    env = Env(batch=2)
    env.ecfg.v_max_policy = 20.0
    rt = gr.ControllerRuntime(env, "estimated", estimator_path).install()
    try:
        obs = sensors(env.B)
        rt.begin(obs)
        rt.pre_action(obs)
        assert torch.equal(rt.history.features[:, 0, 0], obs["speed"][:, 0])
        assert "observation_adapter" not in rt.checkpoint_meta()
    finally:
        rt.release()


def test_auto_rejects_different_control_period(estimator_path):
    env = Env()
    env.cfg.sim.control_rate = 50.0
    rt = gr.ControllerRuntime(env, "auto", estimator_path)
    with pytest.raises(ValueError, match="control_dt"):
        rt.install()


@pytest.mark.parametrize("field", ["v_max_policy", "imu_gyro_scale", "imu_accel_scale"])
def test_auto_rejects_invalid_source_normalization(field, estimator_path):
    env = Env()
    setattr(env.ecfg, field, float("nan"))
    rt = gr.ControllerRuntime(env, "auto", estimator_path)
    with pytest.raises(ValueError, match="normalizers"):
        rt.install()


@pytest.mark.parametrize("change", ["order", "scales"])
def test_auto_rejects_unsupported_estimator_feature_contract(change, tmp_path):
    spec = FeatureSpec()
    if change == "order":
        spec = replace(spec, names=(spec.names[1], spec.names[0]) + spec.names[2:])
    else:
        spec = replace(spec, scales=(20.0,) + spec.scales[1:])
    path = tmp_path / "unsupported.pt"
    save_grip_estimator(path, QuantileGripNet(spec, width=4, hidden=8), spec, {}, {})
    rt = gr.ControllerRuntime(Env(), "auto", str(path))
    with pytest.raises(ValueError, match="canonical estimator feature"):
        rt.install()


def test_default_auto_matches_explicit_historical_solver_bit_for_bit():
    env = Env(batch=3)
    sp = env.tracker.spec
    with torch.random.fork_rng():
        torch.manual_seed(640)
        a = torch.rand(3, mpc.ACT_DIM) * 2 - 1
        args = [a, torch.tensor([.1, 5., 11.]), torch.full((3,), 10.), torch.tensor([0., .2, 1.]),
                torch.full((3,), sp.delay), torch.zeros(3, 2), torch.randn(3, sp.N, 2)]
    mu = torch.tensor([.73423, .94401, 1.15379])
    default = gr.automatic_grip_spec(env)
    historical = replace(default, profile_version="historical-global-v1")
    a = gc.solve_grip(*args, mu, sp, .3302, .4189, 10., default)
    b = gc.solve_grip(*args, mu, sp, .3302, .4189, 10., historical)
    assert all(torch.equal(left, right) for left, right in zip(a, b))
