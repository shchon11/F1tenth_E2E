"""Behavioral contract for evidence-gated friction inference. CPU only."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from f1sim.learn.adaptive_grip import (AdaptiveConsumption, AdaptiveGripEstimator, AdaptiveGripNet,
    adaptive_feature_spec, load_adaptive_grip_estimator, save_adaptive_grip_estimator)
from f1sim.learn.adaptive_grip_train import adaptive_loss, sensor_observation


class FixedNet(nn.Module):
    def __init__(self, q10=0.55, logit=5.0):
        super().__init__()
        self.spec = adaptive_feature_spec()
        self.arch = {"hidden": 4, "wheelbase": 0.33}
        self.q10, self.logit = q10, logit

    def distribution(self, x, valid):
        q = torch.tensor([self.q10, self.q10 + .15, self.q10 + .3]).expand(x.shape[0], 3).clone()
        return q, torch.full((x.shape[0],), self.logit)

    def forward(self, x, valid):
        return self.distribution(x, valid)[0]


def tensors(batch=2, warm=40):
    x = torch.zeros(batch, 40, 11)
    x[..., 6] = .981
    v = torch.zeros(batch, 40, dtype=torch.bool)
    v[:, :warm] = True
    return x, v


def estimator(**kwargs):
    net = FixedNet(**kwargs)
    return AdaptiveGripEstimator(net, net.spec, {"cal_delta": .05}, {})


def test_cold_and_no_information_follow_nominal():
    e = estimator()
    x, valid = tensors(warm=7)
    used, diag = e.lower_mu(x, valid)
    assert torch.allclose(used, torch.full((2,), e.consumption.nominal_mu))
    assert not diag["has_evidence"].any()
    e.net.logit = -5.0
    used, diag = e.lower_mu(*tensors())
    assert torch.allclose(used, torch.full((2,), e.consumption.nominal_mu))
    assert not diag["informative"].any()


def test_informative_update_holds_without_evidence_then_resets():
    e = estimator()
    used, diag = e.lower_mu(*tensors(warm=8))
    assert torch.allclose(used, torch.full((2,), .50))
    assert diag["has_evidence"].all()
    e.net.logit = -5.0
    e.net.q10 = 1.2
    held, diag = e.lower_mu(*tensors())
    assert torch.equal(held, used)
    assert diag["has_evidence"].all() and not diag["informative"].any()
    e.reset_filter(torch.tensor([0]))
    reset, diag = e.lower_mu(*tensors())
    assert reset[0] == e.consumption.nominal_mu
    assert reset[1] == .5
    assert diag["has_evidence"].tolist() == [False, True]
    e.reset_filter()
    assert (e.lower_mu(*tensors())[0] == e.consumption.nominal_mu).all()


def test_friction_drop_immediate_gain_smoothed_and_no_old_mu_floor():
    e = estimator(q10=.25)
    used, _ = e.lower_mu(*tensors())
    assert torch.allclose(used, torch.full((2,), .20))
    e.net.q10 = .8
    next_mu, _ = e.lower_mu(*tensors())
    assert torch.allclose(next_mu, used + e.alpha * (.75 - used))
    e.net.q10 = .15
    assert torch.allclose(e.lower_mu(*tensors())[0], torch.full((2,), .10))


def test_fault_is_conservative_and_latched_until_evidence_or_reset():
    e = estimator(logit=-5.0)
    x, v = tensors()
    x[0, 0, 4] = float("nan")
    used, diag = e.lower_mu(x, v)
    assert used[0] == e.consumption.fault_mu and diag["fault"][0]
    used, diag = e.lower_mu(*tensors())
    assert used[0] == e.consumption.fault_mu
    assert diag["fault"].tolist() == [True, False]
    assert not diag["fault_now"].any()
    e.net.logit = 5.0
    assert not e.lower_mu(*tensors())[1]["fault"].any()
    e.reset_filter()
    e.net.logit = -5.0
    assert (e.lower_mu(*tensors())[0] == e.consumption.nominal_mu).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_predictions_never_admitted(bad):
    e = estimator(q10=bad)
    mu, diag = e.lower_mu(*tensors())
    assert torch.isfinite(mu).all()
    assert not diag["informative"].any()
    assert diag["fault"].all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_confidence_logits_fault_even_when_sigmoid_is_finite(bad):
    e = estimator(logit=bad)
    mu, diag = e.lower_mu(*tensors())
    assert torch.isfinite(mu).all()
    assert not diag["informative"].any()
    assert diag["fault"].all()


def test_stateless_candidate_does_not_advance_live_filter():
    e = estimator()
    first = e.lower_mu(*tensors())[0].clone()
    e.net.q10 = .95
    e.calibrated_candidate(*tensors(batch=5))
    assert torch.equal(e._filtered, first)
    assert torch.allclose(e.lower_mu(*tensors())[0], first + e.alpha * (.9 - first))


def test_network_padding_is_inert_and_quantiles_are_ordered():
    torch.manual_seed(123)
    net = AdaptiveGripNet(hidden=8)
    x, v = tensors(warm=8)
    q = net(x, v)
    x[:, 8:] = float("nan")
    assert torch.equal(net(x, v), q)
    assert (q[:, 1:] >= q[:, :-1]).all()
    assert torch.isfinite(q).all()


def test_internal_deltas_are_causal_chronological_and_wheel_based():
    net = AdaptiveGripNet(hidden=8)
    x, v = tensors(batch=1, warm=3)
    x[0, :3, 0] = torch.tensor([.3, .2, .1])
    d = net.derived_features(x, v)
    assert d[0, -3, 11] == 0  # oldest valid frame has no preceding wheel speed
    assert torch.allclose(d[0, -2:, 11], torch.full((2,), 2.0))
    x[0, 0, 0] = .9
    assert torch.equal(net.derived_features(x, v)[0, :-1], d[0, :-1])


def test_no_excitation_loss_never_regresses_to_mu_prior():
    q = torch.tensor([[.5, .8, 1.1]], requires_grad=True)
    logits = torch.tensor([-1.], requires_grad=True)
    data = {"informative": torch.tensor([False]), "mu": torch.tensor([.6]), "lower_bound": torch.tensor([.3])}
    a = adaptive_loss(q, logits, data)
    data["mu"] = torch.tensor([1.4])
    assert torch.equal(a, adaptive_loss(q, logits, data))
    a.backward()
    assert torch.equal(q.grad, torch.zeros_like(q))
    assert logits.grad.abs().sum() > 0


def test_training_observation_ignores_body_truth_and_preserves_sensor_scale():
    r = SimpleNamespace(imu=torch.full((2, 2, 6), 60.), imu_att=torch.ones(2, 3),
                        odom=torch.full((2, 5), 30.), state=torch.full((2, 8), float("nan")))
    obs = sensor_observation(r)
    assert obs["speed"][0, 0] == 3  # no training-only clipping
    assert obs["imu"][0, 0] == 12
    assert obs["imu"][0, 4] == 6
    r.state = torch.zeros_like(r.state)
    assert all(torch.equal(value, sensor_observation(r)[key]) for key, value in obs.items())


def test_checkpoint_roundtrip_retains_consumption_and_architecture(tmp_path):
    net = AdaptiveGripNet(hidden=8)
    path = tmp_path / "candidate.pt"
    config = AdaptiveConsumption(nominal_mu=1.1, confidence_threshold=.7)
    sha = save_adaptive_grip_estimator(path, net, net.spec, {"cal_delta": .03}, {"deployment_approved": False}, config)
    loaded = load_adaptive_grip_estimator(path)
    assert loaded.sha == sha
    assert loaded.consumption == config
    assert loaded.arch["model_kind"] == "adaptive_grip_v2"
    assert loaded.meta["format"] == "adaptive_grip_v2"
    assert torch.equal(loaded.quantiles(*tensors()), net(*tensors()))
    ck = torch.load(path, weights_only=False)
    del ck["consumption"]["fault_mu"]
    torch.save(ck, path)
    with pytest.raises(ValueError, match="consumption"):
        load_adaptive_grip_estimator(path)


def test_rejects_invalid_calibration_and_contract():
    net = AdaptiveGripNet(hidden=8)
    with pytest.raises(ValueError, match="cal_delta"):
        AdaptiveGripEstimator(net, net.spec, {"cal_delta": -.1}, {})
    with pytest.raises(ValueError, match="canonical"):
        AdaptiveGripNet(replace(net.spec, scales=(1.,) * 11))


@pytest.mark.parametrize("approval", [None, False, "true"])
def test_production_loader_refuses_unreviewed_adaptive_candidate(tmp_path, approval):
    from f1sim.learn.grip_estimator import load_grip_estimator
    net = AdaptiveGripNet(hidden=8)
    path = tmp_path / "candidate.pt"
    meta = {} if approval is None else {"deployment_approved": approval}
    save_adaptive_grip_estimator(path, net, net.spec, {"cal_delta": 0.}, meta)
    with pytest.raises(ValueError, match="deployment review"):
        load_grip_estimator(path)


def test_production_loader_dispatches_reviewed_adaptive_model(tmp_path):
    from f1sim.learn.grip_estimator import load_grip_estimator
    net = AdaptiveGripNet(hidden=8)
    path = tmp_path / "reviewed-test-model.pt"
    save_adaptive_grip_estimator(path, net, net.spec, {"cal_delta": 0.},
                                {"deployment_approved": True, "test_fixture": True})
    loaded = load_grip_estimator(path)
    assert isinstance(loaded, AdaptiveGripEstimator)
    mu, diag = loaded.lower_mu(*tensors(warm=1))
    assert (mu == loaded.consumption.nominal_mu).all()
    assert not diag["has_evidence"].any()


def test_posterior_diagnostic_is_separate_from_applied_calibrated_bound():
    e = estimator(q10=.55)
    used, diag = e.lower_mu(*tensors())
    assert torch.allclose(diag["posterior_mu"], torch.full((2,), .70))
    assert torch.allclose(diag["calibrated_lower_mu"], torch.full((2,), .50))
    assert torch.allclose(diag["calibration_discount"], torch.full((2,), .05))
    assert torch.equal(used, diag["calibrated_lower_mu"])
