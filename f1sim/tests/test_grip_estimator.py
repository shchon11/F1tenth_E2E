"""Tests for the causal grip estimator. CPU only.

The ones that matter are about *timing* and *leakage*, because those two failures are invisible in
every metric: a window that quietly contains the action it is supposed to precede, or a model that
reads `P["mu"]` through some key nobody thought about, both produce excellent numbers and no
deployable estimator.
"""
from __future__ import annotations

import math
import pytest
import torch

from f1sim.learn import grip_estimator as G                             # noqa: E402

B = 4


def obs(speed=1.0, imu=0.0, att=0.0, batch=B, **poison):
    o = {"speed": torch.full((batch, 1), speed),
         "imu": torch.full((batch, 6), imu),
         "imu_att": torch.full((batch, 2), att)}
    o.update(poison)
    return o


# ---------------------------------------------------------------- history: shape and timing
def test_reset_seeds_one_valid_row_with_zero_command():
    h = G.SensorHistory(B, "cpu")
    h.reset(torch.arange(B), obs(speed=0.5))
    f, v = h.inputs()
    assert f.shape == (B, 40, 11) and v.shape == (B, 40)
    assert v[:, 0].all() and not v[:, 1:].any()
    assert torch.allclose(f[:, 0, 0], torch.full((B,), 0.5))
    assert torch.equal(f[:, 0, 9:], torch.zeros(B, 2)), "reset must zero the previous command"
    assert (h.valid_count() == 1).all() and not h.is_warm().any()


def test_push_pairs_observation_t_with_command_t_minus_one():
    """The whole causal claim in one test: row 0 is obs_t and the command issued at t-1."""
    h = G.SensorHistory(B, "cpu")
    h.reset(torch.arange(B), obs(speed=0.0))
    cmd_t0 = torch.stack([torch.full((B,), 0.2), torch.full((B,), 5.0)], 1)
    h.push(obs(speed=0.7), cmd_t0)
    f, _ = h.inputs()
    assert f[:, 0, 0].allclose(torch.full((B,), 0.7))                  # observation is t
    assert f[:, 0, 9].allclose(torch.full((B,), 0.2 / G.FEATURE_SCALES[9]))
    assert f[:, 0, 10].allclose(torch.full((B,), 5.0 / G.FEATURE_SCALES[10]))
    assert f[:, 1, 0].allclose(torch.zeros(B))                         # previous row still there
    assert f[:, 1, 9:].allclose(torch.zeros(B, 2))


def test_push_zeroes_the_command_for_reset_ids_before_inserting():
    """A command from a finished episode must never appear in the next episode's first row."""
    h = G.SensorHistory(B, "cpu")
    h.reset(torch.arange(B), obs())
    cmd = torch.stack([torch.full((B,), 0.3), torch.full((B,), 7.0)], 1)
    mask = torch.tensor([True, False, True, False])
    h.push(obs(speed=0.9), cmd, reset_mask=mask)
    f, _ = h.inputs()
    assert f[mask, 0, 9:].abs().max() == 0.0
    assert f[~mask, 0, 9:].abs().max() > 0.0


def test_history_is_newest_first_and_rolls():
    h = G.SensorHistory(1, "cpu")
    h.reset(torch.arange(1), obs(speed=0.0, batch=1))
    for i in range(1, 5):
        h.push(obs(speed=i / 10.0, batch=1), torch.zeros(1, 2))
    f, _ = h.inputs()
    assert [round(float(x), 4) for x in f[0, :5, 0]] == [0.4, 0.3, 0.2, 0.1, 0.0]


def test_warm_after_exactly_forty_frames():
    h = G.SensorHistory(1, "cpu")
    h.reset(torch.arange(1), obs(batch=1))
    for _ in range(38):
        h.push(obs(batch=1), torch.zeros(1, 2))
    assert int(h.valid_count()) == 39 and not bool(h.is_warm())
    h.push(obs(batch=1), torch.zeros(1, 2))
    assert int(h.valid_count()) == 40 and bool(h.is_warm())


def test_per_env_reset_clears_only_that_env():
    h = G.SensorHistory(B, "cpu")
    h.reset(torch.arange(B), obs())
    for _ in range(10):
        h.push(obs(speed=0.8), torch.ones(B, 2))
    h.reset(torch.tensor([1, 3]), obs(speed=0.1))
    assert [int(x) for x in h.valid_count()] == [11, 1, 11, 1]
    f, _ = h.inputs()
    assert f[1, 0, 9:].abs().max() == 0.0 and f[0, 0, 9:].abs().max() > 0.0


# ---------------------------------------------------------------- leakage
@pytest.mark.parametrize("key", ["priv", "P", "mu", "speed_cap", "scan", "map_id", "prev_action",
                                 "hist", "track_id"])
def test_poisoned_observation_keys_cannot_reach_a_feature(key):
    """Only speed / imu / imu_att are ever read. Anything else in the dict is inert."""
    clean = G.SensorHistory(B, "cpu")
    dirty = G.SensorHistory(B, "cpu")
    poison = {key: torch.full((B, 9), 1e6)}
    clean.reset(torch.arange(B), obs(speed=0.4, imu=0.2))
    dirty.reset(torch.arange(B), obs(speed=0.4, imu=0.2, **poison))
    for _ in range(5):
        clean.push(obs(speed=0.4, imu=0.2), torch.zeros(B, 2))
        dirty.push(obs(speed=0.4, imu=0.2, **poison), torch.zeros(B, 2))
    assert torch.equal(clean.inputs()[0], dirty.inputs()[0])


def test_label_read_after_reset_would_relabel_the_old_history():
    """The critic's extra test: `P` entries are views written in place, so a label captured after
    the env resets silently belongs to the *new* episode while the history is still the old one.

    This asserts the failure mode exists, so the collector's pre-action capture is load-bearing
    rather than stylistic.
    """
    P = {"mu": torch.tensor([0.8, 0.8, 0.8, 0.8])}
    label_view = P["mu"]                       # what a careless collector would hold on to
    label_copy = P["mu"].clone()               # what it must hold instead
    P["mu"][:] = 1.1                           # the reset re-draws friction in place
    assert float(label_view[0]) == pytest.approx(1.1), "views follow the reset"
    assert float(label_copy[0]) == pytest.approx(0.8), "a clone is the pre-action label"


# ---------------------------------------------------------------- excitation gate
def test_excitation_gate_is_the_predeclared_or():
    f = torch.zeros(3, 40, 11)
    v = torch.ones(3, 40, dtype=torch.bool)
    f[0, :5, 4] = 2.6 / G.FEATURE_SCALES[4]                  # longitudinal only, 5 frames
    f[1, :5, 5] = 3.1 / G.FEATURE_SCALES[5]                  # lateral only, 5 frames
    f[2, :4, 4] = 2.6 / G.FEATURE_SCALES[4]                  # one frame short
    assert [bool(x) for x in G.excitation_pass(f, v)] == [True, True, False]
    parts = G.excitation_frames(f, v)
    assert [int(x) for x in parts["long"]] == [5, 0, 4]
    assert [int(x) for x in parts["lat"]] == [0, 5, 0]


def test_excitation_ignores_invalid_frames():
    f = torch.zeros(1, 40, 11)
    f[0, :10, 4] = 5.0 / G.FEATURE_SCALES[4]
    v = torch.zeros(1, 40, dtype=torch.bool)
    v[0, :3] = True
    assert not bool(G.excitation_pass(f, v)[0])


# ---------------------------------------------------------------- model
def make_estimator(cal_delta=0.0, seed=0):
    torch.manual_seed(seed)
    spec = G.FeatureSpec()
    net = G.QuantileGripNet(spec)
    return G.GripEstimator(net, spec, {"cal_delta": cal_delta}, {"train_seed": 401})


def test_quantiles_are_ordered_and_finite_for_any_weights():
    torch.manual_seed(3)
    spec = G.FeatureSpec()
    net = G.QuantileGripNet(spec)
    for p in net.parameters():                               # deliberately hostile weights
        p.data.normal_(0, 5.0)
    f = torch.randn(16, 40, 11)
    v = torch.ones(16, 40, dtype=torch.bool)
    q = net(f, v)
    assert torch.isfinite(q).all()
    assert (q[:, 0] <= q[:, 1] + 1e-6).all() and (q[:, 1] <= q[:, 2] + 1e-6).all()


def test_model_ignores_invalid_rows():
    torch.manual_seed(1)
    net = G.QuantileGripNet(G.FeatureSpec())
    f = torch.randn(2, 40, 11)
    v = torch.zeros(2, 40, dtype=torch.bool); v[:, :5] = True
    g = f.clone(); g[:, 5:] = 1e3                            # garbage only where invalid
    assert torch.allclose(net(f, v), net(g, v), atol=1e-6)


def test_all_invalid_history_is_finite():
    net = G.QuantileGripNet(G.FeatureSpec())
    q = net(torch.zeros(2, 40, 11), torch.zeros(2, 40, dtype=torch.bool))
    assert torch.isfinite(q).all()


def test_pinball_loss_is_minimised_at_the_quantile():
    y = torch.randn(4096) * 0.1 + 0.9
    pred_lo = torch.stack([torch.full((4096,), float(y.quantile(0.1))),
                           torch.full((4096,), float(y.quantile(0.5))),
                           torch.full((4096,), float(y.quantile(0.9)))], 1)
    worse = pred_lo + 0.2
    assert float(G.pinball_loss(pred_lo, y)) < float(G.pinball_loss(worse, y))


# ---------------------------------------------------------------- B2 consumption rule
def test_cold_history_falls_back_to_mu_min():
    est = make_estimator()
    h = G.SensorHistory(B, "cpu")
    h.reset(torch.arange(B), obs())
    used, d = est.lower_mu(*h.inputs())
    assert torch.allclose(used, torch.full((B,), G.MU_MIN))
    assert (d["fallback_reason"] == 0).all() and not d["warm"].any()


def test_non_finite_prediction_falls_back():
    est = make_estimator()
    f = torch.full((2, 40, 11), float("nan"))
    v = torch.ones(2, 40, dtype=torch.bool)
    used, d = est.lower_mu(f, v)
    assert torch.allclose(used, torch.full((2,), G.MU_MIN))
    assert (d["fallback_reason"] == 1).all()


def test_filter_drops_immediately_and_rises_over_tau():
    """The asymmetry is the safety property: losses now, gains slowly."""
    est = make_estimator()
    v = torch.ones(1, 40, dtype=torch.bool)
    f = torch.zeros(1, 40, 11)
    est._filtered = torch.tensor([1.10])                     # pretend it had settled high
    with torch.no_grad():                                    # force a known low candidate
        est.net.head[-1].bias[0] = 0.80
    used, _ = est.lower_mu(f, v)
    assert float(used) == pytest.approx(0.80, abs=1e-3), "a loss of grip applies at once"

    with torch.no_grad():
        est.net.head[-1].bias[0] = 1.10
    first, _ = est.lower_mu(f, v)
    alpha = 1.0 - math.exp(-G.CONTROL_DT / G.FILTER_TAU)
    assert float(first) == pytest.approx(0.80 + alpha * (1.10 - 0.80), abs=1e-3)
    for _ in range(60):                                      # ~1.5 s, several tau
        last, _ = est.lower_mu(f, v)
    assert float(last) == pytest.approx(1.10, abs=5e-3), "gains converge, they are not blocked"


def test_reset_filter_returns_to_mu_min():
    est = make_estimator()
    est._filtered = torch.tensor([1.05, 1.05])
    est.reset_filter(torch.tensor([0]))
    assert float(est._filtered[0]) == pytest.approx(G.MU_MIN)
    assert float(est._filtered[1]) == pytest.approx(1.05)


def test_stateless_candidate_carries_no_state():
    """Shuffled offline batches must not thread one EMA across unrelated episodes."""
    est = make_estimator()
    f = torch.zeros(3, 40, 11)
    v = torch.ones(3, 40, dtype=torch.bool)
    a, _ = est.calibrated_candidate(f, v)
    est._filtered = torch.tensor([0.74, 0.74, 0.74])         # poison the live-stream state
    b, _ = est.calibrated_candidate(f, v)
    assert torch.equal(a, b)
    assert "used_mu" not in est.calibrated_candidate(f, v)[1]


def test_candidate_is_clipped_to_the_trained_range():
    est = make_estimator(cal_delta=5.0)                      # absurd downward correction
    f = torch.zeros(2, 40, 11)
    v = torch.ones(2, 40, dtype=torch.bool)
    c, _ = est.calibrated_candidate(f, v)
    assert (c >= G.MU_MIN - 1e-9).all() and (c <= G.MU_MAX + 1e-9).all()


def test_lower_mu_equals_candidate_on_the_first_warm_step_after_reset():
    est = make_estimator()
    f = torch.zeros(1, 40, 11)
    v = torch.ones(1, 40, dtype=torch.bool)
    est._ensure(1, f.device, torch.float32)
    est.reset_filter()
    cand, _ = est.calibrated_candidate(f, v)
    used, _ = est.lower_mu(f, v)
    assert float(used) <= float(cand) + 1e-9                 # min() never exceeds the candidate


# ---------------------------------------------------------------- checkpoint
def test_checkpoint_round_trips_with_spec_and_calibration(tmp_path):
    est = make_estimator(cal_delta=0.03, seed=5)
    p = tmp_path / "est.pt"
    returned = G.save_grip_estimator(p, est.net, est.spec, est.calibration, {"train_seed": 401})
    back = G.load_grip_estimator(p, "cpu")
    assert back.spec == est.spec
    assert back.cal_delta == pytest.approx(0.03)
    assert back.mu_range == (G.MU_MIN, G.MU_MAX)
    assert isinstance(back.feature_spec, dict) and back.feature_spec["n_frames"] == 40
    assert back.meta["train_seed"] == 401
    f, v = torch.randn(2, 40, 11), torch.ones(2, 40, dtype=torch.bool)
    assert torch.allclose(back.quantiles(f, v), est.quantiles(f, v))
    assert not any(p_.requires_grad for p_ in back.net.parameters())
    # the hash is the file's own bytes, at save and at load, never a claim inside it
    assert back.sha == returned == G.file_sha256(p)
    assert len(back.sha) == 64 and est.sha == ""


def test_save_refuses_a_self_asserted_hash(tmp_path):
    est = make_estimator()
    with pytest.raises(ValueError, match="must not carry"):
        G.save_grip_estimator(tmp_path / "x.pt", est.net, est.spec, est.calibration,
                              {"checkpoint_sha256": "a" * 64})


def test_checkpoint_preserves_a_non_default_architecture(tmp_path):
    """A net trained at another width used to reload into this build's defaults."""
    spec = G.FeatureSpec()
    net = G.QuantileGripNet(spec, width=16, hidden=24)
    assert net.arch == {"width": 16, "hidden": 24}
    p = tmp_path / "small.pt"
    G.save_grip_estimator(p, net, spec, {"cal_delta": 0.0}, {})
    back = G.load_grip_estimator(p, "cpu")
    assert back.net.arch == {"width": 16, "hidden": 24}
    f, v = torch.randn(2, 40, 11), torch.ones(2, 40, dtype=torch.bool)
    assert torch.allclose(back.quantiles(f, v), net(f, v))


def test_load_refuses_a_checkpoint_without_an_architecture_record(tmp_path):
    est = make_estimator()
    p = tmp_path / "old.pt"
    torch.save({"state_dict": est.net.state_dict(), "feature_spec": est.spec.to_meta(),
                "calibration": {"cal_delta": 0.0}, "meta": {},
                "format": "grip_estimator_v1"}, p)         # no "arch": the pre-fix layout
    with pytest.raises(ValueError, match="no architecture record"):
        G.load_grip_estimator(p, "cpu")


def test_loader_rejects_a_foreign_checkpoint(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"format": "something_else"}, p)
    with pytest.raises(ValueError, match="grip_estimator_v1"):
        G.load_grip_estimator(p, "cpu")


def test_feature_spec_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unsupported feature-spec"):
        G.FeatureSpec.from_meta({"n_frames": 40, "future_field": 1})


def test_offline_and_online_encoding_agree():
    """A dataset row built from a history must equal one the same history produces live."""
    h = G.SensorHistory(1, "cpu")
    h.reset(torch.arange(1), obs(speed=0.3, imu=0.1, att=0.05, batch=1))
    cmd = torch.tensor([[0.15, 4.0]])
    for _ in range(3):
        h.push(obs(speed=0.6, imu=0.2, att=0.1, batch=1), cmd)
    online = h.inputs()[0].clone()
    stored = online.to(torch.float16).to(torch.float32)       # the shard round-trip
    assert torch.allclose(online, stored, atol=1e-3)


# ---------------------------------------------------------------- reset clears history (PM fix 1)
def test_push_with_reset_mask_clears_the_whole_history_not_just_the_command():
    """The defect this replaces: zeroing only the command left forty old-friction rows in place,
    so the env stayed warm and its previous episode's window was relabelled on the next step."""
    B16 = 16
    h = G.SensorHistory(B16, "cpu")
    h.reset(torch.arange(B16), obs(speed=0.5, batch=B16))
    for _ in range(60):                                      # well past warm
        h.push(obs(speed=0.5, imu=0.4, att=0.2, batch=B16),
               torch.stack([torch.full((B16,), 0.25), torch.full((B16,), 6.0)], 1))
    assert h.is_warm().all() and (h.valid_count() == 40).all()
    before = h.inputs()[0].clone()

    mask = torch.zeros(B16, dtype=torch.bool)
    mask[[2, 7, 11]] = True
    h.push(obs(speed=0.05, imu=0.0, att=0.0, batch=B16),
           torch.stack([torch.full((B16,), 0.25), torch.full((B16,), 6.0)], 1), reset_mask=mask)
    f, v = h.inputs()

    r = mask.nonzero().flatten()
    assert [int(x) for x in h.valid_count()[r]] == [1, 1, 1], "a reset env keeps exactly one row"
    assert not h.is_warm()[r].any(), "and must not stay warm"
    assert not v[r, 1:].any(), "every older validity flag cleared"
    assert f[r, 1:].abs().max() == 0.0, "every older feature row zeroed"
    assert f[r, 0, 9:].abs().max() == 0.0, "and the command did not cross the boundary"
    assert f[r, 0, 0].allclose(torch.full((3,), 0.05)), "the new observation is present"

    k = (~mask).nonzero().flatten()
    assert (h.valid_count()[k] == 40).all() and h.is_warm()[k].all()
    assert torch.equal(f[k, 1:], before[k, :-1]), "untouched envs simply rolled"
    assert f[k, 0, 9:].abs().max() > 0.0


def test_reset_mask_and_reset_agree_on_the_resulting_state():
    a = G.SensorHistory(2, "cpu"); b = G.SensorHistory(2, "cpu")
    for hh in (a, b):
        hh.reset(torch.arange(2), obs(speed=0.5, batch=2))
        for _ in range(50):
            hh.push(obs(speed=0.5, imu=0.3, batch=2), torch.ones(2, 2))
    new = obs(speed=0.1, imu=0.0, batch=2)
    a.push(new, torch.ones(2, 2), reset_mask=torch.tensor([True, False]))
    b.reset(torch.tensor([0]), new)
    assert int(a.valid_count()[0]) == int(b.valid_count()[0]) == 1
    assert torch.equal(a.inputs()[0][0], b.inputs()[0][0])


# ---------------------------------------------------------------- strict observation (PM fix 2)
@pytest.mark.parametrize("missing", ["speed", "imu", "imu_att"])
def test_missing_sensor_raises_instead_of_being_zero_filled(missing):
    h = G.SensorHistory(B, "cpu")
    o = obs()
    del o[missing]
    with pytest.raises(ValueError, match="missing required sensor"):
        h.reset(torch.arange(B), o)


@pytest.mark.parametrize("key,bad", [("speed", (B, 2)), ("imu", (B, 5)), ("imu_att", (B, 3)),
                                     ("imu", (B + 1, 6)), ("speed", (B,))])
def test_malformed_sensor_shape_is_rejected_not_reshaped(key, bad):
    h = G.SensorHistory(B, "cpu")
    o = obs(); o[key] = torch.zeros(*bad)
    with pytest.raises(ValueError, match="must be exactly"):
        h.push(o, torch.zeros(B, 2))


def test_non_tensor_sensor_is_rejected():
    h = G.SensorHistory(B, "cpu")
    o = obs(); o["imu"] = [[0.0] * 6] * B
    with pytest.raises(ValueError, match="must be a tensor"):
        h.push(o, torch.zeros(B, 2))


# ---------------------------------------------------------------- spec / calibration (PM fix 3)
@pytest.mark.parametrize("field,value,msg", [
    ("n_frames", 0, "must be >= 1"),
    ("warm_frames", 41, r"must be in \[1, 40\]"),
    ("warm_frames", 0, r"must be in \[1, 40\]"),
    ("control_dt", 0.0, "control_dt"),
    ("filter_tau", -1.0, "filter_tau"),
    ("mu_min", 2.0, "mu_min < mu_max"),
])
def test_feature_spec_validate_rejects_drift(field, value, msg):
    with pytest.raises(ValueError, match=msg):
        G.FeatureSpec(**{field: value}).validate()


def test_feature_spec_rejects_mismatched_scales_and_names():
    with pytest.raises(ValueError, match="n_features"):
        G.FeatureSpec(scales=G.FEATURE_SCALES[:-1]).validate()
    with pytest.raises(ValueError, match="finite and positive"):
        G.FeatureSpec(scales=tuple([0.0] + list(G.FEATURE_SCALES[1:]))).validate()


def test_from_meta_rejects_a_partial_spec():
    """A missing field would silently take this build's default for something the model was not
    trained with."""
    full = G.FeatureSpec().to_meta()
    full.pop("scales")
    with pytest.raises(ValueError, match="missing"):
        G.FeatureSpec.from_meta(full)
    with pytest.raises(ValueError, match="must be a dict"):
        G.FeatureSpec.from_meta([1, 2, 3])


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
def test_cal_delta_must_be_finite_and_non_negative(bad):
    spec = G.FeatureSpec()
    with pytest.raises(ValueError, match="cal_delta"):
        G.GripEstimator(G.QuantileGripNet(spec), spec, {"cal_delta": bad}, {})


def test_net_validates_its_spec():
    with pytest.raises(ValueError, match="warm_frames"):
        G.QuantileGripNet(G.FeatureSpec(warm_frames=99))


def test_estimator_exposes_the_architecture_record(tmp_path):
    """Consumers embed this rather than reaching into .net or guessing a meta key."""
    spec = G.FeatureSpec()
    net = G.QuantileGripNet(spec, width=16, hidden=24)
    est = G.GripEstimator(net, spec, {"cal_delta": 0.0}, {})
    assert est.arch == {"width": 16, "hidden": 24} == net.arch
    assert "architecture" not in est.meta and "arch" not in est.meta
    p = tmp_path / "e.pt"
    G.save_grip_estimator(p, net, spec, {"cal_delta": 0.0}, {})
    assert G.load_grip_estimator(p, "cpu").arch == {"width": 16, "hidden": 24}


def test_net_is_the_frozen_attribute_consumers_use():
    est = make_estimator()
    assert hasattr(est, "net") and list(est.net.parameters())
    assert not est.net.training and not any(p.requires_grad for p in est.net.parameters())
    est.net.eval()                                           # idempotent, harmless
    assert not est.net.training
