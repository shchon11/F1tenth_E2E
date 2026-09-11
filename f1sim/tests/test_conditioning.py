"""Stage-1 conditional-policy pilot: migration, parity, gradients, and the critic adapter.

The experiment compares two arms that differ only in what is written into `c`. Everything here
protects that "only": a migration that silently reinitialises a weight, a conditioning path that
receives no gradient, or a policy that substitutes zeros for a missing input would all show up in the
results as an effect of conditioning.
"""
from __future__ import annotations

import pytest
import torch

from f1sim.learn import conditioning as C
from f1sim.learn.model import (ActorCritic, load_checkpoint, load_for_conditioning,
                               save_checkpoint)

SHAPE = dict(n_stack=2, n_beams=64, proprio_dim=12, act_dim=8)


def _base(priv_dim=17, seed=0):
    torch.manual_seed(seed)
    return ActorCritic(priv_dim=priv_dim, **SHAPE)


def _cond_from(base, source="true_mu", priv_adapter=None, priv_dim=17):
    spec = C.spec_for(source)
    m = ActorCritic(priv_dim=priv_dim, cond_dim=spec.dim, cond=spec.to_meta(),
                    priv_adapter=priv_adapter, **SHAPE)
    sd = m.state_dict()
    for k, v in base.state_dict().items():
        sd[k] = v
    m.load_state_dict(sd)
    return m, spec


def _batch(b=4):
    torch.manual_seed(1)
    return torch.randn(b, SHAPE["n_stack"], SHAPE["n_beams"]), torch.randn(b, SHAPE["proprio_dim"])


# ---------------------------------------------------------------- forward parity
@pytest.mark.parametrize("value", [0.0, 1.0, -5.0, 100.0])
def test_conditioning_is_exactly_zero_at_init(value):
    """Not "close": the added term is `0 @ c`, so the two actions are the same number."""
    base = _base()
    cond, _ = _cond_from(base)
    scan, pro = _batch()
    a0 = base.actor(scan, pro)
    a1 = cond.actor(scan, pro, torch.full((scan.shape[0], 1), value))
    assert float((a0 - a1).abs().max()) == 0.0


def test_condition_is_never_implied():
    """A conditional actor refuses a missing input rather than substituting the A0 arm's zeros."""
    base = _base()
    cond, _ = _cond_from(base)
    scan, pro = _batch()
    with pytest.raises(ValueError, match="requires an explicit"):
        cond.actor(scan, pro)
    with pytest.raises(ValueError, match="unconditional"):
        base.actor(scan, pro, torch.zeros(scan.shape[0], 1))
    with pytest.raises(ValueError, match="condition must be"):
        cond.actor(scan, pro, torch.zeros(scan.shape[0], 3))


# ---------------------------------------------------------------- gradients
def test_conditioning_path_gets_gradient_and_leaves_the_rest_alone():
    """`Wc` must train from the first update, and no other parameter's gradient may move.

    Zero-initialising the projection makes the forward identical; it must not also make the backward
    identical, or the arm could never learn to use its input. `dL/dW = delta_pre . c^T`, so a nonzero
    `c` is what makes this hold -- which is also why the A0 arm (c = 0) leaves `Wc` at zero forever,
    exactly as intended.
    """
    base = _base()
    cond, _ = _cond_from(base)
    scan, pro = _batch()
    c = torch.full((scan.shape[0], 1), 0.7)

    base.actor(scan, pro).square().sum().backward()
    cond.actor(scan, pro, c).square().sum().backward()

    assert cond.actor.cond.weight.grad is not None
    assert float(cond.actor.cond.weight.grad.abs().max()) > 0.0

    bg = {n: p.grad for n, p in base.actor.named_parameters() if p.grad is not None}
    cg = {n: p.grad for n, p in cond.actor.named_parameters() if p.grad is not None}
    assert set(cg) - set(bg) == {"cond.weight"}
    for n, g in bg.items():
        assert torch.equal(g, cg[n]), f"gradient of {n} changed"


def test_zero_condition_leaves_the_projection_unmoved():
    """The A0 arm's `c` is zeros, so its conditioning weight receives exactly zero gradient."""
    base = _base()
    cond, _ = _cond_from(base, source="zero")
    scan, pro = _batch()
    cond.actor(scan, pro, torch.zeros(scan.shape[0], 1)).square().sum().backward()
    assert float(cond.actor.cond.weight.grad.abs().max()) == 0.0


# ---------------------------------------------------------------- migration and IO
def test_strict_migration_transfers_every_legacy_weight(tmp_path):
    base = _base()
    p = tmp_path / "base.pt"
    save_checkpoint(p, base)
    spec = C.spec_for("true_mu")
    m, extra, fresh = load_for_conditioning(p, "cpu", spec.dim, spec.to_meta())
    assert fresh == ["actor.cond.weight"], fresh
    assert extra["skipped"] == []
    for n, v in base.state_dict().items():
        assert torch.equal(dict(m.state_dict())[n], v), f"{n} did not transfer unchanged"


def test_strict_load_refuses_a_silent_reinitialisation(tmp_path):
    """The tolerant default is what the existing training path needs; the experiment wants the
    opposite, because a skipped tensor and a real effect are indistinguishable in a results table."""
    base = _base(priv_dim=17)
    p = tmp_path / "base.pt"
    save_checkpoint(p, base)
    load_checkpoint(p, "cpu", override={"priv_dim": 21})              # tolerated, as before
    with pytest.raises(ValueError, match="strict load failed"):
        load_checkpoint(p, "cpu", override={"priv_dim": 21}, strict_names=True)


def test_conditional_checkpoint_round_trips_with_its_metadata(tmp_path):
    base = _base()
    cond, spec = _cond_from(base, priv_adapter=C.ADAPTER_ABSENT_OPPONENT)
    with torch.no_grad():
        cond.actor.cond.weight.normal_()
    p = tmp_path / "cond.pt"
    save_checkpoint(p, cond)
    back, _ = load_checkpoint(p, "cpu", allow_conditional=True)
    assert back.meta["cond_dim"] == 1
    assert back.meta["cond"]["source"] == "true_mu"
    assert back.meta["cond"]["lab_oracle"] is True
    assert back.meta["cond"]["offset"] == C.MU_OFFSET and back.meta["cond"]["scale"] == C.MU_SCALE
    assert back.meta["priv_adapter"] == C.ADAPTER_ABSENT_OPPONENT
    assert torch.equal(back.actor.cond.weight, cond.actor.cond.weight)


def test_legacy_loaders_refuse_a_conditional_checkpoint(tmp_path):
    """The viewer, export, watch, evaluate and the ROS node all call `load_checkpoint` without the
    flag, so each of them refuses rather than inventing an input the car cannot produce."""
    base = _base()
    cond, _ = _cond_from(base)
    p = tmp_path / "cond.pt"
    save_checkpoint(p, cond)
    with pytest.raises(ValueError, match="conditional checkpoint"):
        load_checkpoint(p, "cpu")
    load_checkpoint(p, "cpu", allow_conditional=True)


def test_legacy_checkpoint_metadata_is_unchanged(tmp_path):
    """An unconditional model must not start carrying conditioning keys."""
    base = _base()
    p = tmp_path / "base.pt"
    save_checkpoint(p, base)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert "cond_dim" not in ck["meta"] and "cond" not in ck["meta"]
    assert "priv_adapter" not in ck["meta"]
    assert not any(k.startswith("actor.cond") for k in ck["state_dict"])


# ---------------------------------------------------------------- conditioning values
def test_mu_normalization_is_the_declared_affine_map():
    spec = C.spec_for("true_mu")
    mu = torch.tensor([0.734, 1.0, 1.154])
    c = C.mu_to_c(mu, spec)
    assert torch.allclose(c[:, 0], (mu - 1.0) / 0.25)
    assert float(c.abs().max()) < 1.5          # stays in a sane range for a zero-init projection


def test_make_condition_reads_raw_privileged_mu():
    """`mu_index` indexes the env's own privileged vector; the critic adapter must not move it."""
    spec = C.spec_for("true_mu")
    priv = torch.zeros(3, 17)
    priv[:, 8] = torch.tensor([0.8, 1.0, 1.1])                        # solo priv_mu_index == 8
    c = C.make_condition("true_mu", spec, priv, 8)
    assert torch.allclose(c[:, 0], (priv[:, 8] - 1.0) / 0.25)
    assert torch.equal(C.make_condition("zero", spec, priv, 8), torch.zeros(3, 1))


# ---------------------------------------------------------------- critic adapter
def test_absent_opponent_adapter_inserts_zeros_at_the_opponent_block():
    priv = torch.arange(17, dtype=torch.float32)[None].repeat(2, 1)
    out = C.insert_absent_opponent_columns(priv)
    assert out.shape == (2, 21)
    assert torch.equal(out[:, :8], priv[:, :8])                      # dynamic state untouched
    assert float(out[:, 8:12].abs().max()) == 0.0                    # no opponent
    assert torch.equal(out[:, 12:], priv[:, 8:])                     # params + cap shifted, intact


def test_adapter_refuses_any_other_width():
    for w in (16, 18, 21):
        with pytest.raises(ValueError, match="unexplained layout mismatch"):
            C.insert_absent_opponent_columns(torch.zeros(2, w))


def test_critic_with_adapter_preserves_a_21_wide_critic_exactly():
    """A 21-critic fed adapted-17 must equal the same critic fed the 21 it would have seen."""
    torch.manual_seed(3)
    race = ActorCritic(priv_dim=21, **SHAPE)
    solo = ActorCritic(priv_dim=21, priv_adapter=C.ADAPTER_ABSENT_OPPONENT, **SHAPE)
    solo.load_state_dict(race.state_dict())
    scan, pro = _batch()
    priv17 = torch.randn(scan.shape[0], 17)
    priv21 = C.insert_absent_opponent_columns(priv17)
    with torch.no_grad():
        assert torch.equal(solo.critic(scan, pro, priv17), race.critic(scan, pro, priv21))


def test_adapter_is_absent_by_default():
    m = ActorCritic(priv_dim=17, **SHAPE)
    assert m.critic.priv_adapter is None
    scan, pro = _batch()
    with torch.no_grad():
        m.critic(scan, pro, torch.randn(scan.shape[0], 17))          # strict legacy: 17 fits 17
    with pytest.raises(RuntimeError):
        m.critic(scan, pro, torch.randn(scan.shape[0], 21))


# ---------------------------------------------------------------- the real frozen checkpoint
REAL_CKPT = "/home/shchon11/f1sim_runs/ppo_race_0910/ppo_latest.pt"


@pytest.mark.skipif(not __import__("os").path.exists(REAL_CKPT), reason="frozen checkpoint not present")
def test_frozen_checkpoint_migrates_and_keeps_forward_parity(tmp_path):
    """The synthetic model is small and `plain`-stemmed; the pilot runs this one.

    ResNet stem, 6 x 1081 scan, proprio 366, 8D plan, critic priv 21. If anything about the real
    architecture makes the additive projection or the strict migration behave differently, it shows
    up here and not in a 64-beam toy.
    """
    spec = C.spec_for("true_mu")
    base, _ = load_checkpoint(REAL_CKPT, "cpu")
    m, extra, fresh = load_for_conditioning(REAL_CKPT, "cpu", spec.dim, spec.to_meta())
    assert fresh == ["actor.cond.weight"], fresh
    assert extra["skipped"] == []
    assert base.meta["n_beams"] == 1081 and base.meta["proprio_dim"] == 366
    assert base.meta["act_dim"] == 8 and base.meta["priv_dim"] == 21
    assert base.meta["scan_stem"] == "resnet"

    for n, v in base.state_dict().items():
        assert torch.equal(dict(m.state_dict())[n], v), f"{n} did not transfer unchanged"

    torch.manual_seed(0)
    scan = torch.rand(2, base.meta["n_stack"], 1081)
    pro = torch.randn(2, 366)
    base.eval(); m.eval()
    with torch.no_grad():
        a0 = base.actor(scan, pro)
        for value in (0.0, -1.06, 0.62, 50.0):
            a1 = m.actor(scan, pro, torch.full((2, 1), value))
            assert float((a0 - a1).abs().max()) == 0.0, f"parity broke at c={value}"


@pytest.mark.skipif(not __import__("os").path.exists(REAL_CKPT), reason="frozen checkpoint not present")
def test_frozen_checkpoint_critic_adapter_matches_a_race_shaped_input():
    """The 21-wide critic fed adapted solo-17 equals itself fed the 21 it would have seen."""
    m, _ = load_checkpoint(REAL_CKPT, "cpu")
    solo, _, _ = load_for_conditioning(REAL_CKPT, "cpu", 1, C.spec_for("zero").to_meta(),
                                       priv_adapter=C.ADAPTER_ABSENT_OPPONENT)
    torch.manual_seed(0)
    scan = torch.rand(2, m.meta["n_stack"], 1081); pro = torch.randn(2, 366)
    priv17 = torch.randn(2, 17)
    m.eval(); solo.eval()
    with torch.no_grad():
        assert torch.equal(solo.critic(scan, pro, priv17),
                           m.critic(scan, pro, C.insert_absent_opponent_columns(priv17)))


# ---------------------------------------------------------------- metadata validation
def test_unsupported_or_inconsistent_conditioning_metadata_is_rejected():
    with pytest.raises(ValueError, match="unsupported conditioning metadata keys"):
        C.CondSpec.from_meta({"dim": 1, "kind": C.KIND_MU, "source": "true_mu", "lab_oracle": True,
                              "offset": 1.0, "scale": 0.25, "future_field": 3})
    with pytest.raises(ValueError, match="normalization"):
        C.CondSpec.from_meta({"dim": 1, "kind": C.KIND_MU, "source": "zero", "lab_oracle": False,
                              "offset": 0.0, "scale": 1.0})
    with pytest.raises(ValueError, match="lab_oracle"):
        C.CondSpec.from_meta({"dim": 1, "kind": C.KIND_MU, "source": "zero", "lab_oracle": True,
                              "offset": 1.0, "scale": 0.25})
    with pytest.raises(ValueError, match="only dim=1"):
        C.CondSpec.from_meta({"dim": 4, "kind": C.KIND_MU, "source": "zero", "lab_oracle": False,
                              "offset": 1.0, "scale": 0.25})
    with pytest.raises(ValueError, match="must carry no"):
        C.CondSpec.from_meta({"dim": 0, "kind": C.KIND_MU, "source": "zero", "lab_oracle": False,
                              "offset": 1.0, "scale": 0.25})


# ---------------------------------------------------------------- PPO storage semantics
def test_stored_condition_reproduces_the_sampled_logprob_across_a_reset():
    """The ratio PPO clips compares a stored log probability with a recomputed one.

    Both must come from the identical conditioning. `mu` is re-drawn on every episode reset, so
    recomputing `c` from `P` during SGD would score a transition against a friction the car never
    drove on -- and it would do so silently, since the numbers stay plausible. This pins the
    ordering the rollout loop relies on: freeze `c` from the privileged vector belonging to *this*
    observation, before `env.step` advances anything.
    """
    from f1sim.learn.ppo import sample_rollout_action

    base = _base()
    model, spec = _cond_from(base, source="true_mu")
    scan, pro = _batch()
    b = scan.shape[0]

    priv_t = torch.zeros(b, 17)
    priv_t[:, 8] = torch.tensor([0.80, 0.95, 1.05, 1.15])          # the friction at step t
    c_t = C.make_condition("true_mu", spec, priv_t, 8)             # frozen BEFORE the step

    torch.manual_seed(7)
    act, logp_stored = sample_rollout_action(model, scan, pro, c_t)

    # an episode ends and `mu` is re-drawn; the env's privileged vector now says something else
    priv_after_reset = torch.zeros(b, 17)
    priv_after_reset[:, 8] = torch.tensor([1.10, 0.75, 0.90, 1.00])
    c_after = C.make_condition("true_mu", spec, priv_after_reset, 8)
    assert not torch.allclose(c_t, c_after)

    with torch.no_grad():
        logp_same = model.actor.dist(scan, pro, c_t).log_prob(act).sum(1)
        logp_recomputed = model.actor.dist(scan, pro, c_after).log_prob(act).sum(1)

    assert torch.allclose(logp_stored, logp_same, atol=0, rtol=0), "stored condition did not reproduce"
    # At init the projection is zero, so both agree numerically; make the negative case real by
    # giving the conditioning a weight, which is the state every update after the first is in.
    with torch.no_grad():
        model.actor.cond.weight.normal_(std=0.5)
    with torch.no_grad():
        a2 = model.actor.dist(scan, pro, c_t).log_prob(act).sum(1)
        b2 = model.actor.dist(scan, pro, c_after).log_prob(act).sum(1)
    assert not torch.allclose(a2, b2), "conditioning has no effect; the test cannot detect a mix-up"


def test_zero_arm_is_invariant_to_the_privileged_friction():
    """A0 must be unable to see `mu` at all, however the projection later trains."""
    base = _base()
    model, spec = _cond_from(base, source="zero")
    with torch.no_grad():
        model.actor.cond.weight.normal_(std=0.5)
    scan, pro = _batch()
    priv_a = torch.zeros(scan.shape[0], 17); priv_a[:, 8] = 0.8
    priv_b = torch.zeros(scan.shape[0], 17); priv_b[:, 8] = 1.15
    ca = C.make_condition("zero", spec, priv_a, 8)
    cb = C.make_condition("zero", spec, priv_b, 8)
    assert torch.equal(ca, cb) and float(ca.abs().max()) == 0.0
    with torch.no_grad():
        assert torch.equal(model.actor(scan, pro, ca), model.actor(scan, pro, cb))
