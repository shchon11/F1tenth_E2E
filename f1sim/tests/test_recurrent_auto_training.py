"""CPU contracts for the D3 speed stage and independent recurrent KL targets."""
from __future__ import annotations

import copy

import pytest
import torch

from f1sim.learn.memory import memory_spec
from f1sim.learn.model import ActorCritic, load_checkpoint, load_for_memory, save_checkpoint
from f1sim.learn.policy_adaptation import (RecurrentReference, SpeedRowFreeze,
                                           require_exact_actor, resolve_reference,
                                           stage_lr_kl, stage_schedule,
                                           verified_reference_stream)
from f1sim.learn.ppo import PPOHyper, minibatch_losses


CFG = dict(n_stack=1, n_beams=32, proprio_dim=6, priv_dim=3, act_dim=8)


def _model():
    return ActorCritic(**CFG, memory=memory_spec(hidden_size=16))


def test_zero_initialization_parity_and_exact_speed_row_freeze(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    baseline = ActorCritic(**CFG)
    checkpoint = tmp_path / "original.pt"
    save_checkpoint(checkpoint, baseline)
    model, _, _ = load_for_memory(checkpoint, "cpu", memory_spec(hidden_size=16))
    scan = torch.rand(2, 1, 32)
    pro = torch.rand(2, 6)
    assert torch.equal(baseline.actor.step(scan, pro)[0], model.actor.step(scan, pro)[0])

    before = {name: p.detach().clone() for name, p in model.actor.named_parameters()}
    freeze = SpeedRowFreeze(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        dist, _ = model.actor.step_dist(scan, pro)
        value, _ = model.critic.step(scan, pro, torch.rand(2, 3))
        loss = -dist.log_prob(torch.ones(2, 8) * 0.5).sum() + value.square().sum()
        optimizer.zero_grad()
        loss.backward()
        freeze.before_step()
        optimizer.step()
        freeze.after_step()
    after = dict(model.actor.named_parameters())
    for name, old in before.items():
        if name in ("mu.weight", "mu.bias", "log_std"):
            assert torch.equal(after[name][:6], old[:6]), name
            assert not torch.equal(after[name][6:], old[6:]), name
        else:
            assert torch.equal(after[name], old), name


def test_recurrent_reference_owns_hidden_and_resets_each_car():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    actor = _model().actor
    with torch.no_grad():
        actor.memory.out.weight.normal_(0, 0.2)
    frozen = copy.deepcopy(actor)
    collector = RecurrentReference(frozen, 2, "cpu")
    scan = torch.rand(2, 1, 32)
    pro = torch.rand(2, 6)
    first, _ = collector.step(scan, pro)
    carried, _ = collector.step(scan, pro)
    assert not torch.allclose(first, carried)
    collector.reset(torch.tensor([True, False]))
    reset, _ = collector.step(scan, pro)
    assert torch.allclose(reset[0], first[0], atol=1e-7)
    assert not torch.allclose(reset[1], first[1])
    assert all(not p.requires_grad for p in frozen.parameters())
    assert actor.initial_hidden(2) is not collector.hidden


def test_stored_reference_targets_drive_curvature_kl():
    torch.set_num_threads(1)
    model = _model()
    T, m = 2, 2
    scan = torch.rand(T, m, 1, 32)
    pro = torch.rand(T, m, 6)
    priv = torch.rand(T, m, 3)
    act = torch.zeros(T, m, 8)
    with torch.no_grad():
        d = model.evaluate_sequence(scan, pro, priv, act, h=model.initial_hidden(m),
                                    keep=torch.ones(T, m))[3]
    common = dict(scan=scan, pro=pro, priv=priv, act=act,
                  logp_old=torch.zeros(T * m), adv=torch.zeros(T * m),
                  ret=torch.zeros(T * m), val_old=torch.zeros(T * m),
                  w=torch.ones(T * m), hyper=PPOHyper(.2, .5, 0., 1.),
                  sequence=(model.initial_hidden(m), torch.ones(T, m)), kl_scope="curvature")
    center = d.mean.reshape(T, m, 8).detach()
    std = d.stddev.reshape(T, m, 8).detach()
    same = minibatch_losses(model, None, **common, reference_params=(center, std))
    shifted = center.clone()
    shifted[..., 6:] += .5
    speed_shift = minibatch_losses(model, None, **common, reference_params=(shifted, std))
    shifted[..., :6] += .5
    curve_shift = minibatch_losses(model, None, **common, reference_params=(shifted, std))
    assert float(same["kl_ref"].detach()) == pytest.approx(0, abs=1e-6)
    assert torch.allclose(speed_shift["kl_ref"], same["kl_ref"])
    assert float(curve_shift["kl_ref"].detach()) > .1


def test_original_reference_identity_survives_resume_and_rejects_changed_file(tmp_path):
    original = tmp_path / "original.pt"
    save_checkpoint(original, _model(), {"run": "D3", "update": 500})
    identity = resolve_reference(str(original), None)
    assert identity["sha256"] and identity["meta"]["act_dim"] == 8
    assert resolve_reference("", identity) == identity
    loaded, extra = load_checkpoint(verified_reference_stream(identity), "cpu", strict_names=True)
    assert loaded.actor.mu.out_features == 8 and extra["run"] == "D3"
    stream = verified_reference_stream(identity)
    save_checkpoint(original, _model(), {"run": "replacement"})
    assert torch.load(stream, map_location="cpu", weights_only=True)["extra"]["run"] == "D3"
    with pytest.raises(ValueError, match="hash changed"):
        verified_reference_stream(identity)
    with pytest.raises(ValueError, match="hash changed"):
        resolve_reference("", identity)


def test_adaptive_actor_loader_refuses_missing_or_migrated_tensor():
    model = _model()
    checkpoint = {"state_dict": {k: v.clone() for k, v in model.state_dict().items()}}
    require_exact_actor(model, checkpoint)
    checkpoint["state_dict"].pop("actor.mu.bias")
    with pytest.raises(RuntimeError, match="every actor tensor"):
        require_exact_actor(model, checkpoint)
    checkpoint["state_dict"]["actor.mu.bias"] = model.actor.mu.bias.detach().clone()
    checkpoint["state_dict"]["actor.mu.weight"] = model.actor.mu.weight[:7].detach().clone()
    with pytest.raises(RuntimeError, match="every actor tensor"):
        require_exact_actor(model, checkpoint)
    checkpoint["state_dict"]["actor.mu.weight"] = model.actor.mu.weight.detach().clone() + .01
    with pytest.raises(RuntimeError, match="every actor tensor"):
        require_exact_actor(model, checkpoint)


def test_adaptive_stage_resume_uses_remaining_budget_and_sealed_schedule():
    params = {"lr": 3e-4, "lr_end": 1e-4, "kl_coef": .3, "kl_decay": 8,
              "cap_steps": 8, "cap0": 4., "cap1": 8., "horizon": 2, "envs": 4}
    sealed, completed = stage_schedule("speed", 8, 4, 100, params)
    assert completed == 0
    assert stage_lr_kl(sealed, completed) == (3e-4, .3)
    resumed, completed = stage_schedule("speed", 8, 4, 100, params, sealed, 104, 4)
    assert resumed == sealed and completed == 4
    assert stage_lr_kl(resumed, completed) == pytest.approx((2e-4, .15))
    done, completed = stage_schedule("speed", 8, 4, 100, params, sealed, 108, 8)
    assert done == sealed and completed == done["total_steps"]
    full, completed = stage_schedule("full", 12, 4, 108, params)
    assert full["origin_total_steps"] == 108 and completed == 0
    with pytest.raises(ValueError, match="schedule"):
        stage_schedule("speed", 12, 4, 100, params, sealed, 104, 4)
    with pytest.raises(ValueError, match="schedule"):
        stage_schedule("speed", 8, 4, 100, {**params, "lr": .001}, sealed, 104, 4)
    with pytest.raises(ValueError, match="schedule"):
        stage_schedule("speed", 8, 4, 100, {**params, "horizon": 4, "envs": 2}, sealed, 104, 4)
    with pytest.raises(ValueError, match="stage_steps"):
        stage_schedule("speed", 8, 4, 100, params, sealed, 104, 0)
    with pytest.raises(ValueError, match="outside"):
        stage_schedule("speed", 8, 4, 100, params, sealed, 112, 12)
