"""Recurrent PPO: the loss with memory off is the loss it was, and BPTT actually learns over time.

The oracle in `tests/data/ppo_loss_oracle.json` was recorded by
`scripts/make_ppo_loss_oracle.py` against the code as of ae1f4df, before any of the memory work
existed. It pins the model's forward AND the loss arithmetic, bit for bit, on one fixed batch.
"""
from __future__ import annotations

import copy
import json
import os

import pytest
import torch

from f1sim.learn.memory import Hidden, memory_spec
from f1sim.learn.model import ActorCritic, load_for_memory, save_checkpoint
from f1sim.learn.ppo import PPOHyper, minibatch_losses

ORACLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ppo_loss_oracle.json")


def _oracle():
    with open(ORACLE) as f:
        return json.load(f)


def _fixture(rec, memory=None, scan_channels=None):
    """The oracle's model and batch, rebuilt from its recorded seed and config."""
    cfg = dict(rec["config"])
    torch.manual_seed(rec["seed"])
    model = ActorCritic(**cfg, memory=memory, scan_channels=scan_channels)
    g = torch.Generator().manual_seed(rec["seed"] + 1)
    r = lambda *s: torch.rand(*s, generator=g)
    b = rec["batch"]
    batch = dict(scan=r(b, cfg["n_stack"], cfg["n_beams"]), pro=r(b, cfg["proprio_dim"]) * 2 - 1,
                 priv=r(b, cfg["priv_dim"]) * 2 - 1, act=r(b, cfg["act_dim"]) * 2 - 1,
                 logp=r(b) * 2 - 1, adv=r(b) * 2 - 1, ret=r(b) * 2 - 1, val=r(b) * 2 - 1,
                 mask=(r(b) > 0.25).float())
    with torch.no_grad():
        d = model.actor.step_dist(batch["scan"], batch["pro"], None)[0]
        batch["logp"] = (d.log_prob(batch["act"]).sum(1) + (r(b) * 0.6 - 0.3)).float()
    ref = copy.deepcopy(model.actor).eval()
    with torch.no_grad():
        ref.mu.weight.mul_(1.03); ref.log_std.add_(0.05)
    for p in ref.parameters():
        p.requires_grad_(False)
    return model, ref, batch


def _losses(model, ref, batch, rec, **kw):
    h = rec["hyper"]
    hyper = PPOHyper(clip=h["clip"], vf=h["vf"], ent=h["ent"], kl_coef=h["kl_coef"],
                     aux_grip=h["aux_grip"], aux_opp=h["aux_opp"],
                     priv_mu_index=h["priv_mu_index"], aux_opp_range_m=h["aux_opp_range_m"],
                     m_gt_1=h["m_gt_1"])
    return minibatch_losses(model, ref, scan=batch["scan"], pro=batch["pro"], priv=batch["priv"],
                            act=batch["act"], logp_old=batch["logp"], adv=batch["adv"],
                            ret=batch["ret"], val_old=batch["val"], w=batch["mask"],
                            hyper=hyper, **kw)


def test_memory_off_loss_is_bit_identical_to_the_frozen_oracle():
    """`--memory off` is the run it was: same loss, same gradient norm, to the last bit."""
    torch.set_num_threads(1)
    rec = _oracle()
    model, ref, batch = _fixture(rec)
    out = _losses(model, ref, batch, rec)
    for key, want in rec["values"].items():
        got = float(out[key].detach())
        assert got == want, f"{key}: {got!r} != frozen {want!r}"
    assert out["hidden"] is None, "a feedforward model must report no hidden state"
    model.zero_grad()
    out["loss"].backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    assert float(gn) == rec["grad_norm"]


def test_memory_warm_start_reproduces_the_same_loss_on_the_same_batch(tmp_path):
    """A warm-started memory model scores the fixed batch as the feedforward original does.

    Same claim as the model-level parity test, made where it matters for training: the first
    update of a warm-started run starts on the original's loss surface, not near it. Run through
    the RECURRENT path -- sequence blocks, a start hidden state, the BPTT mask -- so what is
    compared is the whole update, not just the network.

    Not bit-exact, and cannot be: the sequence path applies the output heads to a block assembled
    from the steps, so the GEMM is blocked differently from one call over the whole batch. The
    tolerance is float32 rounding (~1e-7 relative), not slack for a real difference.
    """
    torch.set_num_threads(1)
    rec = _oracle()
    model, ref, batch = _fixture(rec)
    path = str(tmp_path / "ff.pt")
    save_checkpoint(path, model, {"spec": {}})
    mem, _extra, fresh = load_for_memory(path, "cpu", memory_spec(hidden_size=64))
    assert all(".memory." in f for f in fresh), fresh
    mem_ref = copy.deepcopy(mem.actor).eval()
    with torch.no_grad():                     # the oracle's reference, on the same weights
        mem_ref.mu.weight.mul_(1.03); mem_ref.log_std.add_(0.05)
    for p_ in mem_ref.parameters():
        p_.requires_grad_(False)
    T, m = 4, rec["batch"] // 4
    seq = lambda t: t.reshape(T, m, *t.shape[1:])
    h = rec["hyper"]
    out = minibatch_losses(
        mem, mem_ref, scan=seq(batch["scan"]), pro=seq(batch["pro"]), priv=seq(batch["priv"]),
        act=seq(batch["act"]), logp_old=batch["logp"], adv=batch["adv"], ret=batch["ret"],
        val_old=batch["val"], w=batch["mask"],
        hyper=PPOHyper(clip=h["clip"], vf=h["vf"], ent=h["ent"], kl_coef=h["kl_coef"],
                       aux_grip=h["aux_grip"], aux_opp=h["aux_opp"],
                       priv_mu_index=h["priv_mu_index"], aux_opp_range_m=h["aux_opp_range_m"],
                       m_gt_1=h["m_gt_1"]),
        sequence=(mem.initial_hidden(m), torch.ones(T, m)))
    for key, want in rec["values"].items():
        got = float(out[key].detach())
        assert got == pytest.approx(want, rel=1e-5, abs=1e-6), f"{key}: {got!r} vs original {want!r}"
    assert out["hidden"].actor.shape == (1, m, 64)
    assert out["hidden"].critic.shape == (1, m, 64)


def test_truncated_bptt_learns_a_task_the_feedforward_actor_cannot():
    """The signal is in the FIRST step of a sequence and the target only at the LAST.

    A feedforward actor sees neither step in the other's company and can do no better than the
    mean; a recurrent one has to carry the bit across the sequence to fit it. This is the gradient
    -through-time check: if the hidden state were detached between steps, or the sequence replayed
    from zeros, the recurrent actor would score like the feedforward one.
    """
    torch.set_num_threads(1)
    torch.manual_seed(0)
    T, m, beams = 8, 32, 32
    cfg = dict(n_stack=1, n_beams=beams, proprio_dim=4, priv_dim=2, act_dim=2, scan_stem="plain")

    def batch(gen):
        bit = (torch.rand(m, generator=gen) > 0.5).float()
        scan = torch.full((T, m, 1, beams), 0.5)
        scan[0, :, 0, :] = bit[:, None]                 # the signal, first step only
        pro = torch.zeros(T, m, 4)
        pro[-1, :, 0] = 1.0                             # "answer now", last step only
        target = torch.where(bit > 0.5, 0.6, -0.6)
        return scan, pro, target

    def fit(memory):
        torch.manual_seed(1)
        model = ActorCritic(**cfg, memory=memory)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        gen = torch.Generator().manual_seed(2)
        for _ in range(500):
            scan, pro, target = batch(gen)
            h = model.initial_hidden(m)
            mus = []
            for t in range(T):
                mu, ha = model.actor.step(scan[t], pro[t], None, None if h is None else h.actor)
                h = Hidden(ha, None)
                mus.append(mu)
            loss = ((mus[-1][:, 0] - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            scan, pro, target = batch(torch.Generator().manual_seed(99))
            h = model.initial_hidden(m)
            for t in range(T):
                mu, ha = model.actor.step(scan[t], pro[t], None, None if h is None else h.actor)
                h = Hidden(ha, None)
        return float(((mu[:, 0] - target) ** 2).mean().detach())

    ff = fit(None)
    rec = fit(memory_spec(hidden_size=32, critic="none"))
    # The feedforward actor cannot beat predicting the mean of +-0.6, i.e. 0.36. The recurrent one
    # spends its first ~250 steps growing the zero-initialised projection (which is what makes the
    # warm start exact) and then fits the task outright.
    assert ff > 0.25, f"the feedforward actor should not be able to fit this: mse {ff}"
    assert rec < 0.02, f"the recurrent actor should carry the bit across the sequence: mse {rec}"
    assert rec < ff / 10


def test_hidden_state_resets_on_episode_boundaries():
    """`keep = 0` at a step makes that step start from zeros -- the same state a fresh episode has."""
    torch.set_num_threads(1)
    torch.manual_seed(4)
    T, m = 5, 3
    cfg = dict(n_stack=2, n_beams=64, proprio_dim=6, priv_dim=2, act_dim=2, scan_stem="plain")
    model = ActorCritic(**cfg, memory=memory_spec(hidden_size=16)).eval()
    with torch.no_grad():                    # a trained-looking memory: the projection must matter
        model.actor.memory.out.weight.normal_(0, 0.4)
    g = torch.Generator().manual_seed(5)
    scan = torch.rand(T, m, 2, 64, generator=g)
    pro = torch.rand(T, m, 6, generator=g)
    priv = torch.rand(T, m, 2, generator=g)
    act = torch.rand(T, m, 2, generator=g) * 2 - 1

    def run(keep, h0):
        with torch.no_grad():
            return model.evaluate_sequence(scan, pro, priv, act, None, h0, keep)[0].reshape(T, m)

    hot = model.initial_hidden(m)
    hot = Hidden(torch.randn(1, m, 16, generator=g), torch.randn(1, m, 16, generator=g))
    keep = torch.ones(T, m)
    keep[0, 1] = 0.0                         # env 1 starts the chunk on a fresh episode
    with_reset = run(keep, hot)
    from_zero = run(torch.ones(T, m), model.initial_hidden(m))
    carried = run(torch.ones(T, m), hot)
    assert torch.allclose(with_reset[0, 1], from_zero[0, 1]), "a reset row must start from zeros"
    assert not torch.allclose(with_reset[0, 0], from_zero[0, 0]), "an un-reset row must carry state"
    assert torch.allclose(with_reset[0, 0], carried[0, 0])


def test_a_feedforward_critic_is_allowed_and_carries_nothing():
    """`--memory-critic none`: the actor remembers, the critic does not, and the update still runs
    (the one place `buf_h0_critic is None` has to be handled)."""
    torch.set_num_threads(1)
    torch.manual_seed(6)
    T, m = 3, 4
    cfg = dict(n_stack=2, n_beams=48, proprio_dim=5, priv_dim=3, act_dim=2, scan_stem="plain")
    model = ActorCritic(**cfg, memory=memory_spec(hidden_size=8, critic="none"))
    assert model.actor.has_memory and not model.critic.has_memory
    h = model.initial_hidden(m)
    assert h.actor.shape == (1, m, 8) and h.critic is None
    g = torch.Generator().manual_seed(7)
    out = model.evaluate_sequence(torch.rand(T, m, 2, 48, generator=g), torch.rand(T, m, 5, generator=g),
                                  torch.rand(T, m, 3, generator=g),
                                  torch.rand(T, m, 2, generator=g) * 2 - 1, None, h, torch.ones(T, m))
    assert out[0].shape == (T * m,) and out[-1].critic is None and out[-1].actor.shape == (1, m, 8)


@pytest.mark.parametrize("argv,message", [
    (["--memory", "gru", "--controller", "fixed_low", "--estimator", "x"], "validated"),
    (["--memory", "gru", "--minibatch", "8", "--horizon", "32"], "whole env chunks"),
    (["--scan-channels", "elbow"], "known channels"),
    (["--memory", "gru", "--cond", "zero", "--fresh-opt"], "not a supported combination"),
])
def test_the_cli_refuses_combinations_nothing_has_measured(monkeypatch, argv, message):
    """Each of these would produce a run that looks like the one it is named after and is not."""
    import sys
    from f1sim.learn import ppo
    monkeypatch.setattr(sys, "argv", ["ppo", "--name", "t", "--device", "cpu"] + argv)
    with pytest.raises(SystemExit) as e:
        ppo.main()
    assert message in str(e.value), e.value
