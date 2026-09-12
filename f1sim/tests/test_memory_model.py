"""The recurrent actor: warm-start parity, the hidden state's lifecycle, and the Jetson budget.

The claim the whole warm start rests on is that adding memory to the frozen original changes
nothing at step 0. That is checked here against the real `_baselines/frozen_original_48cc698f.pt`
when it is on this machine, and against a small stand-in built the same way when it is not, so the
test is meaningful on a bare checkout and exact on this one.
"""
from __future__ import annotations

import os

import pytest
import torch

from f1sim.learn import budget as budget_mod
from f1sim.learn.memory import Hidden, memory_spec, reset_hidden
from f1sim.learn.model import ActorCritic, load_checkpoint, load_for_memory, save_checkpoint
from f1sim.learn.obs import SCAN_CHANNELS, ScanAugment, decayed_occupancy, scan_edges

BASELINE = budget_mod.BASELINE
SMALL = dict(n_stack=6, n_beams=256, proprio_dim=64, priv_dim=21, act_dim=8,
             scan_deltas=True, temporal_encoder="cnn", scan_stem="resnet")


def _baseline(tmp_path, device="cpu"):
    """(model, meta) for the frozen original, or a small stand-in with the same architecture."""
    if os.path.isfile(BASELINE):
        m, extra = load_checkpoint(BASELINE, device)
        return BASELINE, m, dict(m.meta)
    torch.manual_seed(11)
    m = ActorCritic(**SMALL)
    path = str(tmp_path / "stand_in.pt")
    save_checkpoint(path, m, {"spec": {}})
    return path, m, dict(m.meta)


def _inputs(meta, batch=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, meta["n_stack"], meta["n_beams"], generator=g),
            torch.rand(batch, meta["proprio_dim"], generator=g) * 2 - 1,
            torch.rand(batch, meta["priv_dim"], generator=g) * 2 - 1)


@pytest.mark.parametrize("channels", [None, ["memory"], ["edges"], ["memory", "edges"]])
def test_warm_start_is_bit_identical(tmp_path, channels):
    """Every weight copied, the memory's projection zero, the new scan columns zero: at step 0 the
    warm-started actor's action, aux heads and value are the original's, bit for bit."""
    torch.set_num_threads(1)
    path, base, meta = _baseline(tmp_path)
    base.eval()
    mem, _extra, fresh = load_for_memory(
        path, "cpu", memory_spec(hidden_size=128),
        scan_channels=({"channels": channels} if channels else None))
    mem.eval()
    assert fresh and all(".memory." in f for f in fresh), fresh
    scan, pro, priv = _inputs(meta)
    aug = ScanAugment(channels, meta["n_beams"], scan.shape[0]) if channels else None
    scan_in = scan if aug is None else aug(scan)
    with torch.no_grad():
        a0, lp0, h0 = base.act(scan, pro, deterministic=True)
        a1, lp1, h1 = mem.act(scan_in, pro, deterministic=True, h=None)
        v0 = base.critic(scan, pro, priv)
        v1, _hc = mem.critic.step(scan_in, pro, priv, None)
        g0, o0 = base.actor.forward_all(scan, pro)[1:]
        g1, o1 = mem.actor.step_all(scan_in, pro, None, None)[1:3]
    assert h0 is None and h1.actor.shape == (1, scan.shape[0], 128)
    assert torch.equal(a0, a1), (a0 - a1).abs().max()
    assert torch.equal(lp0, lp1)
    assert torch.equal(v0, v1), (v0 - v1).abs().max()
    assert torch.equal(g0, g1) and torch.equal(o0, o1)
    # ... and it stays identical for as long as the projection is zero, whatever the hidden state.
    with torch.no_grad():
        a2, _lp, _h = mem.act(scan_in, pro, deterministic=True, h=h1)
    assert torch.equal(a0, a2)


def test_a_trained_memory_changes_the_action(tmp_path):
    """The parity above is a property of the zero init, not of a path that does nothing."""
    torch.set_num_threads(1)
    path, base, meta = _baseline(tmp_path)
    mem, _e, _f = load_for_memory(path, "cpu", memory_spec(hidden_size=64))
    mem.eval()
    with torch.no_grad():
        mem.actor.memory.out.weight.normal_(0, 0.3)
    scan, pro, _priv = _inputs(meta)
    with torch.no_grad():
        a_zero, _l, h = mem.act(scan, pro, deterministic=True, h=None)
        a_carried, _l, _h = mem.act(scan, pro, deterministic=True, h=h)
    assert not torch.allclose(a_zero, a_carried), "a carried hidden state must change the action"


def test_hidden_resets_per_env_on_boundaries():
    h = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    out = reset_hidden(h, torch.tensor([False, True, False]))
    assert torch.equal(out[:, 0], h[:, 0]) and torch.equal(out[:, 2], h[:, 2])
    assert float(out[:, 1].abs().max()) == 0.0
    assert reset_hidden(None, torch.tensor([True])) is None
    with pytest.raises(ValueError):
        reset_hidden(h, torch.tensor([True, False]))          # wrong width must not broadcast
    both = Hidden(h.clone(), h.clone()).reset(torch.tensor([True, False, False]))
    assert float(both.actor[:, 0].abs().max()) == 0.0
    assert float(both.critic[:, 0].abs().max()) == 0.0
    assert torch.equal(both.critic[:, 1], h[:, 1])


def test_scan_channels_say_what_they_claim():
    """Edges are the beam-to-beam discontinuity; the occupancy channel remembers and decays."""
    r = torch.ones(1, 1, 16)
    r[0, 0, 8] = 0.2                                        # one close return
    e = scan_edges(r[:, 0])
    assert float(e[0, 0]) == 0.0                            # the first beam has no predecessor
    assert float(e[0, 8]) == pytest.approx(0.8) and float(e[0, 9]) == pytest.approx(0.8)
    assert float(e[0, 4]) == 0.0

    aug = ScanAugment(["memory"], 16, 1, tau_s=2.0, dt=0.025)
    out = aug(r)
    assert out.shape == (1, 2, 16)
    assert torch.equal(out[:, 1], r[:, 0]), "with no history the memory is the scan itself"
    clear = torch.ones(1, 1, 16)
    seen = [float(aug(clear)[0, 1, 8]) for _ in range(40)]   # 1 s of nothing there any more
    assert seen[0] > 0.2 and seen[-1] > seen[0], "the trace must fade"
    assert seen[-1] < 1.0, "and it must still be visible after a second"
    # exp(-1 s / 2 s) of the way back: 0.2 closeness decays to 0.8 * exp(-0.5) of the gap
    assert seen[-1] == pytest.approx(1.0 - 0.8 * float(torch.tensor(-0.5).exp()), abs=2e-3)
    aug.reset()
    assert float(aug(clear)[0, 1, 8]) == 1.0, "a reset forgets"


def test_scan_channel_preview_does_not_advance_the_memory():
    aug = ScanAugment(["memory"], 8, 2, tau_s=2.0)
    close = torch.full((2, 1, 8), 0.3)
    aug(close)
    before = aug.mem.clone()
    aug.preview(torch.ones(2, 1, 8))
    assert torch.equal(aug.mem, before)
    one = aug.preview(torch.ones(1, 1, 8), index=torch.tensor([1]))
    assert one.shape == (1, 2, 8) and torch.equal(aug.mem, before)


def test_scan_augment_refuses_a_batch_it_holds_no_memory_for():
    aug = ScanAugment(["memory"], 8, 2)
    with pytest.raises(ValueError):
        aug(torch.ones(3, 1, 8))
    with pytest.raises(ValueError):
        ScanAugment(["nope"], 8, 1)
    assert list(SCAN_CHANNELS) == ["memory", "edges"]        # the order the conv columns are in


def test_feedforward_entry_points_refuse_a_memory_actor(tmp_path):
    """A recurrent policy run from zeros every step looks exactly like the right one. It is an
    error rather than a silent downgrade."""
    path, _base, meta = _baseline(tmp_path)
    mem, _e, _f = load_for_memory(path, "cpu", memory_spec(hidden_size=32))
    scan, pro, priv = _inputs(meta, batch=2)
    for call in (lambda: mem.actor(scan, pro), lambda: mem.actor.dist(scan, pro),
                 lambda: mem.actor.forward_all(scan, pro), lambda: mem.actor.features(scan, pro),
                 lambda: mem.critic(scan, pro, priv)):
        with pytest.raises(RuntimeError, match="memory"):
            call()


def test_load_for_memory_refuses_a_checkpoint_that_already_has_memory(tmp_path):
    path, _base, _meta = _baseline(tmp_path)
    mem, _e, _f = load_for_memory(path, "cpu", memory_spec(hidden_size=32))
    again = str(tmp_path / "mem.pt")
    save_checkpoint(again, mem, {"spec": {}})
    with pytest.raises(ValueError, match="already carries memory"):
        load_for_memory(again, "cpu", memory_spec(hidden_size=32))
    # ... but the ordinary loader takes it back, memory and all.
    back, _extra = load_checkpoint(again, "cpu")
    assert back.meta["memory"] == mem.meta["memory"]
    assert back.has_memory


@pytest.mark.skipif(not os.path.isfile(BASELINE), reason="the frozen baseline is not on this machine")
@pytest.mark.parametrize("channels", [None, ["memory", "edges"]])
def test_jetson_budget_ratio(channels):
    """CONTRACT.md's rule, asserted: the new actor's forward within 1.5x the frozen original's and
    its parameter count within 2x, both measured CPU / 1 thread / batch 1 / fp32.

    This is a PROXY. The Jetson is not on this machine; what is measured is the same network on the
    same CPU, so the ratio is the transferable part and the absolute milliseconds are not.
    """
    torch.set_num_threads(1)
    model, _e, _f = load_for_memory(BASELINE, "cpu", memory_spec(hidden_size=128),
                                    scan_channels=({"channels": channels} if channels else None))
    r = budget_mod.against_baseline(model, iters=60)
    assert r["forward_ratio"] < 1.5, r
    assert r["step_ratio"] < 1.5, r          # forward + the extra channels, which run on the car too
    assert r["param_ratio_actor"] < 2.0, r
    assert r["param_ratio_total"] < 2.0, r


def test_channels_without_memory_are_a_legitimate_arm(tmp_path):
    """`--scan-channels` alone: no GRU, the same by-name transfer, the same zeroed new columns, and
    still bit-identical at step 0. It is the ablation that says whether the channels help on their
    own, so it has to load."""
    torch.set_num_threads(1)
    path, base, meta = _baseline(tmp_path)
    base.eval()
    m, _extra, fresh = load_for_memory(path, "cpu", None, scan_channels={"channels": ["edges"]})
    m.eval()
    assert fresh == [] and not m.has_memory and m.meta["scan_channels"]["channels"] == ["edges"]
    scan, pro, priv = _inputs(meta)
    aug = ScanAugment(["edges"], meta["n_beams"], scan.shape[0])
    with torch.no_grad():
        a0, _l, h0 = base.act(scan, pro, deterministic=True)
        a1, _l, h1 = m.act(aug(scan), pro, deterministic=True)
        v0, v1 = base.critic(scan, pro, priv), m.critic(aug(scan), pro, priv)
    assert h0 is None and h1 is None
    assert torch.equal(a0, a1) and torch.equal(v0, v1)
    with pytest.raises(ValueError, match="extra steps"):
        load_for_memory(path, "cpu", None, None)
