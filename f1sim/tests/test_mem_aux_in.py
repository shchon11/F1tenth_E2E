"""(11-20) `Actor2.aux_in`: the trunk also reads the box memory's own answer (plan_rl --mem-aux-in), detached."""
import pytest
import torch

from f1sim.learn.plan_policy import Actor2, STATE_DIM2, N_SCANS

BEAMS, M, MB, AUX = 1081, 16, 8, 12


def _inputs(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N_SCANS, BEAMS, generator=g) * 8.0, torch.randn(n, STATE_DIM2, generator=g)


def _actor(aux_in=True):
    torch.manual_seed(0)
    return Actor2(BEAMS, memory=M, mem_front=True, aux_dim=AUX, mem_value=True, box_memory=MB, aux_in=aux_in)


def test_the_trunk_reads_twelve_more_values():
    a, b = _actor(False), _actor(True)
    assert b.trunk[0].in_features == a.trunk[0].in_features + 12
    sc, st = _inputs()
    means, log_std, v = b(sc, st, torch.randn(4, M + MB))
    assert means.shape[0] == 4 and torch.isfinite(means).all()


def test_the_answer_is_the_read_out_absent_boxes_zero():
    b = _actor()
    hs = torch.randn(5, M + MB)
    with torch.no_grad():
        raw = b.aux_read(hs).view(5, 3, 4)
        feed = b.aux_feed(hs).view(5, 3, 4)
    p = torch.sigmoid(raw[..., 0])
    assert torch.allclose(feed[..., 0], p, atol=1e-6)
    assert torch.allclose(feed[..., 1:], raw[..., 1:] * p[..., None], atol=1e-6)
    with torch.no_grad():
        b.aux.bias[0::4] = -50.0                                      # every box surely absent
        assert float(b.aux_feed(hs).abs().max()) < 1e-6


def test_driving_cannot_bend_the_answer():
    b = _actor()
    sc, st = _inputs()
    f = b.features(sc, st)
    hs = b.mem_cell(f, torch.randn(4, b.memory))                   # a zero state would leave the recurrent weights' gradient 0
    b.mem_trunk(f, hs).pow(2).sum().backward()
    assert b.aux.weight.grad is None and b.aux.bias.grad is None      # the read-out: only the side task teaches it
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in b.gru_box.parameters())   # its raw state: as before


def test_rollout_and_training_use_the_same_trunk():
    b = _actor()
    sc, st = _inputs()
    h = torch.randn(4, M + MB)
    with torch.no_grad():
        m1, _, _ = b(sc, st, h)
        f = b.features(sc, st)
        m2, _, _ = b.heads(b.mem_trunk(f, b.mem_cell(f, h)))
    assert torch.allclose(m1, m2, atol=1e-6)


def test_needs_the_box_memory_and_its_read_out():
    with pytest.raises(ValueError):
        Actor2(BEAMS, memory=M, mem_front=True, aux_dim=AUX, aux_in=True)            # no box memory
    with pytest.raises(ValueError):
        Actor2(BEAMS, memory=M, mem_front=True, box_memory=MB, aux_in=True)          # no read-out
