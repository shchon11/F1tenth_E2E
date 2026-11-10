"""Actor2's memory before the trunk (`mem_front`, 11-11): the plan reads the memory through the trunk's nonlinear layers,
what was seen earlier reaches a later plan, and the old layout (memory added after the trunk) is untouched."""
import torch

from f1sim.learn.plan_policy import Actor2, N_SCANS, STATE_DIM2

BEAMS = 360


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N_SCANS, BEAMS, generator=g) * 10.0, torch.randn(n, STATE_DIM2, generator=g)


def test_front_memory_reaches_the_plan_through_the_trunk():
    torch.manual_seed(0)
    a = Actor2(BEAMS, memory=32, mem_front=True).eval()
    assert not hasattr(a, "mem_out")
    assert a.trunk[0].in_features == 256 + STATE_DIM2 + 32 and a.gru.input_size == 256 + STATE_DIM2
    with torch.no_grad():
        a.pi.weight.normal_(0.0, 0.1)                       # the output layer starts near zero: make the plan readable
    sc, st = _inputs(4, 1)
    h = torch.randn(4, 32)
    mean, log_std, v_hat = a(sc, st, h)
    assert mean.shape == (4, a.pi.out_features) and a.h_out.shape == (4, 32)
    # the pieces the training's replay calls give the same plan
    f, h1 = a.remember(a.features(sc, st), h)
    assert torch.allclose(a.heads(f)[0], mean) and torch.allclose(h1, a.h_out)
    # the same inputs with another memory: another plan
    mean0, _, _ = a(sc, st, torch.zeros(4, 32))
    assert (mean - mean0).abs().max() > 1e-3
    # not an offset added to the plan: the memory's effect depends on what is seen now
    sc2, st2 = _inputs(4, 2)
    d1 = a(sc, st, h)[0] - a(sc, st, torch.zeros(4, 32))[0]
    f2 = a.features(sc2, st2)
    with torch.no_grad():
        ha = a.gru(a.features(sc, st), h); hb = a.gru(a.features(sc, st), torch.zeros(4, 32))
        d2 = a.heads(a.trunk(torch.cat([f2, ha], 1)))[0] - a.heads(a.trunk(torch.cat([f2, hb], 1)))[0]
    assert (d1 - d2).abs().max() > 1e-4


def test_what_was_seen_before_reaches_a_later_plan():
    torch.manual_seed(0)
    a = Actor2(BEAMS, memory=32, mem_front=True).eval()
    sc0, st0 = _inputs(2, 3)
    sc0.requires_grad_(True)
    a(sc0, st0, None)
    h = a.h_out
    for k in range(5):                                      # five later decisions on other inputs
        sc, st = _inputs(2, 10 + k)
        mean, _, _ = a(sc, st, h)
        h = a.h_out
    g, = torch.autograd.grad(mean.pow(2).sum(), sc0)
    assert g.abs().max() > 0.0


def test_the_old_layout_is_as_it_was():
    torch.manual_seed(0)
    a = Actor2(BEAMS, memory=32).eval()
    assert a.trunk[0].in_features == 256 + STATE_DIM2 and a.gru.input_size == 256 and hasattr(a, "mem_out")
    sc, st = _inputs(3, 4)
    # its memory starts adding nothing: the plan is the memoryless one whatever the state
    m1, _, _ = a(sc, st, torch.randn(3, 32))
    m0 = a.pi(a.trunk(torch.cat([a.enc(sc), a.norm(st)], 1)))
    assert torch.allclose(m1, m0, atol=1e-6)
