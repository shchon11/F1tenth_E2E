"""Actor2's memory-reading value read-out (`mem_value`) and the update gate's holding start (`mem_keep_s`), 11-12: both
off leave the actor as it was; the read-out teaches the memory and nothing that feeds it or drives; the gate starts
keeping what it holds for the times asked; a checkpoint with the read-out loads and drives the same."""
import math

import torch

from f1sim import plan_steps as PS
from f1sim.learn.plan_policy import Actor2, N_SCANS, STATE_DIM2, load_actor2

BEAMS, M = 360, 32


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N_SCANS, BEAMS, generator=g) * 10.0, torch.randn(n, STATE_DIM2, generator=g)


def _made(**kw):
    torch.manual_seed(0)
    return Actor2(BEAMS, memory=M, mem_front=True, **kw)


def test_both_off_is_the_actor_as_it_was_and_on_adds_only_what_it_says():
    base, val, keep = _made().state_dict(), _made(mem_value=True).state_dict(), _made(mem_keep_s=30.0).state_dict()
    assert set(val) - set(base) == {"v_mem.weight", "v_mem.bias"} and set(keep) == set(base)
    assert all(torch.equal(base[k], val[k]) for k in base)                      # the read-out is made last: nothing else moves
    gate = slice(M, 2 * M)                                                      # GRUCell's rows: reset, update, candidate
    for k in base:
        if k in ("gru.bias_ih", "gru.bias_hh"):
            rest = torch.ones(3 * M, dtype=torch.bool); rest[gate] = False
            assert torch.equal(base[k][rest], keep[k][rest]) and not torch.equal(base[k][gate], keep[k][gate])
        else:
            assert torch.equal(base[k], keep[k])


def test_the_gate_starts_keeping_for_the_times_asked():
    for log in (False, True):
        torch.manual_seed(1)
        a = Actor2(BEAMS, memory=256, mem_front=True, mem_keep_s=30.0, mem_keep_log=log)
        n = a.memory
        z = torch.sigmoid(a.gru.bias_ih[n: 2 * n] + a.gru.bias_hh[n: 2 * n]).detach()   # the share kept per decision with nothing coming in
        t = (1.0 / (1.0 - z)) * PS.DT                                           # [s]
        assert float(t.max()) <= 30.0 + 1e-3 and float(t.min()) >= PS.DT - 1e-6
        if log:
            assert 0.3 < float((t < 1.7).float().mean()) < 0.7                  # half of the values stay short
        else:
            assert 10.0 < float(t.median()) < 20.0 and float((t < 1.0).float().mean()) < 0.1
    torch.manual_seed(1)
    d = Actor2(BEAMS, memory=256, mem_front=True)
    zd = torch.sigmoid(d.gru.bias_ih[256: 512] + d.gru.bias_hh[256: 512]).detach()
    assert float((1.0 / (1.0 - zd)).max()) * PS.DT < 0.3                        # PyTorch's default: every value ~0.2 s
    # what it does: a state put in is still there 5 s later on the same inputs; with the default start it is gone
    sc, st = _inputs(4, 2)
    left = {}
    for name, act in (("keep", a), ("default", d)):
        with torch.no_grad():
            f = act.features(sc, st)
            h0 = torch.zeros(4, 256); h1 = torch.ones(4, 256) * 0.5
            for _ in range(50):
                h0, h1 = act.gru(f, h0), act.gru(f, h1)
            left[name] = float((h1 - h0).abs().mean())
    assert left["keep"] > 0.01 and left["default"] < 1e-4


def test_the_value_readout_teaches_the_memory_and_nothing_that_feeds_it_or_drives():
    a = _made(mem_value=True)
    L, n = 6, 3
    sc, st = _inputs(L * n, 1)
    f = a.features(sc, st).view(L, n, -1)
    h0 = torch.randn(n, M)
    done = torch.zeros(L, n); done[2, 1] = 1.0                                  # car 1 restarts in decision 2
    hs = a.memory_alone(f, h0, done)
    h, ref = h0, []
    for i in range(L):                                                          # the replay that drives
        if i > 0:
            h = h * (1.0 - done[i - 1])[:, None]
        _, h = a.remember(f[i], h)
        ref.append(h)
    assert torch.allclose(hs, torch.cat(ref, 0), atol=1e-6)                     # the same states
    loss = (a.v_mem(hs)[:, 0] - torch.randn(L * n)).pow(2).mean()
    a.zero_grad(); loss.backward()
    assert all(q.grad is not None and float(q.grad.abs().sum()) > 0 for q in list(a.gru.parameters()) + list(a.v_mem.parameters()))
    rest = list(a.enc.parameters()) + list(a.trunk.parameters()) + list(a.pi.parameters()) + list(a.v_est.parameters())
    assert all(q.grad is None or float(q.grad.abs().sum()) == 0 for q in rest)
    # a restart: what car 1 held before it does not reach the decisions after it
    h0b = h0.clone(); h0b[1] += 1.0
    hb = a.memory_alone(f, h0b, done).view(L, n, M); ha = hs.view(L, n, M)
    assert (hb[:3, 1] - ha[:3, 1]).abs().max() > 1e-4 and torch.equal(hb[3:, 1], ha[3:, 1])
    assert torch.equal(hb[:, 0], ha[:, 0]) and torch.equal(hb[:, 2], ha[:, 2])


def test_a_checkpoint_with_the_readout_loads_and_drives_the_same():
    a, b = _made(mem_value=True).eval(), _made().eval()
    sc, st = _inputs(4, 3)
    h = torch.randn(4, M)
    with torch.no_grad():
        ma, mb = a(sc, st, h)[0], b(sc, st, h)[0]
    assert torch.equal(ma, mb)                                                  # driving never reads it
    for act, has in ((a, True), (b, False)):
        ck = {"actor": act.state_dict(), "beams": BEAMS, "args": {"memory": M, "mem_front": True, "std0": 0.6}}
        got = load_actor2(ck, "cpu", check_mount=False)
        assert hasattr(got, "v_mem") == has
        with torch.no_grad():
            assert torch.equal(got(sc, st, h)[0], ma)
