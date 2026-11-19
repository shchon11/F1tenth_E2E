"""(11-19) `Actor2.box_memory`: a second GRU for the side task beside the policy's own memory (plan_rl --box-memory)."""
import torch

from f1sim.learn.plan_policy import Actor2, STATE_DIM2, N_SCANS

BEAMS, M, MB, AUX = 1081, 16, 8, 12


def _inputs(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N_SCANS, BEAMS, generator=g) * 8.0, torch.randn(n, STATE_DIM2, generator=g)


def _from_one_memory(a: Actor2) -> Actor2:
    """as plan_rl --init does: the trunk's new columns zero, the new weights fresh"""
    torch.manual_seed(1)
    b = Actor2(BEAMS, memory=M, mem_front=True, aux_dim=AUX, mem_value=True, box_memory=MB)
    sd = dict(a.state_dict())
    w = sd["trunk.0.weight"]
    sd["trunk.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], b.trunk[0].in_features - w.shape[1])], 1)
    for k, v in b.state_dict().items():
        if k not in sd or (k in ("aux.weight", "aux.bias") and sd[k].shape != v.shape):
            sd[k] = v
    b.load_state_dict(sd)
    return b


def test_same_driving_from_a_checkpoint_without_it():
    torch.manual_seed(0)
    a = Actor2(BEAMS, memory=M, mem_front=True, aux_dim=AUX, mem_value=True)
    b = _from_one_memory(a)
    assert b.memory == M + MB and b.memory_main == M
    sc, st = _inputs()
    h = torch.randn(4, M)
    with torch.no_grad():
        ma, la, va = a(sc, st, h)
        mb, lb, vb = b(sc, st, torch.cat([h, torch.randn(4, MB)], 1))     # whatever the box memory holds
    assert torch.allclose(ma, mb, atol=1e-6) and torch.allclose(va, vb, atol=1e-6)
    assert torch.allclose(b.h_out[:, :M], a.h_out, atol=1e-6)


def test_the_side_task_reads_and_teaches_only_the_box_memory():
    torch.manual_seed(0)
    b = Actor2(BEAMS, memory=M, mem_front=True, aux_dim=AUX, mem_value=True, box_memory=MB)
    sc, st = _inputs()
    f = b.features(sc, st)
    hs = b.memory_alone(torch.stack([f, f, f]), torch.zeros(4, b.memory), torch.zeros(3, 4))
    assert hs.shape == (12, M + MB)
    b.aux_read(hs).pow(2).sum().backward(retain_graph=True)
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in b.gru.parameters())
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in b.gru_box.parameters())
    b.zero_grad()
    b.vmem_read(hs).sum().backward()                                    # the value read-out: the policy's own memory only
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in b.gru_box.parameters())


def test_one_memory_actor_unchanged():
    torch.manual_seed(0)
    a = Actor2(BEAMS, memory=M, mem_front=True, aux_dim=AUX)
    assert a.memory == a.memory_main == M and a.box_memory == 0 and not hasattr(a, "gru_box")
    assert a.aux.in_features == M
