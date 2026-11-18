"""(11-19) --mem-prev: the memory's update replays the previous rollout's last decisions before this one (`mem_prefix`), so
the loss on this rollout reaches what the memory stored then."""
import torch

from f1sim.learn.plan_rl import mem_prefix

K, B, F, H = 6, 4, 5, 3


def _setup(seed=0):
    torch.manual_seed(seed)
    gru = torch.nn.GRUCell(F, H)
    pF = torch.randn(K, B, F)
    pHM = torch.randn(K, B, H)
    pD = torch.zeros(K, B)
    h_now = torch.randn(B, H)
    return gru, pF, pHM, pD, h_now


def _manual(gru, pF, pHM, pD, b):
    h = pHM[0, b:b + 1]
    for i in range(K):
        if i > 0 and pD[i - 1, b] > 0:
            h = pHM[i, b:b + 1]
        h = gru(pF[i, b:b + 1], h)
    return h[0]


def test_replay_without_restarts_is_the_gru_over_the_prefix():
    gru, pF, pHM, pD, h_now = _setup()
    cars = torch.tensor([2, 0])
    out = mem_prefix(gru, pF, pHM, pD, cars, h_now[cars])
    for j, b in enumerate(cars.tolist()):
        assert torch.allclose(out[j], _manual(gru, pF, pHM, pD, b), atol=1e-6)


def test_restart_inside_and_after_the_prefix():
    gru, pF, pHM, pD, h_now = _setup(1)
    pD[2, 1] = 1.0                       # car 1 restarted after decision 2: decision 3 drove from pHM[3]
    pD[K - 1, 3] = 1.0                   # car 3 restarted after the previous rollout's last decision
    cars = torch.arange(B)
    out = mem_prefix(gru, pF, pHM, pD, cars, h_now)
    assert torch.allclose(out[1], _manual(gru, pF, pHM, pD, 1), atol=1e-6)
    assert torch.allclose(out[3], h_now[3])                            # this rollout began from the state it was driven from
    # what car 1 saw before its restart cannot matter
    pF2 = pF.clone(); pF2[:3, 1] += 1.0
    assert torch.allclose(mem_prefix(gru, pF2, pHM, pD, cars, h_now)[1], out[1], atol=1e-6)


def test_the_loss_now_reaches_the_memory_through_the_prefix():
    gru, pF, pHM, pD, h_now = _setup(2)
    cars = torch.arange(B)
    out = mem_prefix(gru, pF, pHM, pD, cars, h_now)
    out.pow(2).sum().backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in gru.parameters())
    # what was seen early in the prefix still moves the state the rollout begins with
    pF2 = pF.clone(); pF2[0] += 1.0
    with torch.no_grad():
        assert float((mem_prefix(gru, pF2, pHM, pD, cars, h_now) - out).abs().max()) > 1e-4
