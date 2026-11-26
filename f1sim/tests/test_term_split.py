"""(11-26) term_split's pure parts: the GAE is the trainer's, the box-pass window"""
import torch

from f1sim.learn.term_split import gae, near_passes


def test_gae_linear_in_terms():
    torch.manual_seed(0)
    T, B = 20, 5
    D = (torch.rand(T, B) < 0.1).float()
    FM = torch.ones(T, B)
    R1, R2, V1, V2 = torch.randn(T, B), torch.randn(T, B), torch.randn(T, B), torch.randn(T, B)
    l1, l2 = torch.randn(B), torch.randn(B)
    a = gae(R1, V1, l1, D, FM, 0.99, 0.95) + gae(R2, V2, l2, D, FM, 0.99, 0.95)
    b = gae(R1 + R2, V1 + V2, l1 + l2, D, FM, 0.99, 0.95)
    assert torch.allclose(a, b, atol=1e-5)


def test_near_passes_window_and_restart():
    T, B, C = 30, 2, 1
    ds = torch.zeros(T, B, C)
    for t in range(T):
        ds[t, :, 0] = 2.0 - 0.1 * t                 # the box 2 m ahead, passed during decision 19 (ds 0.1 -> 0.0)
    live = torch.ones(T, B, C, dtype=torch.bool)
    pos = torch.zeros(T, B, C, 2)
    D = torch.zeros(T, B)
    D[12, 1] = 1.0                                   # car 1 restarted during decision 12: its window stops there
    p, near = near_passes(ds, live, pos, D, 15)
    assert p[19, 0] and p[:, 0].sum() == 1
    assert near[5:20, 0].all() and not near[4, 0] and not near[20:, 0].any()
    assert near[13:20, 1].all() and not near[:13, 1].any()
