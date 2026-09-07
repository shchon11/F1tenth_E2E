"""The compiled single-car tracker (mpc_fast) must reproduce the training tracker (mpc, torch) exactly."""
import time

import numpy as np
import torch

from f1sim import mpc, mpc_fast

WB, S_MAX, V_MAX = 0.324, 0.42, 8.0


def _rand_case(g, spec):
    a = torch.rand(mpc.ACT_DIM, generator=g, dtype=torch.float64) * 2 - 1
    v = float(torch.rand(1, generator=g) * 7); cap = float(3 + torch.rand(1, generator=g) * 5)
    yr = float((torch.rand(1, generator=g) - 0.5) * 3); dl = float(0.01 + torch.rand(1, generator=g) * 0.05)
    up = (torch.rand(2, generator=g, dtype=torch.float64) - 0.5) * torch.tensor([0.6, 6.0], dtype=torch.float64)
    warm = (torch.rand(spec.N, 2, generator=g, dtype=torch.float64) - 0.5) * torch.tensor([0.6, 6.0], dtype=torch.float64)
    return a, v, cap, yr, dl, up, warm


def test_solve_matches_torch_float64():
    spec = mpc.PlanSpec(); g = torch.Generator().manual_seed(0)
    old = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    try:
        worst = 0.0
        for _ in range(40):
            a, v, cap, yr, dl, up, warm = _rand_case(g, spec)
            u_t, z_t, r_t = mpc.solve(a[None], torch.tensor([v]), torch.tensor([cap]), torch.tensor([yr]), torch.tensor([dl]),
                                      up[None], warm[None], spec, WB, S_MAX, V_MAX)
            u_n, z_n, r_n = mpc_fast.solve(a.numpy(), v, cap, yr, dl, up.numpy(), warm.numpy(), spec, WB, S_MAX, V_MAX)
            worst = max(worst, np.abs(u_t[0].numpy() - u_n).max(), np.abs(z_t[0].numpy() - z_n).max(), np.abs(r_t[0].numpy() - r_n).max())
        assert worst < 1e-6, worst
    finally:
        torch.set_default_dtype(old)


def test_tracker_sequence_matches_torch_tracker():
    """Stateful: warm start and last command carried across steps, torch float32 vs numba float64."""
    spec = mpc.PlanSpec(); g = torch.Generator().manual_seed(1)
    ref = mpc.PlanTracker(1, "cpu", WB, S_MAX, V_MAX, spec); fast = mpc_fast.PlanTrackerFast(WB, S_MAX, V_MAX, spec)
    worst = (0.0, 0.0); v = 0.0
    for t in range(80):
        a = (torch.rand(mpc.ACT_DIM, generator=g) * 2 - 1) * 0.6
        yr = float((torch.rand(1, generator=g) - 0.5) * 2)
        c_t = ref(a[None], torch.tensor([v]), torch.tensor([6.0]), torch.tensor([yr]), delay=0.03)[0]
        c_n = fast(a.numpy(), v, 6.0, yr, delay=0.03)
        worst = (max(worst[0], abs(float(c_t[0]) - c_n[0])), max(worst[1], abs(float(c_t[1]) - c_n[1])))
        v = min(6.0, max(0.0, v + (c_n[1] - v) * 0.3))          # crude plant so speeds move through the range
    assert worst[0] < 1e-4 and worst[1] < 1e-3, worst


def test_tracker_is_fast():
    fast = mpc_fast.PlanTrackerFast(WB, S_MAX, V_MAX)
    a = np.array([0.2, 0.1, -0.1, 0.0, 0.3, 0.4])
    for _ in range(50): fast(a, 3.0, 6.0, 0.4)
    n = 500; t0 = time.perf_counter()
    for _ in range(n): fast(a, 3.0, 6.0, 0.4)
    ms = (time.perf_counter() - t0) / n * 1e3
    print(f"mpc_fast tracker step: {ms:.3f} ms")
    assert ms < 1.0, ms
