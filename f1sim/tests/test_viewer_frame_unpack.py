"""The viewer's per-car frame pack is unpacked by STATE_DIM, not by a literal column list."""
import numpy as np
import torch

from f1sim.dynamics import STATE_DIM
from f1sim.viewer.native import unpack_frame


def test_frame_fields_follow_state_dim():
    n = 3
    state = torch.arange(n * STATE_DIM, dtype=torch.float32).reshape(n, STATE_DIM) * 0.01
    roll = torch.full((n, 1), 0.11); pitch = torch.full((n, 1), 0.22)
    lap = torch.full((n, 1), 3.0); coll = torch.full((n, 1), 1.0)
    s = torch.full((n, 1), 44.0); wall = torch.full((n, 1), 0.55)
    rear = torch.tensor([[1., 2., 3., 4.]]).repeat(n, 1); length = torch.full((n, 1), 0.5)
    pack = torch.cat([state, roll, pitch, lap, coll, s, wall, rear, length], 1).numpy()
    fr = unpack_frame(pack)
    assert np.allclose(fr["roll"], 0.11) and np.allclose(fr["pitch"], 0.22)
    assert np.allclose(fr["lap"], 3.0) and np.allclose(fr["coll"], 1.0)
    assert np.allclose(fr["s"], 44.0) and np.allclose(fr["wall"], 0.55)
    assert fr["rear"].shape == (n, 4) and np.allclose(fr["rear"][0], [1, 2, 3, 4])
    assert np.allclose(fr["len"], 0.5)
    assert np.allclose(fr["steer"], state[:, 6].numpy())
    # the bug this guards against: the wheel speed (state column 7) must never be read as roll
    assert not np.allclose(fr["roll"], state[:, 7].numpy())
