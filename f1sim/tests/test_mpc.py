import math, time
import torch
from f1sim.mpc import PlanSpec, PlanTracker, decode, encode, reference, ACT_DIM

WB, SMAX, VMAX = 0.3302, 0.4189, 8.0


def test_encode_decode_roundtrip():
    spec = PlanSpec(); v = torch.full((4,), 3.0)
    off = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.6, 0.9], [-0.5, -0.2, 0.4], [1.0, -1.0, 0.0]])
    a = encode(off, torch.full((4,), 2.0), torch.full((4,), 5.0), VMAX, spec)
    b, Lp, v0, v1 = decode(a, v, VMAX, torch.full((4,), 9.0), spec)
    assert torch.allclose(v0, torch.full((4,), 2.0), atol=1e-5) and torch.allclose(v1, torch.full((4,), 5.0), atol=1e-5)
    from f1sim.mpc import XI
    g = b[:, 0:1] * XI[None] ** 2 + b[:, 1:2] * XI[None] ** 3 + b[:, 2:3] * XI[None] ** 4
    assert torch.allclose(g * Lp[:, None], off, atol=1e-5)          # the offsets are reproduced at the stations


def test_tracker_straight_and_arc():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = PlanTracker(3, dev, WB, SMAX, VMAX)
    v = torch.tensor([3.0, 3.0, 3.0], device=dev); cap = torch.full((3,), 8.0, device=dev)
    # env 0: straight plan, env 1: constant left curvature 0.5 1/m, env 2: right curvature
    spec = tr.spec; Lp = (spec.horizon_s * 3.0)
    Lp = max(spec.len_min, min(spec.len_max, Lp)); xs = torch.tensor([Lp / 3, 2 * Lp / 3, Lp])
    R = 2.0; arc = R - torch.sqrt((R ** 2 - xs ** 2).clamp_min(0.0))         # circle of radius R tangent at the origin
    off = torch.stack([torch.zeros(3), arc, -arc]).to(dev)
    a = encode(off, torch.full((3,), 3.0, device=dev), torch.full((3,), 3.0, device=dev), VMAX, spec)
    for _ in range(3):
        cmd = tr(a, v, cap)
    steer = cmd[:, 0].cpu()
    assert abs(steer[0]) < 0.02
    d_kin = math.atan(WB / R)                                             # kinematic steer for that radius
    assert 0.5 * d_kin < steer[1] < 1.3 * d_kin, (steer[1], d_kin)
    assert abs(steer[2] + steer[1]) < 0.03                                # mirror symmetric
    assert torch.allclose(cmd[:, 1].cpu(), torch.full((3,), 3.0), atol=0.3)  # speed target followed


def test_tracker_speed_and_runtime():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B = 2048 if dev == "cuda" else 64
    tr = PlanTracker(B, dev, WB, SMAX, VMAX)
    a = torch.zeros(B, ACT_DIM, device=dev); a[:, 3] = 0.5; a[:, 4] = 0.5     # straight, speed target 6 m/s
    v = torch.full((B,), 2.0, device=dev); cap = torch.full((B,), 8.0, device=dev)
    cmd = tr(a, v, cap)
    assert (cmd[:, 1] > 2.3).all() and (cmd[:, 1] < 3.5).all()            # accelerating towards the target, a_max-limited
    if dev == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10): tr(a, v, cap)
    if dev == "cuda": torch.cuda.synchronize()
    dt = (time.time() - t0) / 10
    print(f"tracker {B} envs: {dt * 1e3:.1f} ms/step")
    assert dt < 0.2
