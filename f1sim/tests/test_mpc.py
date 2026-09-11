import math, time
import torch
from f1sim.mpc import PlanSpec, PlanTracker, decode, encode, reference, ACT_DIM, N_KNOTS

WB, SMAX, VMAX = 0.3302, 0.4189, 8.0


def test_encode_decode_roundtrip():
    spec = PlanSpec(); v = torch.full((3,), 3.0)
    # one row per case, N_KNOTS wide: the knot count is a constant of the plan action space, not 4
    kap = torch.stack([torch.zeros(N_KNOTS),
                       torch.full((N_KNOTS,), 0.5),
                       torch.linspace(-1.2, 0.8, N_KNOTS)])
    a = encode(kap, torch.full((3,), 2.0), torch.full((3,), 5.0), VMAX, spec)
    k, Lp, v0, v1 = decode(a, v, VMAX, torch.full((3,), 9.0), spec)
    assert torch.allclose(k, kap, atol=1e-5) and torch.allclose(v0, torch.full((3,), 2.0), atol=1e-5) and torch.allclose(v1, torch.full((3,), 5.0), atol=1e-5)
    from f1sim.mpc import path_points
    x, y, psi, s = path_points(k, Lp)
    assert torch.allclose(psi[1, -1], torch.tensor(0.5 * Lp[1]), atol=1e-4)      # constant curvature: heading = kappa * s
    assert abs(float(x[1, -1] - math.sin(0.5 * Lp[1]) / 0.5)) < 0.02             # ... on a circle of radius 2 m


def test_tracker_straight_and_arc():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = PlanTracker(3, dev, WB, SMAX, VMAX)
    v = torch.tensor([3.0, 3.0, 3.0], device=dev); cap = torch.full((3,), 8.0, device=dev)
    # env 0: straight plan, env 1: constant left curvature 0.5 1/m, env 2: right curvature
    spec = tr.spec; R = 2.0
    kap = torch.tensor([[0.0] * N_KNOTS, [1 / R] * N_KNOTS, [-1 / R] * N_KNOTS],
                       device=dev)                                        # straight, left circle, right circle
    a = encode(kap, torch.full((3,), 3.0, device=dev), torch.full((3,), 3.0, device=dev), VMAX, spec)
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
    a = torch.zeros(B, ACT_DIM, device=dev)                       # straight; the two speed slots
    a[:, N_KNOTS] = 0.5; a[:, N_KNOTS + 1] = 0.5                  # follow the curvature knots
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
