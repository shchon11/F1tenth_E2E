import pytest
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


# ---------------------------------------------------------------- speed modes (opt-in)
def test_speed_mode_dimensions_and_the_linear_reader_refuses_the_others():
    import pytest
    from f1sim.mpc import act_dim
    assert act_dim() == act_dim("linear") == ACT_DIM == N_KNOTS + 2
    assert act_dim("envelope") == N_KNOTS + 2 and act_dim("knots") == 2 * N_KNOTS
    with pytest.raises(ValueError):
        act_dim("spline")
    # `decode` is what every runtime layer reads a plan through, and it means (v_start, v_end) by
    # the last two numbers. Handed an envelope plan it must fail, not return a plausible wrong plan.
    a = torch.zeros(2, ACT_DIM); v = torch.full((2,), 3.0); cap = torch.full((2,), 8.0)
    with pytest.raises(ValueError):
        decode(a, v, VMAX, cap, PlanSpec(speed_mode="envelope"))


def test_knots_profile_passes_through_its_knots():
    from f1sim.mpc import decode_profile, encode_knots
    spec = PlanSpec(speed_mode="knots")
    kap = torch.zeros(1, N_KNOTS); vk = torch.tensor([[2.0, 3.0, 1.5, 1.5, 4.0, 5.0]])
    a = encode_knots(kap, vk, VMAX, spec)
    _, _, prof = decode_profile(a, torch.full((1,), 3.0), VMAX, torch.full((1,), 8.0), spec)
    at = torch.linspace(0, spec.n_profile - 1, N_KNOTS).round().long()
    assert prof.shape == (1, spec.n_profile) and torch.allclose(prof[0, at], vk[0], atol=0.12)
    assert float(prof.min()) >= 1.5 - 1e-5                  # an apex *inside* the plan, which two numbers cannot say
    _, _, capped = decode_profile(a, torch.full((1,), 3.0), VMAX, torch.full((1,), 2.5), spec)
    assert float(capped.max()) <= 2.5 + 1e-6


def test_envelope_profile_follows_the_corner_and_the_grip_belief():
    from f1sim.mpc import decode_profile, encode_envelope, path_points
    spec = PlanSpec(speed_mode="envelope", envelope_forward=False)
    # straight into a 1.25 m radius corner and out again; the car is at 6 m/s so the plan is 9 m long
    kap = torch.tensor([[0.0, 0.0, 0.8, 0.8, 0.0, 0.0]]).repeat(3, 1)
    a_hat = torch.tensor([8.0, 4.0, 8.0]); v_end = torch.tensor([8.0, 8.0, 1.0])
    v = torch.full((3,), 6.0); cap = torch.full((3,), 8.0)
    k, Lp, prof = decode_profile(encode_envelope(kap, a_hat, v_end, VMAX, spec), v, VMAX, cap, spec)
    _, _, _, s, kd = path_points(k, Lp, n=spec.n_profile, return_kappa=True)
    # never faster than the corner allows at the believed grip, and it gets there by braking in time
    assert (prof <= torch.sqrt(a_hat[:, None] / kd.abs().clamp_min(1e-3)) + 1e-4).all()
    i_apex = int(kd[0].argmax())
    assert abs(float(prof[0, i_apex]) - math.sqrt(8.0 / 0.8)) < 0.05 and float(prof[0, 0]) > float(prof[0, i_apex]) + 1.0
    dec = (prof[:, :-1] ** 2 - prof[:, 1:] ** 2) / (2 * (Lp / (spec.n_profile - 1))[:, None])
    assert float(dec.max()) <= spec.a_brake_profile + 1e-3
    # one scalar is the whole "how slippery is it": halve it and every lateral-limited point slows by sqrt(2)
    assert (prof[1] <= prof[0] + 1e-5).all() and abs(float(prof[1, i_apex] / prof[0, i_apex]) - math.sqrt(0.5)) < 0.01
    # and the end speed says what the curvature cannot (a hairpin past the plan, a car in the way)
    assert float(prof[2, -1]) <= 1.0 + 1e-5 and (prof[2] <= prof[0] + 1e-5).all()
    # the drive pass starts from the speed the car has, not the one the corner would allow
    ramp = decode_profile(encode_envelope(kap[:1] * 0, a_hat[:1], v_end[:1], VMAX, spec), torch.full((1,), 2.0), VMAX, cap[:1],
                          PlanSpec(speed_mode="envelope"))[2]
    assert abs(float(ramp[0, 0]) - 2.0) < 1e-5 and (ramp[0, 1:] >= ramp[0, :-1] - 1e-6).all() and float(ramp[0, -1]) < 8.0


def test_tracker_runs_every_speed_mode():
    from f1sim.mpc import act_dim
    for mode in ("envelope", "knots"):
        tr = PlanTracker(4, "cpu", WB, SMAX, VMAX, spec=PlanSpec(speed_mode=mode), compile_solver=False)
        a = torch.zeros(4, act_dim(mode)); a[:, :N_KNOTS] = 0.2
        cmd = tr(a, torch.full((4,), 3.0), torch.full((4,), 8.0))
        assert cmd.shape == (4, 2) and torch.isfinite(cmd).all() and (cmd[:, 0] > 0).all()
        assert tr.last_ref.shape == (4, tr.spec.N + 1, 4)

# ==================================================================== what a curvature knot means
def test_a_feasible_knot_is_a_share_of_the_grip_the_speed_leaves():
    """`kappa_mode="feasible"`: +-1 is the tightest arc the tyres hold at the plan's own speed, so
    the box spans what the car can drive instead of a fixed 1.6 1/m it cannot.

    Under "absolute" at 5 m/s a 21-point sweep of one knot produced two trajectories -- everything
    past |a| = 0.2 asks for more than 8 m/s^2 and the tracker saturates -- while the policy's own
    exploration (log_std ~ 0.15, i.e. 0.24 1/m) is wider than that whole band."""
    import torch
    from f1sim.mpc import PlanSpec, N_KNOTS, decode, encode, kappa_scale
    sp = PlanSpec(kappa_mode="feasible", kappa_a_lat=8.0)
    v = torch.tensor([0.5, 2.0, 5.0, 8.0])
    scale = kappa_scale(v, sp).flatten()
    assert float(scale[0]) == pytest.approx(sp.kappa_max)          # standing still: the whole box
    assert float(scale[1]) == pytest.approx(sp.kappa_max)          # 2 m/s: 8 / 4 = 2 > 1.6, clamped
    assert float(scale[2]) == pytest.approx(8.0 / 25.0)            # 5 m/s: the grip limit itself
    assert float(scale[3]) == pytest.approx(8.0 / 64.0)
    a = torch.zeros(4, N_KNOTS + 2); a[:, :N_KNOTS] = 1.0
    k, _, v0, v1 = decode(a, v, 9.0, torch.full_like(v, 9.0), sp)
    lat = k[:, 0] * v.square()                                     # lateral acceleration asked for
    assert float(lat[2]) == pytest.approx(8.0) and float(lat[3]) == pytest.approx(8.0)
    # and the inverse is the inverse, at the speed the plan starts from
    back = encode(k[:, :N_KNOTS], v0, v1, 9.0, sp, v_meas=v)
    assert torch.allclose(back[:, :N_KNOTS], a[:, :N_KNOTS], atol=1e-6)
    with pytest.raises(ValueError, match="v_meas"):
        encode(k[:, :N_KNOTS], v0, v1, 9.0, sp)                    # silently wrong plans, refused


def test_absolute_is_what_it_always_was():
    import torch
    from f1sim.mpc import PlanSpec, N_KNOTS, decode, encode
    sp = PlanSpec()
    assert sp.kappa_mode == "absolute"
    v = torch.tensor([1.0, 6.0])
    a = torch.zeros(2, N_KNOTS + 2); a[:, :N_KNOTS] = 0.5
    k, _, v0, v1 = decode(a, v, 9.0, torch.full_like(v, 9.0), sp)
    assert torch.allclose(k, torch.full_like(k, 0.5 * sp.kappa_max))
    assert torch.allclose(encode(k[:, :N_KNOTS], v0, v1, 9.0, sp)[:, :N_KNOTS], a[:, :N_KNOTS], atol=1e-6)
