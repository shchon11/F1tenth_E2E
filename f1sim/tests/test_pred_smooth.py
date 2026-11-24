"""(11-25) plan_rl --pred-smooth, the parts that need no simulator: the other car's predicted second as a path integrated from
its pose now, speed and curvature (`OppPredictor.smooth`), in the old output format; the fork from a checkpoint with the old
head (`plan_rl.init_actor`) keeps the encoder and the 'there' branch and zeroes the trunk's point columns."""
import math

import numpy as np
import pytest
import torch

from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import (Actor2, N_SCANS, STATE_DIM2, PRED_IN, PRED_OUT, PRED_N, PRED_SCALE, PRED_KIN_OUT, PRED_NB,
                                     load_actor2, pred_points, pred_zigzag, unicycle)

BEAMS = 90
MEM = 16
K_LOCK, A_LAT = 2.0, 6.0


def _kin():
    """a car's curvature limit as `plan_steps.Limits` makes it: speeds 0..12 m/s every 0.05, min(lock, a_lat / v^2)"""
    v = torch.linspace(0.0, 12.0, 241)
    return v, torch.minimum(torch.full_like(v, K_LOCK), A_LAT / v.clamp_min(1e-3) ** 2)


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N_SCANS, BEAMS, generator=g), torch.randn(n, STATE_DIM2, generator=g)


def _old_parent():
    torch.manual_seed(0)
    a = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True).eval()
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.05 * torch.randn_like(p))              # trained-looking: the trunk reads the old points too
        a.norm.mean.normal_(); a.norm.var.uniform_(0.5, 2.0)
    return a


def _smooth(**kw):
    return Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True, pred_smooth=True, pred_kin=_kin(), **kw)


def test_constant_speed_and_curvature_is_an_arc():
    v, k, dt = 4.0, 0.5, 0.1
    n = 1
    pts = unicycle(torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.full((n, PRED_N), v), torch.full((n, PRED_N), k), dt)[0]
    r = 1.0 / k
    # on the circle about (0, r), each step v dt of arc further on
    assert torch.allclose((pts - torch.tensor([0.0, r])).norm(dim=-1), torch.full((PRED_N,), r), atol=1e-5)
    ang = torch.atan2(pts[:, 0], r - pts[:, 1])
    assert torch.allclose(ang, torch.arange(1, PRED_N + 1) * v * k * dt, atol=1e-5)
    # straight on (curvature 0) and standing (speed 0)
    s = unicycle(torch.zeros(1), torch.zeros(1), torch.full((1,), 0.3), torch.full((1, PRED_N), 2.0), torch.zeros(1, PRED_N))[0]
    assert torch.allclose(s, torch.arange(1, PRED_N + 1)[:, None] * 0.2 * torch.tensor([math.cos(0.3), math.sin(0.3)]), atol=1e-5)
    z = unicycle(torch.ones(1), torch.ones(1), torch.zeros(1), torch.zeros(1, PRED_N), torch.full((1, PRED_N), 1.0))[0]
    assert torch.allclose(z, torch.ones(PRED_N, 2))


def test_curvature_stays_within_the_car_limit():
    a = _smooth().eval()
    g = torch.Generator().manual_seed(3)
    o = torch.randn(4096, PRED_KIN_OUT, generator=g) * 20.0      # far outside anything a head would say
    v, k = a.pred.kin_vk(o)
    assert bool((v >= 0).all())
    # the limit as the car's table gives it (`plan_steps.Limits.kappa_at`: linear between its speeds), above its top speed
    # the top's lateral acceleration
    vg, kg = (x.double().numpy() for x in _kin())
    vv = v.detach().double().numpy()
    lim = torch.from_numpy(np.where(vv > vg[-1], kg[-1] * (vg[-1] / np.maximum(vv, 1e-9)) ** 2, np.interp(vv, vg, kg))).float()
    assert bool((k.abs() <= lim * (1 + 1e-5) + 1e-7).all())
    assert bool((lim <= torch.minimum(torch.full_like(v, K_LOCK), A_LAT / v.clamp_min(1e-6) ** 2) * 1.01).all())
    assert float(k.abs().max()) > 0.9 * K_LOCK                  # and reaches it: the lock at low speed
    # the integrated path's turn per metre never exceeds the limit at the steps' speeds either (outputs a head might give;
    # steps turning < 0.5 rad, where a chord is its arc)
    o = torch.randn(4096, PRED_KIN_OUT, generator=g) * 2.0
    v, k = a.pred.kin_vk(o)
    vv = v.detach().double().numpy()
    lim = torch.from_numpy(np.where(vv > vg[-1], kg[-1] * (vg[-1] / np.maximum(vv, 1e-9)) ** 2, np.interp(vv, vg, kg))).float()
    pts, z = (x.detach() for x in a.pred.kin_path(o))
    full = torch.cat([z[:, None, :2], pts], 1)
    seg = full[:, 1:] - full[:, :-1]
    turn = torch.atan2(seg[..., 1], seg[..., 0]).diff(dim=1).remainder(2 * math.pi)
    turn = torch.minimum(turn, 2 * math.pi - turn)
    L = 0.5 * (seg[:, 1:].norm(dim=-1) + seg[:, :-1].norm(dim=-1))
    phi = (v * k * 0.1).abs().detach()
    ok = (L > 0.05) & (phi[:, 1:] < 0.5) & (phi[:, :-1] < 0.5)
    kmax = torch.maximum(lim[:, 1:], lim[:, :-1])
    assert int(ok.sum()) > 10000 and float((turn[ok] / L[ok]).max()) > 0.5 * K_LOCK
    assert bool((turn[ok] / L[ok] <= kmax[ok] * 1.02 + 1e-3).all())


def test_the_output_format_is_unchanged():
    a = _smooth().eval()
    sc, st = _inputs(5, 1)
    mean, log_std, v_hat = a(sc, st, torch.zeros(5, MEM))
    q = a.pred_out
    assert q.shape == (5, PRED_OUT) and a.pred_feed(q).shape == (5, PRED_IN)
    pts, pv = pred_points(q)
    assert pts.shape == (5, PRED_N, 2) and pv.shape == (5,)
    # the points are the integrated path of the last layer's output
    f = a.features(sc, st)
    h1 = a.mem_cell(f, torch.zeros(5, MEM))
    q2, z = a.pred_q(sc, f, h1, pose=True)
    o = a.pred.mlp(torch.cat([a.pred.enc(sc), f[:, 256:], h1], 1))
    p_ref, z_ref = a.pred.kin_path(o)
    assert torch.allclose(q2, q, atol=1e-6) and torch.allclose(q2[:, :-1].view(5, PRED_N, 2) * PRED_SCALE, p_ref, atol=1e-5)
    assert torch.allclose(z, z_ref) and torch.allclose(q2[:, -1], o[:, -1])
    # a fresh head: a car here heading as we do, ~PRED_V0 straight on
    z, pts = z.detach(), pts.detach()
    assert float(z[:, :2].abs().max()) < 0.5 and float(z[:, 2].abs().max()) < 0.2
    assert 3.0 < float(pts[:, -1, 0].mean()) < 5.0


def test_a_fork_from_the_old_head_keeps_its_encoder_and_there_branch_and_drives_as_its_parent():
    base = _old_parent()
    fork = _smooth()
    added = PR.init_actor(fork, base.state_dict())
    fork.eval()
    assert added == "smooth"
    sb, sf = base.state_dict(), fork.state_dict()
    for k_ in sb:
        if k_.startswith("pred.enc.") or k_.startswith("pred.mlp.0.") or k_.startswith("pred.mlp.2."):
            assert torch.equal(sb[k_], sf[k_]), k_
    assert sf["pred.mlp.4.weight"].shape == (PRED_KIN_OUT, 256)
    assert torch.equal(sf["pred.mlp.4.weight"][-1], sb["pred.mlp.4.weight"][-1])
    assert torch.equal(sf["pred.kin_k"], _kin()[1])                     # the actor's own limit (not in the old checkpoint)
    assert torch.equal(sf["pred.mlp.4.bias"][-1], sb["pred.mlp.4.bias"][-1])
    tw_b, tw_f = sb["trunk.0.weight"], sf["trunk.0.weight"]
    assert bool((tw_f[:, -2 * PRED_N:] == 0).all())
    assert torch.equal(tw_f[:, :-2 * PRED_N], tw_b[:, :-2 * PRED_N])    # the 'there' column (-PRED_IN) and everything else kept
    assert float(tw_b[:, -2 * PRED_N:].abs().max()) > 0                 # (the parent did read its points)
    # the fork drives as the parent with ITS point columns zeroed (the fork = the parent minus what its old points pulled):
    # to float rounding -- the 'there' logit is the same row of weights, but one of a 12-row instead of a 21-row product, and
    # the matrix kernel sums in a shape-dependent order (measured: 1 ulp of the logit, 6e-8 in the plan's mean)
    ref = _old_parent()
    with torch.no_grad():
        ref.trunk[0].weight[:, -2 * PRED_N:] = 0.0
    hr = hf = torch.zeros(5, MEM)
    for k in range(6):
        sc, st = _inputs(5, k)
        mr, lr, vr = ref(sc, st, hr); hr = ref.h_out
        mf, lf, vf = fork(sc, st, hf); hf = fork.h_out
        assert torch.equal(lr, lf) and torch.equal(hr, hf)               # the memory never reads the prediction
        assert torch.allclose(mr, mf, rtol=0, atol=1e-6) and torch.allclose(vr, vf, rtol=0, atol=1e-6)
        assert torch.allclose(ref.pred_out[:, -1], fork.pred_out[:, -1], rtol=0, atol=1e-6)   # the same P(there)
    # and once the trunk's point weights move, the new path matters
    with torch.no_grad():
        fork.trunk[0].weight[:, -2 * PRED_N:].normal_(0.0, 0.5)
    assert float((fork(sc, st, torch.zeros(5, MEM))[0] - ref(sc, st, torch.zeros(5, MEM))[0]).abs().max()) > 1e-4


def test_the_gradient_flows_through_the_integration():
    a = _smooth()
    with torch.no_grad():                                     # off the flat start, so every output moves the path
        a.pred.mlp[-1].weight.normal_(0.0, 0.05)
    a.train()
    sc, st = _inputs(8, 2)
    f = a.features(sc, st)
    h1 = a.mem_cell(f, torch.zeros(8, MEM))
    q, z = a.pred_q(sc, f, h1, pose=True)
    tgt = torch.randn(8, PRED_N, 2)
    loss = torch.nn.functional.smooth_l1_loss(q[:, :-1].view(8, PRED_N, 2), tgt / PRED_SCALE)
    g = torch.autograd.grad(loss, [a.pred.mlp[-1].weight, a.pred.enc.conv[0].weight])
    gw = g[0].abs().sum(1)
    assert bool((gw[:-1] > 0).all())                          # pose, every speed and curvature coefficient
    assert float(gw[-1]) == 0.0                               # the 'there' row: not from the points
    assert float(g[1].abs().sum()) > 0                        # down into the head's encoder
    # the integration itself, against finite differences
    x0, y0, th0 = (torch.randn(3, dtype=torch.float64, requires_grad=True) for _ in range(3))
    v = (torch.rand(3, PRED_N, dtype=torch.float64) * 6).requires_grad_(True)
    k = (torch.randn(3, PRED_N, dtype=torch.float64) * 0.5).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda *a_: unicycle(*a_), (x0, y0, th0, v, k))
    k0 = torch.zeros(3, PRED_N, dtype=torch.float64, requires_grad=True)   # straight: sinc at 0
    assert torch.autograd.gradcheck(lambda *a_: unicycle(*a_), (x0, y0, th0, v, k0))


def test_checkpoints_load_with_their_own_head():
    s = _smooth()
    with torch.no_grad():
        s.pred.mlp[-1].weight.normal_(0.0, 0.05)
    s.eval()
    args = {"memory": MEM, "mem_front": True, "std0": 0.6, "pred_smooth": True}
    a = load_actor2({"actor": s.state_dict(), "args": args, "beams": BEAMS}, "cpu", check_mount=False)
    assert a.pred_smooth and a.pred.smooth and torch.equal(a.pred.kin_k, s.pred.kin_k)
    sc, st = _inputs(4, 5)
    assert torch.equal(a(sc, st, torch.zeros(4, MEM))[0], s(sc, st, torch.zeros(4, MEM))[0])
    assert torch.equal(a.pred_out, s.pred_out)
    old = _old_parent()
    a0 = load_actor2({"actor": old.state_dict(), "args": {"memory": MEM, "mem_front": True}, "beams": BEAMS}, "cpu", check_mount=False)
    assert a0.pred_opp and not a0.pred_smooth
    # a smooth checkpoint into an old-head actor: refused with the reason
    with pytest.raises(ValueError, match="pred-smooth"):
        PR.init_actor(Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True), s.state_dict())
    # a smooth checkpoint into a smooth actor: all of it
    s2 = _smooth()
    assert PR.init_actor(s2, s.state_dict()) == ""
    assert all(torch.equal(x, y) for x, y in zip(s2.state_dict().values(), s.state_dict().values()))


def test_zigzag_measure():
    t = torch.linspace(0.1, 1.0, PRED_N)
    real = torch.stack([4 * t, 0.3 * t ** 2], -1)[None]
    assert float(pred_zigzag(real, real).abs().max()) < 1e-4
    zz = real.clone(); zz[0, 1::2, 1] += 0.1                 # every other point 10 cm aside
    assert float(pred_zigzag(zz, real)[0]) > 10.0
    stand = torch.zeros(1, PRED_N, 2)                         # a standing car: no turn counts, the decision is left out
    assert pred_zigzag(zz, stand).numel() == 0
