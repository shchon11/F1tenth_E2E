"""plan_rl --plan-grad-cost (`learn.plan_grad_cost`): the differentiable decode equals the env's, each cost's gradient against
finite differences, the gates (knowable, at fault, prefix / rollout end / restarts) and the signed wall field. Env-free (CPU)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from f1sim import plan_steps as PS
from f1sim import map_controller as M
from f1sim.learn import plan_car_cost as PCC
from f1sim.learn import plan_grad_cost as PG
from f1sim.learn.plan_policy import plan_world, PT_IDX

L, W = 0.58, 0.31
K = PS.N_STEP


@pytest.fixture(scope="module")
def lim():
    lut, nominal = M.config_lut()
    return PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu", margin=0.95)


def _inputs(n, lim, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, PS.ACT_DIM, generator=g) * 0.8
    v0 = torch.rand(n, generator=g) * 8.0
    d0 = (torch.rand(n, generator=g) * 2 - 1) * PS.kappa_to_steer(lim.kappa_at(v0), lim).clamp(max=lim.s_max) * 0.9
    st = torch.zeros(n, 8)
    st[:, 0] = torch.randn(n, generator=g) * 5; st[:, 1] = torch.randn(n, generator=g) * 5
    st[:, 2] = (torch.rand(n, generator=g) * 2 - 1) * math.pi
    return u, v0, d0, st


# ------------------------------------------------------------------------------------------------------------------ decode
def test_decode_equals_env_decode(lim):
    # the env: plan_v2 rolls tanh(u) from the clamped speed estimate and the last command, Driver.decision lays it in the world
    u, v0, d0, st = _inputs(512, lim)
    with torch.no_grad():
        p = PS.roll(torch.tanh(u), v0.clamp(-PS.V_REV, PS.V_FWD), d0, lim)
        x, y, psi = plan_world(p, st)
        ref = torch.stack([x[:, PT_IDX], y[:, PT_IDX], psi[:, PT_IDX]], -1)
    ug = u.clone().requires_grad_(True)
    px, py, ppsi = PG.decode(ug, v0, d0, st, lim)
    got = torch.stack([px, py, ppsi], -1)
    assert got.requires_grad
    assert torch.equal(got.detach(), ref)                                  # bit for bit, with the graph on


def test_decode_gradient_finite_differences(lim):
    # end to end: a smooth objective of the 10 world points, its gradient on the 20 outputs vs central differences
    u, v0, d0, st = _inputs(16, lim, seed=3)
    wx = torch.randn(K); wy = torch.randn(K); wp = torch.randn(K)
    obj = lambda a: (PG.decode(a, v0, d0, st, lim)[0] * wx + PG.decode(a, v0, d0, st, lim)[1] * wy
                     + PG.decode(a, v0, d0, st, lim)[2] * wp).sum()
    ug = u.clone().requires_grad_(True)
    g, = torch.autograd.grad(obj(ug), ug)
    assert float(g.abs().sum()) > 0 and bool(torch.isfinite(g).all())
    assert float(g[:, 1:PS.N_STEP].abs().sum()) > 0 and float(g[:, PS.N_STEP + 1:].abs().sum()) > 0   # the tail outputs move it
    rng = np.random.default_rng(0)
    eps, worst = 2e-3, 0.0
    for _ in range(24):
        i, j = int(rng.integers(16)), int(rng.integers(PS.ACT_DIM))
        up, um = u.clone(), u.clone()
        up[i, j] += eps; um[i, j] -= eps
        fd = (float(obj(up)) - float(obj(um))) / (2 * eps)
        worst = max(worst, abs(fd - float(g[i, j])) / max(1.0, abs(fd)))
    assert worst < 2e-2, worst


# ------------------------------------------------------------------------------------------------------------------- terms
def _poses(n, seed):
    g = torch.Generator().manual_seed(seed)
    px = torch.randn(n, K, generator=g, dtype=torch.float64) * 0.3
    py = torch.randn(n, K, generator=g, dtype=torch.float64) * 0.3
    ps = torch.randn(n, K, generator=g, dtype=torch.float64) * 0.5
    return px, py, ps


def test_car_term_gradcheck_and_numbers():
    px, py, ps = _poses(3, 1)
    op = torch.stack([px.detach() + 0.3, py.detach() + 0.05], -1)          # the other car close: the soft band is on
    opsi = ps.detach() * 0.5
    mask = torch.ones(3, K, dtype=torch.bool)
    w = torch.tensor(PCC.weights(K, 0.3), dtype=torch.float64)
    f = lambda a, b, c: PG.car_term(a, b, c, op, opsi, mask, w, L, W, 0.05, 0.05)[0]
    assert torch.autograd.gradcheck(f, (px.requires_grad_(), py.requires_grad_(), ps.requires_grad_()), eps=1e-6, atol=1e-5)
    # the overlap flag is plan_car_cost.overlap's test
    cost, near, over = PG.car_term(px.detach(), py.detach(), ps.detach(), op, opsi, mask, w, L, W, 0.05, 0.05)
    ref = PCC.overlap(torch, torch.stack([px.detach(), py.detach()], -1), ps.detach(), op, opsi, L, W, 0.05).any(-1)
    assert torch.equal(over, ref) and bool(over.all()) and bool((cost > 0).all())
    # far apart: exactly 0, no gradient
    far = op + 5.0
    c2, n2, o2 = PG.car_term(px.detach(), py.detach(), ps.detach(), far, opsi, mask, w, L, W, 0.05, 0.05)
    assert float(c2.abs().sum()) == 0.0 and not bool(n2.any()) and not bool(o2.any())


def test_car_term_mask_and_weights():
    # one point overlapping: the cost is that point's weight x its penetration; masked -> 0
    px = torch.zeros(1, K, dtype=torch.float64); py = torch.full((1, K), 10.0, dtype=torch.float64); ps = torch.zeros(1, K, dtype=torch.float64)
    op = torch.zeros(1, K, 2, dtype=torch.float64); op[..., 1] = -10.0
    op[0, 3] = torch.tensor([0.4, 10.0])                                   # point 4 (0.4 s): the other car 0.4 m ahead
    w = torch.tensor(PCC.weights(K, 0.3), dtype=torch.float64)
    m = torch.ones(1, K, dtype=torch.bool)
    c, nr, ov = PG.car_term(px, py, ps, op, torch.zeros(1, K, dtype=torch.float64), m, w, L, W, 0.05, 0.05)
    one = PG.car_term(px[:, 3:4], py[:, 3:4], ps[:, 3:4], op[:, 3:4], torch.zeros(1, 1, dtype=torch.float64), m[:, :1],
                      torch.ones(1, dtype=torch.float64), L, W, 0.05, 0.05)[0]
    assert abs(float(c) - float(w[3]) * float(one)) < 1e-12 and bool(ov) and bool(nr)
    m2 = m.clone(); m2[0, 3] = False
    assert float(PG.car_term(px, py, ps, op, torch.zeros(1, K, dtype=torch.float64), m2, w, L, W, 0.05, 0.05)[0]) == 0.0


def _box_faces(cx, cy, yaw, hx, hy, F=8):
    """a rectangle as the env keeps a prop: world normals, offsets (padding n = 0, d = 0), valid, vertices"""
    from f1sim.prop_math import _to_world, _polygon_vertices
    pn = torch.zeros(1, F, 2, dtype=torch.float64); pd = torch.zeros(1, F, dtype=torch.float64)
    for j, (n, d) in enumerate((((1.0, 0.0), hx), ((0.0, 1.0), hy), ((-1.0, 0.0), hx), ((0.0, -1.0), hy))):
        pn[0, j] = torch.tensor(n, dtype=torch.float64); pd[0, j] = d
    valid = pn.pow(2).sum(-1) > 0.5
    nw, dw = _to_world(pn, pd, torch.tensor([[cx, cy, yaw]], dtype=torch.float64))
    return nw, dw, valid, _polygon_vertices(nw, dw, valid)


def test_box_term_gradcheck_and_narrow_box():
    nw, dw, vv, ver = _box_faces(0.5, 0.05, 0.3, 0.12, 0.10)
    bn, bd, bva, bve = nw[None], dw[None], vv[None], ver[None]            # (1, 1, F, ...)
    bm = torch.ones(1, 1, dtype=torch.bool)
    P8 = PG.outline8(L, W, dtype=torch.float64)
    w = torch.ones(K, dtype=torch.float64)
    px = torch.linspace(-0.5, 0.4, K, dtype=torch.float64)[None]; py = torch.zeros(1, K, dtype=torch.float64)
    ps = torch.full((1, K), 0.05, dtype=torch.float64)
    f = lambda a, b, c: PG.box_term(a, b, c, bn, bd, bva, bve, bm, P8, w, L, W, 0.02, 0.02)[0]
    assert torch.autograd.gradcheck(f, (px.clone().requires_grad_(), (py + 0.01).requires_grad_(), ps.clone().requires_grad_()),
                                    eps=1e-6, atol=1e-5)
    # a box narrower than the gap between two footprint points, dead ahead between them: the box's corners inside our
    # footprint count (our 8 points alone would miss it)
    nw2, dw2, vv2, ver2 = _box_faces(0.0, 0.0, 0.0, 0.05, 0.05)
    one = lambda a: PG.box_term(a, torch.zeros(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64), nw2[None],
                                dw2[None], vv2[None], ver2[None], bm, P8, torch.ones(1, dtype=torch.float64), L, W, 0.0, 0.01)
    c_in, _, hard_in = one(torch.zeros(1, 1, dtype=torch.float64))
    assert bool(hard_in) and float(c_in) > 0
    c_out, near_out, hard_out = one(torch.full((1, 1), 2.0, dtype=torch.float64))
    assert float(c_out) == 0.0 and not bool(near_out) and not bool(hard_out)
    # a box that does not count
    c_m = PG.box_term(px, py, ps, bn, bd, bva, bve, torch.zeros(1, 1, dtype=torch.bool), P8, w, L, W, 0.02, 0.02)[0]
    assert float(c_m) == 0.0


def test_box_record_picks_seen_live_near():
    from f1sim.prop_math import _polygon_vertices
    C, F = 5, 8
    pn = torch.zeros(1, C, F, 2); pd = torch.full((1, C, F), float("inf"))
    for c in range(C):
        for j, n in enumerate(((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))):
            pn[0, c, j] = torch.tensor(n); pd[0, c, j] = 0.15
    poses = torch.tensor([[[3.0, 0.0, 0.0], [1.0, 0.0, 0.5], [20.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.5, 0.0, 0.0]]])
    sid = torch.tensor([[0, 0, 0, 0, -1]])                                # slot 4 dead
    po = SimpleNamespace(p_poses=poses, p_sid=sid, p_n=pn, p_d=pd)
    seen = torch.tensor([[True, True, True, False, True]])                # slot 3 never seen, slot 2 too far
    bn, bd, bv, bver, bm = PG.box_record(po, torch.tensor([0]), seen, torch.zeros(1, 2), 4, 12.0)
    assert bm.tolist() == [[True, True, False, False]]
    # the nearest first: slot 1 (1 m), then slot 0 (3 m); slot 1's faces turned by its yaw, its corners 0.15 * sqrt 2 out
    c, s = math.cos(0.5), math.sin(0.5)
    assert torch.allclose(bn[0, 0, 0], torch.tensor([c, s]), atol=1e-6)
    r = (bver[0, 0, :4] - torch.tensor([1.0, 0.0])).norm(dim=-1)
    assert torch.allclose(r, torch.full((4,), 0.15 * math.sqrt(2)), atol=1e-5)
    assert torch.allclose(bver[0, 1, :4].mean(0), torch.tensor([3.0, 0.0]), atol=1e-5)


# -------------------------------------------------------------------------------------------------------------------- walls
@pytest.fixture(scope="module")
def field():
    from f1sim.track import Track, TrackTensors
    res = 0.05
    occ = np.zeros((80, 120), bool)
    occ[:, :20] = True                                                     # a wall on the left: cells with centre x < 1.0
    occ[:5, :] = True; occ[-5:, :] = True; occ[:, -5:] = True
    t = Track.from_occupancy(occ, res, origin=(0.0, 0.0))
    tt = TrackTensors([t], "cpu", dtype=torch.float32)
    return PG.WallField(tt), tt


def test_wall_field_sign_and_distance(field):
    f, tt = field
    # cell centres: the sign is the occupancy (sample_edt <= 0 in a wall)
    g = torch.Generator().manual_seed(0)
    xy = torch.stack([torch.randint(0, 120, (400,), generator=g), torch.randint(0, 80, (400,), generator=g)], -1).float() * 0.05
    tid = torch.zeros(400, dtype=torch.long)
    sd = f.sample(xy, tid)
    wall = tt.sample_edt(xy, tid) <= 0.0
    assert torch.equal(sd < 0, wall)
    # beside the flat left wall (surface between the cell centres 0.95 and 1.00, at x = 0.975): the distance to it
    x = torch.linspace(0.6, 1.6, 41)
    pts = torch.stack([x, torch.full_like(x, 2.0)], -1)
    d = f.sample(pts, torch.zeros(41, dtype=torch.long))
    assert torch.allclose(d, x - 0.975, atol=1e-4)
    # off the grid: a wall
    assert float(f.sample(torch.tensor([[-3.0, -3.0]]), torch.zeros(1, dtype=torch.long))) < 0


def test_wall_term_gradcheck_and_push(field):
    f, _ = field
    P8 = PG.outline8(L, W, dtype=torch.float64)
    w = torch.ones(K, dtype=torch.float64)
    px = torch.full((1, K), 1.2, dtype=torch.float64); py = torch.linspace(1.5, 2.5, K, dtype=torch.float64)[None]
    ps = torch.full((1, K), 0.3, dtype=torch.float64)
    fn = lambda a, b, c: PG.wall_term(a, b, c, torch.zeros(1, dtype=torch.long), f, P8, w, 0.02, 0.02)[0]
    a = (px + 0.0123).requires_grad_(); b = (py + 0.0077).requires_grad_(); c = ps.clone().requires_grad_()
    assert torch.autograd.gradcheck(fn, (a, b, c), eps=1e-6, atol=1e-5)
    cost = fn(a, b, c)
    gx, = torch.autograd.grad(cost, a)
    assert float(cost.detach()) > 0 and bool((gx <= 0).all()) and float(gx.sum()) < 0       # pushed away from the left wall (+x)
    far = fn(torch.full((1, K), 3.0, dtype=torch.float64), py, torch.zeros(1, K, dtype=torch.float64))
    assert float(far) == 0.0


# -------------------------------------------------------------------------------------------------------------------- gates
def test_later_poses_gates():
    T, B = 30, 2
    op_xy = torch.zeros(T + 1, B, 2); op_xy[..., 0] = torch.arange(T + 1).float()[:, None] * 0.4
    op_psi = torch.zeros(T + 1, B); op_rem = 100.0 - op_xy[..., 0]
    op_ok = torch.ones(T + 1, B, dtype=torch.bool)
    brk = torch.zeros(T, B, dtype=torch.bool); brk[12, 1] = True            # car 1 restarted during decision 12
    know = torch.ones(T, B, dtype=torch.bool); know[5, 0] = False
    o = PG.later_poses(op_xy, op_psi, op_rem, op_ok, brk, know, list(range(1, K + 1)))
    ok = o["ok"]
    assert not bool(ok[5, 0].any())                                         # not knowable: nothing compared
    assert int(ok[25, 0].sum()) == 5                                        # the rollout's end: t + k <= T
    assert int(ok[T - 1, 0].sum()) == 1
    assert int(ok[10, 1].sum()) == 2 and int(ok[12, 1].sum()) == 0          # a restart in decision 12 cuts after it
    assert int(ok[13, 1].sum()) == K
    # the poses: point k is the other car k decisions on; the previous point's pose the one before (the decision's own first)
    assert torch.allclose(o["xy"][3, 0, :, 0], (3 + torch.arange(1, K + 1)).float() * 0.4)
    assert torch.allclose(o["xy_prev"][3, 0, :, 0], (3 + torch.arange(0, K)).float() * 0.4)
    assert torch.allclose(o["rem_prev"][3, 0], 100.0 - (3 + torch.arange(0, K)).float() * 0.4)


def test_fault_gate():
    n = 3
    me0 = torch.zeros(n, 2); psi0 = torch.zeros(n); rem0 = torch.full((n,), 50.0)
    pl = torch.zeros(n, K, 2); pl[..., 0] = torch.arange(1, K + 1).float() * 0.5
    plpsi = torch.zeros(n, K); plrem = 50.0 - pl[..., 0]
    gap = torch.tensor([2.0, -0.3, -1.0])                                   # the other car ahead / just behind / well behind
    oxy = torch.zeros(n, K, 2)
    oxy[..., 0] = torch.cat([me0[:, None, 0], pl[:, :-1, 0]], 1) + gap[:, None]
    orem = 50.0 - oxy[..., 0]
    gate = PG.fault_gate(me0, psi0, rem0, pl, plpsi, plrem, oxy, orem, torch.full((n,), 100.0))
    assert bool(gate[0].all()) and bool(gate[1].all()) and not bool(gate[2].any())
    # no distance to go (in a wall): along our heading instead
    gate2 = PG.fault_gate(me0, psi0, torch.full((n,), float("nan")), pl, plpsi, torch.full((n, K), float("nan")), oxy, orem,
                          torch.full((n,), 100.0))
    assert torch.equal(gate2, gate)


def test_pen_and_gae():
    d = torch.tensor([0.0, 0.1, 0.15, 0.5])        # 0.5: past m + 4 s
    p = PG.pen(d, 0.1, 0.02)
    assert float(p[3]) == 0.0 and float(p[0]) > float(p[1]) > float(p[2]) > 0.0
    assert abs(float(p[1]) - 0.02 * math.log(2.0)) < 1e-7
    r = torch.zeros(5, 1); r[3] = -1.0
    done = torch.zeros(5, 1); done[1] = 1.0
    a = PG.gae_zero_baseline(r, done, 0.9, 0.5)
    assert abs(float(a[2]) + 0.45) < 1e-6 and float(a[1]) == 0.0 and abs(float(a[3]) + 1.0) < 1e-7
