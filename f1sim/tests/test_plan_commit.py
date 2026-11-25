"""plan_rl --commit-plan (`learn.plan_commit`): the coefficient noise's log-probability, the commitment cost's geometry and
time alignment, the gate (no promise / contact / push / scene), and fork parity (the forked actor's mean is its parent's, bit
for bit, and a saved checkpoint of the mode loads in the exam's / console's loader unchanged). Env-free (CPU)."""
import math
import os

import pytest
import torch

from f1sim import plan_steps as PS
from f1sim import map_controller as M
from f1sim.learn import plan_commit as C
from f1sim.learn.plan_policy import plan_world, PT_IDX, tanh_logp

N = PS.N_STEP
PARENT = os.path.expanduser("~/f1sim_runs/race33b/plan_u975.pt")


@pytest.fixture(scope="module")
def lim():
    lut, nominal = M.config_lut()
    return PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu", margin=0.95)


# --------------------------------------------------------------------------------------------------------------- the basis
def test_basis_orthonormal_and_ordered():
    Q = C.coef_basis()
    assert torch.allclose(Q @ Q.T, torch.eye(N, dtype=Q.dtype), atol=1e-12)
    assert torch.allclose(Q[0], torch.full((N,), 1.0 / math.sqrt(N), dtype=Q.dtype))       # degree 0: a constant
    t = (torch.arange(N, dtype=torch.float64) + 0.5) / N * 2 - 1
    for j in range(N):                                     # row j is a polynomial of degree j: exact fit at degree j, not below
        V = torch.stack([t ** i for i in range(j + 1)], 1)
        res = Q[j] - V @ torch.linalg.lstsq(V, Q[j][:, None]).solution[:, 0]
        assert float(res.abs().max()) < 1e-9
        if j > 0:
            V1 = V[:, :-1]
            res1 = Q[j] - V1 @ torch.linalg.lstsq(V1, Q[j][:, None]).solution[:, 0]
            assert float(res1.abs().max()) > 1e-3
    x = torch.randn(7, 2 * N)
    assert torch.allclose(C.to_coef(C.to_steps(x, Q.float()), Q.float()), x, atol=1e-5)


# ----------------------------------------------------------------------------------------------------------- log-prob
def test_logp_is_the_sum_over_coefficients_and_the_full_gaussian():
    g = torch.Generator().manual_seed(0)
    Q = C.coef_basis()                                    # float64 for an exact comparison
    B = 64
    mean = torch.randn(B, 2 * N, generator=g, dtype=torch.float64)
    log_std = (torch.rand(2 * N, generator=g, dtype=torch.float64) * 1.5 - 2.5)
    z = torch.randn(B, 2 * N, generator=g, dtype=torch.float64)
    u = C.sample(mean, log_std, Q, z)
    lp = C.commit_logp(u, mean, log_std, Q)
    # (a) the sum over the 20 coefficients of N(c; 0, sigma), c = Q (u - mean) = sigma z, plus tanh's Jacobian
    jac = 2.0 * (math.log(2.0) - u - torch.nn.functional.softplus(-2.0 * u))
    ref_a = torch.distributions.Normal(torch.zeros_like(z), log_std.exp().expand_as(z)).log_prob(log_std.exp() * z).sum(-1) - jac.sum(-1)
    assert torch.allclose(lp, ref_a, atol=1e-9)
    # (b) the same as the full 20-d Gaussian of u (covariance blockdiag(Q^T diag(sigma^2) Q)) through tanh
    S = torch.zeros(2 * N, 2 * N, dtype=torch.float64)
    s2 = log_std.exp().pow(2)
    S[:N, :N] = Q.T @ torch.diag(s2[:N]) @ Q
    S[N:, N:] = Q.T @ torch.diag(s2[N:]) @ Q
    mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=S)
    assert torch.allclose(lp, mvn.log_prob(u) - jac.sum(-1), atol=1e-8)
    # (c) with every sigma equal the basis drops out: the old independent per-output Gaussian (`tanh_logp`)
    ls_eq = torch.full((2 * N,), -1.0, dtype=torch.float64)
    assert torch.allclose(C.commit_logp(u, mean, ls_eq, Q), tanh_logp(u, mean, ls_eq.expand_as(u)), atol=1e-9)


def test_logp_gradient_and_sampling_covariance():
    g = torch.Generator().manual_seed(1)
    Q = C.coef_basis().float()
    ls = C.init_log_std(3, 0.2, 0.3, 0.05)
    mean = torch.randn(4, 2 * N, generator=g).requires_grad_(True)
    u = C.sample(mean.detach(), ls, Q, torch.randn(4, 2 * N, generator=g))
    lp = C.commit_logp(u, mean, ls, Q).sum()
    gm, = torch.autograd.grad(lp, mean)
    # d log N(u; mean, S) / d mean = S^-1 (u - mean), S^-1 = Q^T diag(1/sigma^2) Q per channel
    Si = torch.zeros(2 * N, 2 * N)
    Si[:N, :N] = Q.T @ torch.diag(ls[:N].exp().pow(-2)) @ Q
    Si[N:, N:] = Q.T @ torch.diag(ls[N:].exp().pow(-2)) @ Q
    assert torch.allclose(gm, (u - mean.detach()) @ Si.T, rtol=1e-4, atol=1e-3)
    # the noise's covariance per output is the model's: smooth (degree < 3) coefficients 0.2 / 0.3, the rest 0.05
    z = torch.randn(200000, 2 * N, generator=g)
    x = C.to_steps(ls.exp() * z, Q)
    emp = (x.T @ x) / x.shape[0]
    S = torch.zeros(2 * N, 2 * N)
    S[:N, :N] = Q.T @ torch.diag(ls[:N].exp().pow(2)) @ Q
    S[N:, N:] = Q.T @ torch.diag(ls[N:].exp().pow(2)) @ Q
    assert float((emp - S).abs().max()) < 2e-3
    assert torch.allclose(C.step_std(ls, Q), S.diagonal().sqrt(), atol=1e-6)


def test_floors_and_start():
    f = C.floor_vec(3, 0.15, 0.2, 0.05)
    assert torch.allclose(f[:3].exp(), torch.full((3,), 0.15)) and torch.allclose(f[N:N + 3].exp(), torch.full((3,), 0.2))
    assert torch.allclose(f[3:N].exp(), torch.full((N - 3,), 0.05)) and torch.allclose(f[N + 3:].exp(), torch.full((N - 3,), 0.05))
    f0 = C.floor_vec(2, 0.0, 0.1, 0.0)
    assert bool(torch.isinf(f0[:N]).all()) and bool(torch.isinf(f0[N + 2:]).all()) and math.isclose(float(f0[N].exp()), 0.1, rel_tol=1e-6)
    s = C.init_log_std(3, 0.2, 0.25, 0.1).exp()
    assert torch.allclose(s, torch.tensor([0.2] * 3 + [0.1] * 7 + [0.25] * 3 + [0.1] * 7))


# ------------------------------------------------------------------------------------------------- the cost's geometry
def _world(p, st):
    x, y, _ = plan_world(p, st)
    return torch.stack([x[:, PT_IDX], y[:, PT_IDX]], -1)


def test_kept_promise_costs_nothing_and_alignment(lim):
    # a plan rolled on from its own 0.1 s state with its own actions one step on is the same trajectory: the new plan's
    # point k IS the old plan's point k+1 (positions and speeds) -> zero gap, zero cost
    g = torch.Generator().manual_seed(2)
    B = 256
    u = torch.randn(B, 2 * N, generator=g) * 0.7
    v0 = torch.rand(B, generator=g) * 7.0 + 0.5
    d0 = torch.zeros(B)
    st = torch.zeros(B, 8); st[:, 0] = torch.randn(B, generator=g) * 3; st[:, 1] = torch.randn(B, generator=g) * 3
    st[:, 2] = (torch.rand(B, generator=g) * 2 - 1) * math.pi
    old = PS.roll(torch.tanh(u), v0, d0, lim)
    old_w = _world(old, st)
    # the car where the old plan put it after 0.1 s: pose, speed, steering at its step-1 end
    x1, y1, ps1 = plan_world(old, st)
    st1 = st.clone(); st1[:, 0] = x1[:, PS.SUB]; st1[:, 1] = y1[:, PS.SUB]; st1[:, 2] = ps1[:, PS.SUB]
    u1 = torch.cat([u[:, 1:N], u[:, N - 1:N], u[:, N + 1:], u[:, 2 * N - 1:]], 1)      # its actions one step on (last repeated)
    new = PS.roll(torch.tanh(u1), old["v"][:, PS.SUB], old["delta"][:, PS.SUB], lim)
    new_w = _world(new, st1)
    ok = torch.ones(B, dtype=torch.bool)
    out = C.promise_cost(new_w, new["pv"], old_w, old["pv"], ok, w=0.01, ref=0.75, v_s=0.2)
    assert float(out["dp"].max()) < 2e-4 and float(out["dv"].max()) < 1e-4
    assert float(out["cost"].max()) < 1e-6
    # mis-aligned by a step (point k against old k) the same plans are far apart: the alignment is what makes it zero
    mis = (new_w[:, :N - 1] - old_w[:, :N - 1]).norm(dim=-1)
    assert float(mis[v0 > 2.0].median()) > 0.1


def test_cost_geometry_one_point_and_huber():
    # a straight old plan at 5 m/s, the new plan its exact continuation; move one compared point sideways by d
    B, v = 1, 5.0
    t = torch.arange(1, N + 1, dtype=torch.float32) * PS.DT
    old = torch.stack([v * t, torch.zeros(N)], -1)[None]                     # old points 1..10 (0.1 .. 1.0 s)
    old_v = torch.full((B, N), v)
    new = torch.stack([v * (t + PS.DT), torch.zeros(N)], -1)[None]           # new points 1..10 (0.2 .. 1.1 s on the old clock)
    new_v = torch.full((B, N), v)
    ok = torch.ones(B, dtype=torch.bool)
    W, R = 0.01, 0.75
    assert float(C.promise_cost(new, new_v, old, old_v, ok, W, R, 0.2)["cost"]) < 1e-12       # (float rounding of 0.1 k)
    for k in (1, 4, 9):                                                      # the new plan's point k (0.1 k s), compared with old k+1
        for d in (0.05, 0.3, 1.5):
            nw = new.clone(); nw[0, k - 1, 1] += d
            o = C.promise_cost(nw, new_v, old, old_v, ok, W, R, 0.2)
            x = d / (R * 0.1 * k)
            h = x * x if x <= 1 else 2 * x - 1
            assert math.isclose(float(o["cost"]), W * h / (N - 1), rel_tol=1e-5)
            assert math.isclose(float(o["dp"][0, k - 1]), d, rel_tol=1e-5) and float(o["dp"].sum()) == pytest.approx(d, rel=1e-4)
    # a 0.3 m gap at the promise's 0.5 s point (the new plan's 0.4 s point) is exactly the reference: huber(1) = 1
    nw = new.clone(); nw[0, 3, 1] += 0.3
    assert math.isclose(float(C.promise_cost(nw, new_v, old, old_v, ok, W, R, 0.2)["cost"]), W / (N - 1), rel_tol=1e-5)
    # the speed gap counts as v_s x dv metres
    nv = new_v.clone(); nv[0, 8] += 1.0
    o = C.promise_cost(new, nv, old, old_v, ok, W, R, 0.2)
    x = 0.2 / (R * 0.9)
    assert math.isclose(float(o["cost"]), W * x * x / (N - 1), rel_tol=1e-5) and math.isclose(float(o["dv"][0, 8]), 1.0, rel_tol=1e-6)
    # not charged where there is no promise
    assert float(C.promise_cost(nw, nv, old, old_v, ~ok, W, R, 0.2)["cost"]) == 0.0


def test_gate_start_restart_contact_push_scene():
    G = C.PromiseGate(4, "cpu")
    off = torch.zeros(4)
    assert not bool(G.charge(off, 0.05).any())                               # the start: no promise
    G.after(torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    assert bool(G.charge(off, 0.05).all())                                   # a plan drawn: the next one is charged
    assert G.charge(torch.tensor([0.0, 0.06, 0.05, 0.2]), 0.05).tolist() == [True, False, True, False]   # pushed off its point
    G.after(torch.tensor([True, False, False, False]), torch.tensor([False, True, False, False]))
    assert G.charge(off, 0.05).tolist() == [False, False, True, True]        # a restart / a contact in the last decision
    G.after(torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    assert bool(G.charge(off, 0.05).all())                                   # one decision later both are promises again
    # a car slower than v_min re-plans freely (standing, stuck): backing out of a box costs nothing
    v = torch.tensor([0.0, 0.29, 0.3, -2.0])
    assert G.charge(off, 0.05, v, 0.3).tolist() == [False, False, True, True]
    G.after(torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    G.scene(torch.tensor([1, 3]))
    assert G.charge(off, 0.05).tolist() == [False, True, False, True]        # a scene comes with its last plan


def test_shaping_reads_the_car_in_commit_mode():
    # PlanReward._phi_pts with phi_car: -(the car's distance left) / lap, whatever the plan's points say
    from types import SimpleNamespace
    from f1sim.learn.plan_reward import PlanReward
    pr = PlanReward.__new__(PlanReward)
    pr.env = SimpleNamespace(sim=SimpleNamespace(tid=torch.zeros(3, dtype=torch.long)))
    pr.lid = torch.arange(3)
    pr.goal = SimpleNamespace(lap=torch.tensor([40.0]), remaining=lambda xy, tid: torch.full(xy.shape[:-1], 5.0))
    pr.phi_k, pr.phi_car = None, False
    pts = torch.zeros(3, N, 2)
    d_car = torch.tensor([10.0, 20.0, 30.0])
    assert torch.allclose(pr._phi_pts(pts, d_car), torch.full((3,), -5.0 / 40.0))           # the plan's points (as before)
    pr.phi_car = True
    assert torch.allclose(pr._phi_pts(pts, d_car), -d_car / 40.0)                             # the car's own progress


# ------------------------------------------------------------------------------------------------------------ fork parity
@pytest.mark.skipif(not os.path.exists(PARENT), reason="race33b u975 is not on this machine")
def test_fork_mean_is_the_parents_bit_for_bit(lim):
    from f1sim.learn.plan_policy import Actor2, load_actor2, plan_v2
    from f1sim.learn.plan_rl import init_actor
    ck = torch.load(PARENT, map_location="cpu", weights_only=False)
    a = ck["args"]
    parent = load_actor2(ck, "cpu", check_mount=False)                       # what the exam / console load
    # the actor as plan_rl builds it for a fork with race33b's flags (pred_kin: any table, the checkpoint's buffers win)
    _lk = PS.Limits.from_nominal(*M.config_lut()[::-1], M.GRIP_LEVELS.index(1.0), "cpu", margin=1.0)
    fork = Actor2(int(ck["beams"]), std0=float(a.get("std0", 0.6)), memory=int(a["memory"]), aux_dim=0, mem_front=bool(a["mem_front"]),
                  mem_value=float(a["mem_value"]) > 0, mem_keep_s=0.0, branches=1, box_memory=0, aux_in=False,
                  pred_opp=bool(a["pred_opp"]), pred_smooth=bool(a["pred_smooth"]), pred_kin=(_lk.v_grid, _lk.k_tyre))
    assert init_actor(fork, ck["actor"]) == ""
    with torch.no_grad():
        fork.log_std.copy_(C.init_log_std(3, 0.2, 0.2, 0.1))              # --commit-plan's start (only the noise changes)
        fork.log_std_floor.copy_(C.floor_vec(3, 0.15, 0.15, 0.05))
    fork.eval()
    g = torch.Generator().manual_seed(5)
    B = 32
    scans = torch.rand(B, 3, int(ck["beams"]), generator=g)
    state = torch.randn(B, int(fork.norm.mean.numel()), generator=g)
    h = torch.randn(B, fork.memory, generator=g) * 0.3
    with torch.no_grad():
        m0, _, v0 = parent(scans, state, h)
        m1, ls1, v1 = fork(scans, state, h)
    assert torch.equal(m0, m1) and torch.equal(v0, v1) and torch.equal(parent.h_out, fork.h_out)
    assert not torch.equal(ls1[0], parent.log_std)                            # the noise is the mode's
    # a checkpoint of the mode (the same keys and shapes) loads in the exam's / console's loader and drives the same mean
    ck2 = dict(ck, actor=fork.state_dict(), args=dict(a, commit_plan=0.01, tail_consistency=0.0))
    again = load_actor2(ck2, "cpu", check_mount=False)
    with torch.no_grad():
        m2, _, _ = again(scans, state, h)
    assert torch.equal(m0, m2)
