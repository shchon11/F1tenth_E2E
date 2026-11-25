"""(11-25) plan_rl --pred-clear, the parts that need no simulator: the trunk's 10 clearances between where the PREVIOUS plan
puts our car at t_k = 0.1 k s and where the head predicts the other car(s) then (`Actor2.clear_in`, built on the count-agnostic
`plan_car_cost.clearance` / `car_cover`); the previous plan's time alignment (`PlanObs2.set_plan2` / `prev_poses`); 'far'
without a car or a previous plan; a fork that drives exactly as its parent; the update reading exactly the rollout's input;
the moving-car read (`plan_car_cost.moving_read`) the log / exam table report."""
import math
import types

import numpy as np
import pytest
import torch

from f1sim import map_controller as M
from f1sim import plan_steps as PS
from f1sim.learn import plan_car_cost as C
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import (Actor2, N_SCANS, STATE_DIM2, PRED_N, PRED_OUT, PRED_SCALE, PRED_IN, PC_N, PC_FAR, PC_SCALE,
                                     PC_LO, PC_HI, PlanObs2, load_actor2, prev_plan_poses, pred_headings)

BEAMS = 90
MEM = 16
L, W, MARGIN = 0.58, 0.31, 0.05
GEOM = (L, W, MARGIN)
R = C.disc_radius(L, W)


def _lim():
    lut, nominal = M.config_lut()
    return PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu")


def _obs(n):
    """a PlanObs2 without an env: only what set_plan2 / reset_rows / prev_poses touch"""
    o = PlanObs2.__new__(PlanObs2)
    o.rel = torch.zeros(n, PS.N_STEP - 1, 2); o.rel_v = torch.zeros(n, PS.N_STEP); o.rel_psi = torch.zeros(n, PS.N_STEP - 1)
    o.prev_ok = torch.zeros(n, dtype=torch.bool); o.cmd_d = torch.zeros(n); o.cmd_v = torch.zeros(n)
    o.prev_branch = None; o.pace_idx = None; o.mem_h = None
    o.scans = torch.zeros(n, N_SCANS, BEAMS)
    st = torch.zeros(n, 7)
    o.env = types.SimpleNamespace(sim=types.SimpleNamespace(state=st), last_result=types.SimpleNamespace(scan=torch.ones(n, BEAMS)),
                                  learner_ids=torch.arange(n))
    o.lid = torch.arange(n); o.range = 1.0
    o.plan_xy = torch.zeros(n, PS.N_STEP, 2); o.plan_v = torch.zeros(n, PS.N_STEP)
    return o


def _q(pts, logit=10.0):
    """a raw prediction (n, PRED_OUT) from points (n, 10, 2) [m] in our frame and the 'there' logit"""
    n = pts.shape[0]
    return torch.cat([(pts / PRED_SCALE).reshape(n, -1), torch.full((n, 1), float(logit))], 1)


def _actor(seed=0, **kw):
    torch.manual_seed(seed)
    return Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True, pred_clear=True, pc_geom=GEOM, **kw).eval()


def _straight_prev(n, v):
    """the previous plan's poses at t_1..t_10 for a car going straight at v [m/s] (frame of the car now)"""
    k = torch.arange(1, PC_N + 1, dtype=torch.float32)
    poses = torch.zeros(n, PC_N, 3)
    poses[..., 0] = v * PS.DT * k
    return poses, torch.ones(n, dtype=torch.bool)


# ------------------------------------------------------------------------------------------------ the previous plan's timing
def test_previous_plan_is_aligned_to_the_decision_times():
    lim = _lim()
    n = 4
    g = torch.Generator().manual_seed(1)
    act = torch.tanh(torch.randn(n, PS.ACT_DIM, generator=g) * 0.7)
    p = PS.roll(act, torch.tensor([1.0, 3.0, 5.0, 7.0]), torch.zeros(n), lim)
    obs = _obs(n)
    obs.set_plan2(p)
    poses, ok = obs.prev_poses()
    assert poses.shape == (n, PC_N, 3) and bool(ok.all())
    # drawn 0.1 s ago: its point j is at t - 0.1 + 0.1 j, so t_k (k = 1..9) is its point k + 1, in the frame of its point 1 (the
    # pose it said the car would be in now)
    i1 = PS.SUB
    x1, y1, h1 = p["x"][:, i1], p["y"][:, i1], p["psi"][:, i1]
    for k in range(1, PC_N):
        j = (k + 1) * PS.SUB
        dx, dy = p["x"][:, j] - x1, p["y"][:, j] - y1
        want = torch.stack([dx * torch.cos(h1) + dy * torch.sin(h1), -dx * torch.sin(h1) + dy * torch.cos(h1)], -1)
        assert torch.allclose(poses[:, k - 1, :2], want, atol=1e-5), k
        assert torch.allclose(poses[:, k - 1, 2], p["psi"][:, j] - h1, atol=1e-6), k
    # t_10 (after the plan's end): its last point carried on 0.1 s along its last heading at its last speed
    h9 = poses[:, 8, 2]; v9 = p["pv"][:, -1]
    ext = poses[:, 8, :2] + (v9 * PS.DT)[:, None] * torch.stack([torch.cos(h9), torch.sin(h9)], -1)
    assert torch.allclose(poses[:, 9, :2], ext, atol=1e-6) and torch.allclose(poses[:, 9, 2], h9)


def test_a_steady_straight_plan_puts_the_car_at_v_t():
    lim = _lim()
    n = 3
    v = torch.tensor([2.0, 4.0, 6.0])
    # an action that holds the speed and goes straight: zero turn, each step's speed at the speed now
    a = torch.zeros(n, PS.ACT_DIM)
    p = PS.roll(a, v, torch.zeros(n), lim)
    assert torch.allclose(p["pv"], v[:, None].expand(-1, PS.N_STEP), atol=1e-4)        # (the zero action holds the speed)
    obs = _obs(n)
    obs.set_plan2(p)
    poses, _ = obs.prev_poses()
    k = torch.arange(1, PC_N + 1, dtype=torch.float32)
    assert torch.allclose(poses[..., 0], v[:, None] * PS.DT * k, atol=1e-4)
    assert torch.allclose(poses[..., 1:], torch.zeros(n, PC_N, 2), atol=1e-5)


def test_a_restart_has_no_previous_plan_until_one_is_drawn():
    lim = _lim()
    obs = _obs(3)
    assert not bool(obs.prev_ok.any())                                   # a start: nothing drawn yet
    obs.set_plan2(PS.roll(torch.zeros(3, PS.ACT_DIM), torch.full((3,), 3.0), torch.zeros(3), lim))
    assert bool(obs.prev_ok.all())
    obs.reset_rows(torch.tensor([1]))
    assert obs.prev_ok.tolist() == [True, False, True] and float(obs.rel_psi[1].abs().sum()) == 0.0
    obs.set_plan2(PS.roll(torch.zeros(3, PS.ACT_DIM), torch.full((3,), 3.0), torch.zeros(3), lim), torch.tensor([False, True, False]))
    assert bool(obs.prev_ok.all())


# ------------------------------------------------------------------------------------------------ the clearance
def test_a_car_moving_away_opens_up_and_a_standing_one_closes_in():
    a = _actor()
    prev = _straight_prev(2, 2.0)                                        # we go on at 2 m/s
    k = torch.arange(1, PRED_N + 1, dtype=torch.float32)
    pts = torch.zeros(2, PRED_N, 2)
    pts[0, :, 0] = 2.6 + 3.0 * PS.DT * k                                 # 2.6 m ahead now, going away at 3 m/s
    pts[1, :, 0] = 2.6                                                   # 2.6 m ahead now, standing
    c = a.clear_in(_q(pts), prev, psi=torch.zeros(2, PRED_N)) * PC_SCALE
    assert bool((c[0].diff() > 0).all()) and bool((c[1].diff() < 0).all())
    assert bool((c[0] > c[1]).all())
    # nose to tail on one line: our front disc against its rear disc, |gap| - 2 L / 3 - 2 r + margin
    gap = pts[:, :, 0] - prev[0][:, :, 0]
    want = (gap - 2.0 * L / 3.0 - 2.0 * R + MARGIN).clamp(PC_LO, PC_HI)
    assert torch.allclose(c, want, atol=1e-5)
    assert float(c[1, -1]) < 0.0 <= float(c[1, 0])                       # the standing car is run into within the second


def test_the_smooth_heads_own_heading_places_the_footprint():
    a = _actor(pred_smooth=True, pred_kin=_kin())
    sc, st, prev = _inputs(5, 1)
    with torch.no_grad():
        a(sc, st, torch.zeros(5, MEM), prev=prev)
    psi = a.pred.psi_out
    assert psi.shape == (5, PRED_N)
    assert torch.equal(a.pc_out, a.clear_in(a.pred_out, prev, psi=psi))        # the clearances used the head's own heading
    # it is the heading at each step's END (the path is exact arcs: the chord between two points runs along the mean of
    # the headings at its ends)
    pts = a.pred_out[:, :2 * PRED_N].reshape(-1, PRED_N, 2) * PRED_SCALE
    seg = pts[:, 1:] - pts[:, :-1]
    chord = torch.atan2(seg[..., 1], seg[..., 0])
    mid = 0.5 * (psi[:, 1:] + psi[:, :-1])
    d = torch.remainder(chord - mid + math.pi, 2 * math.pi) - math.pi
    assert float(d[seg.norm(dim=-1) > 1e-3].abs().max()) < 1e-4


def test_side_by_side_is_clear_at_body_width_plus_margin():
    a = _actor()
    prev = _straight_prev(1, 3.0)
    for off, sign in ((0.40, 1.0), (0.30, -1.0)):
        pts = prev[0][:, :, :2].clone(); pts[..., 1] = off                # alongside, the same speed, `off` m to the left
        c = a.clear_in(_q(pts), prev, psi=torch.zeros(1, PRED_N)) * PC_SCALE
        assert torch.allclose(c, torch.full_like(c, off - 2.0 * R + MARGIN), atol=1e-5)
        assert math.copysign(1.0, float(c[0, 0])) == sign                # 0.40 m apart: clear; 0.30 m: an overlap


def test_min_over_cars_and_empty_slots_change_nothing():
    """the interface: more predicted cars are more elements in the same minimum, an empty slot (confidence 0) is none"""
    prev = _straight_prev(1, 3.0)
    me_xy, me_psi = prev[0][..., :2], prev[0][..., 2]
    k = torch.arange(1, PRED_N + 1, dtype=torch.float32)
    car_a = torch.zeros(1, PRED_N, 2); car_a[..., 0] = 3.0 + 1.0 * PS.DT * k     # ahead, slower than us
    car_b = torch.zeros(1, PRED_N, 2); car_b[..., 0] = 0.3 * k; car_b[..., 1] = 0.5   # alongside on the left
    psi = torch.zeros(1, PRED_N)

    def one(xy, conf=1.0):
        el = C.car_cover(torch, xy[:, None], psi[:, None], L, W, conf=torch.full((1, 1), conf))
        return C.clearance(torch, me_xy, me_psi, *el, L, W, MARGIN)
    two = C.car_cover(torch, torch.stack([car_a, car_b], 1), torch.stack([psi, psi], 1), L, W, conf=torch.ones(1, 2))
    both = C.clearance(torch, me_xy, me_psi, *two, L, W, MARGIN)
    assert two[0].shape == (1, 6, PRED_N, 2) and two[2].shape == (1, 6)
    assert torch.allclose(both, torch.minimum(one(car_a), one(car_b)))
    assert not torch.allclose(one(car_a), one(car_b))                    # (each car is the nearest somewhere)
    # a third slot, empty (confidence 0) or below 0.5: the same minimum; the input is 10 numbers whatever the count
    xyz = torch.stack([car_a, car_b, torch.zeros(1, PRED_N, 2)], 1)
    for conf3 in (0.0, 0.4):
        three = C.car_cover(torch, xyz, torch.zeros(1, 3, PRED_N), L, W, conf=torch.tensor([[1.0, 1.0, conf3]]))
        assert torch.equal(C.clearance(torch, me_xy, me_psi, *three, L, W, MARGIN), both)
    # per-t confidence ((B, M, K)) works the same; points of radius 0 (a predicted scan) are elements too
    el = C.car_cover(torch, car_a[:, None], psi[:, None], L, W, conf=torch.ones(1, 1, PRED_N))
    assert torch.allclose(C.clearance(torch, me_xy, me_psi, *el, L, W, MARGIN), one(car_a))
    pt = car_a[:, None]                                                    # one point per t_k
    c_pt = C.clearance(torch, me_xy, me_psi, pt, 0.0, torch.ones(1, 1), L, W, MARGIN)
    assert torch.allclose(c_pt, ((car_a[..., 0] - me_xy[..., 0]) - L / 3.0 - R + MARGIN).clamp(PC_LO, PC_HI), atol=1e-5)
    # numpy gives the same
    npc = C.clearance(np, me_xy.numpy(), me_psi.numpy(), *(x.numpy() if torch.is_tensor(x) else x for x in two), L, W, MARGIN)
    assert np.allclose(npc, both.numpy(), atol=1e-6)


def test_far_without_a_car_or_without_a_previous_plan():
    a = _actor()
    prev = _straight_prev(3, 0.0)                                        # we stand
    pts = torch.zeros(3, PRED_N, 2); pts[..., 0] = 0.5                    # right on top of us
    c = a.clear_in(_q(pts, logit=10.0), prev, psi=torch.zeros(3, PRED_N))
    assert bool((c < 0).all())
    # P(there) < 0.5: nothing counts
    c = a.clear_in(_q(pts, logit=-0.1), prev, psi=torch.zeros(3, PRED_N))
    assert torch.equal(c, torch.full_like(c, PC_FAR / PC_SCALE))
    # no previous plan (a start / a restart): far, whatever is predicted
    ok = torch.tensor([True, False, True])
    c = a.clear_in(_q(pts, logit=10.0), (prev[0], ok), psi=torch.zeros(3, PRED_N))
    assert torch.equal(c[1], torch.full((PC_N,), PC_FAR / PC_SCALE)) and bool((c[[0, 2]] < 0).all())
    # and through the policy's own path: a fresh PlanObs2 has none
    obs = _obs(2)
    sc, st = torch.rand(2, N_SCANS, BEAMS), torch.randn(2, STATE_DIM2)
    a(sc, st, torch.zeros(2, MEM), prev=obs.prev_poses())
    assert torch.equal(a.pc_out, torch.full((2, PC_N), PC_FAR / PC_SCALE))
    with pytest.raises(ValueError):
        a(sc, st, torch.zeros(2, MEM))                                    # a pred_clear actor needs its last plan


# ------------------------------------------------------------------------------------------------ fork, load, update
def _parent(smooth=False):
    torch.manual_seed(0)
    kw = dict(pred_smooth=True, pred_kin=_kin()) if smooth else {}
    a = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True, **kw).eval()
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.05 * torch.randn_like(p))
        a.norm.mean.normal_(); a.norm.var.uniform_(0.5, 2.0)
    return a


def _kin():
    v = torch.linspace(0.0, 12.0, 241)
    return v, torch.minimum(torch.full_like(v, 2.0), 6.0 / v.clamp_min(1e-3) ** 2)


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    prev = (torch.randn(n, PC_N, 3, generator=g) * torch.tensor([2.0, 0.5, 0.2]), torch.rand(n, generator=g) > 0.2)
    return torch.rand(n, N_SCANS, BEAMS, generator=g), torch.randn(n, STATE_DIM2, generator=g), prev


@pytest.mark.parametrize("smooth", [False, True])
def test_a_fork_with_the_clearances_drives_exactly_as_its_parent(smooth):
    base = _parent(smooth)
    kw = dict(pred_smooth=True, pred_kin=_kin()) if smooth else {}
    fork = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True, pred_clear=True, pc_geom=GEOM, **kw)
    added = PR.init_actor(fork, base.state_dict())
    fork.eval()
    assert added == "" and fork.trunk[0].in_features == base.trunk[0].in_features + PC_N
    assert bool((fork.trunk[0].weight[:, -PC_N:] == 0).all())
    assert torch.equal(fork.trunk[0].weight[:, :-PC_N], base.trunk[0].weight)
    hb = hf = torch.zeros(5, MEM)
    for k in range(6):
        sc, st, prev = _inputs(5, k)
        mb, lb, vb = base(sc, st, hb); hb = base.h_out
        mf, lf, vf = fork(sc, st, hf, prev=prev); hf = fork.h_out
        # the same up to the sum's rounding (a wider matrix product groups its terms differently: ~1 ulp); the memory exactly
        assert torch.allclose(mb, mf, rtol=0, atol=1e-6) and torch.allclose(vb, vf, rtol=0, atol=5e-6) and torch.equal(hb, hf)
        assert fork.pc_out.shape == (5, PC_N) and bool((fork.pc_out < PC_FAR / PC_SCALE).any() or True)
    # once the plan's weights on the clearances move, they matter (they are wired in)
    with torch.no_grad():
        fork.trunk[0].weight[:, -PC_N:].normal_(0.0, 0.5)
    sc, st, prev = _inputs(5, 9)
    prev[0][..., :2] = 0.0                                                # sitting on the predicted car: clearances < far
    mf2, _, _ = fork(sc, st, torch.zeros(5, MEM), prev=prev)
    mb2, _, _ = base(sc, st, torch.zeros(5, MEM))
    assert (mf2 - mb2).abs().max() > 1e-4


def test_a_smooth_conversion_with_the_clearances_zeroes_the_point_columns_only():
    """an old-head parent -> a smooth + clearance fork: the 20 point columns zero, P(there)'s column kept, the 10 new ones zero"""
    torch.manual_seed(0)
    old = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True).eval()
    with torch.no_grad():
        old.trunk[0].weight.normal_(0.0, 0.1)
    fork = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True, pred_smooth=True, pred_kin=_kin(),
                  pred_clear=True, pc_geom=GEOM)
    assert PR.init_actor(fork, old.state_dict()) == "smooth"
    w, w0 = fork.trunk[0].weight, old.trunk[0].weight
    n0 = w0.shape[1]
    assert torch.equal(w[:, :n0 - 2 * PRED_N], w0[:, :n0 - 2 * PRED_N])          # P(there)'s column (the one before) kept
    assert bool((w[:, n0 - 2 * PRED_N:] == 0).all())                            # the points and the clearances: zero


def test_load_and_init_refusals():
    a = _actor()
    with torch.no_grad():
        a.trunk[0].weight[:, -PC_N:].normal_()
    ck = {"actor": a.state_dict(), "args": {"memory": MEM, "mem_front": True, "std0": 0.6}, "beams": BEAMS}
    b = load_actor2(ck, "cpu", check_mount=False)
    assert b.pred_clear and b.pc_dims == pytest.approx(GEOM) and torch.equal(b.trunk[0].weight, a.trunk[0].weight)
    no = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True)
    with pytest.raises(ValueError):
        PR.init_actor(no, a.state_dict())                                 # a clearance checkpoint needs --pred-clear
    # a fork of a clearance checkpoint keeps ITS OWN geometry (this run's car and margin)
    c = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True, pred_clear=True, pc_geom=(0.6, 0.3, 0.1))
    PR.init_actor(c, a.state_dict())
    assert c.pc_dims == (0.6, 0.3, 0.1) and torch.allclose(c.pc_geom, torch.tensor([0.6, 0.3, 0.1]))
    with pytest.raises(ValueError):
        Actor2(BEAMS, memory=MEM, mem_front=True, pred_clear=True, pc_geom=GEOM)   # needs the prediction head


def test_the_update_reads_exactly_the_rollouts_clearances():
    """plan_rl's update replays the memory and re-runs the head (whose own optimizer moves it between minibatches); the
    clearances are taken from the rollout's store, so the trunk reads what the rollout read -- a recompute would not"""
    a = _actor(seed=3)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.05 * torch.randn_like(p))
        a.trunk[0].weight[:, -PC_N:].normal_(0.0, 0.3)
    T, n = 6, 4
    h = torch.zeros(n, MEM)
    X, S, H, PCL, MEAN, PREV = [], [], [], [], [], []
    for t in range(T):                                                    # the rollout (with a restart at t = 3, row 1)
        sc, st, prev = _inputs(n, 20 + t)
        if t == 3:
            prev[1][1] = False
        H.append(h.clone())
        with torch.no_grad():
            m, _, _ = a(sc, st, h, prev=prev)
        h = a.h_out
        X.append(sc); S.append(st); PCL.append(a.pc_out.clone()); MEAN.append(m); PREV.append(prev)
    assert torch.equal(PCL[3][1], torch.full((PC_N,), PC_FAR / PC_SCALE))
    bX, bS, bPCL = torch.cat(X), torch.cat(S), torch.cat(PCL)

    def update_means(pc_of):
        f = a.features(bX, bS).view(T, n, -1)
        hh, hs = H[0], []
        for i in range(T):
            hh = a.mem_cell(f[i], hh); hs.append(hh)
        hcat, fflat = torch.cat(hs, 0), f.reshape(T * n, -1)
        q = a.pred_q(bX, fflat, hcat)
        return a.heads(a.mem_trunk(fflat, hcat, a.pred_feed(q), pc_of(q)))[0]
    with torch.no_grad():
        m_upd = update_means(lambda q: bPCL)
    assert torch.allclose(m_upd, torch.cat(MEAN), atol=1e-5)               # PPO's ratio starts at 1
    # the head moves (its own optimizer, `pred_step`): the stored input is still the rollout's; a recompute is not
    with torch.no_grad():
        for p in a.pred.parameters():
            p.add_(0.2 * torch.randn_like(p))
        q2 = a.pred_q(bX, a.features(bX, bS), torch.cat([H[0]] * T))      # (the memory as stored: only the head moved)
        again = a.clear_in(q2, (torch.cat([p_[0] for p_ in PREV]), torch.cat([p_[1] for p_ in PREV])))
    assert not torch.equal(again, bPCL)


def test_the_update_reads_the_rollouts_prediction_inputs_too():
    """(11-25) the trunk's 21 prediction inputs (`pred_feed` of the head's output) are stored at the rollout as well: after the
    head's own optimizer moves it, the update (plan_rl's `actor_on`: the memory replayed, the head re-run for its own loss)
    still feeds the trunk what the rollout did, so the policy's log-probabilities at the start of the update are the
    rollout's; the prediction made again would not be"""
    a = _actor(seed=4)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(0.05 * torch.randn_like(p))
        a.trunk[0].weight[:, -(PRED_IN + PC_N):].normal_(0.0, 0.3)     # the plan reads the prediction and the clearances
    T, n = 5, 4
    h = torch.zeros(n, MEM)
    X, S, PIN, PCL, MEAN = [], [], [], [], []
    for t in range(T):                                                    # the rollout, as plan_rl stores it
        sc, st, prev = _inputs(n, 40 + t)
        with torch.no_grad():
            m, _, _ = a(sc, st, h, prev=prev)
        h = a.h_out
        X.append(sc); S.append(st); MEAN.append(m)
        PIN.append(a.pred_feed(a.pred_out).clone()); PCL.append(a.pc_out.clone())
    bX, bS, bPIN, bPCL = torch.cat(X), torch.cat(S), torch.cat(PIN), torch.cat(PCL)
    assert bPIN.shape == (T * n, PRED_IN)

    def update(stored):
        f = a.features(bX, bS).view(T, n, -1)
        hh, hs = torch.zeros(n, MEM), []
        for i in range(T):
            hh = a.mem_cell(f[i], hh); hs.append(hh)
        hcat, fflat = torch.cat(hs, 0), f.reshape(T * n, -1)
        q = a.pred_q(bX, fflat, hcat)                                     # (the head's own loss still reads this)
        pin = bPIN if stored else a.pred_feed(q)
        return a.heads(a.mem_trunk(fflat, hcat, pin, bPCL))[0], a.pred_feed(q)
    with torch.no_grad():
        m0, rec0 = update(True)
        assert torch.allclose(m0, torch.cat(MEAN), atol=1e-5) and torch.allclose(rec0, bPIN, atol=1e-5)
        for p in a.pred.parameters():                                     # a step of the head's own optimizer
            p.add_(0.05 * torch.randn_like(p))
        m1, rec1 = update(True)
        m2, _ = update(False)
    assert torch.allclose(m1, torch.cat(MEAN), atol=1e-5)                # stored: the rollout's input, the rollout's plan
    assert float((rec1 - bPIN).abs().mean()) > 1e-3                      # the prediction made again has drifted
    assert float((m2 - torch.cat(MEAN)).abs().max()) > 1e-4              # and would have moved the policy's output


# ------------------------------------------------------------------------------------------------ the moving-car read
@pytest.mark.parametrize("xp", [np, torch])
def test_moving_read(xp):
    """a car 2 m ahead: going on at our speed -> the plan through its spot now, clear of its future (wanted); standing -> the
    plan meets its future too; a plan 1 m to the side -> not through its spot now"""
    T, K = 1, 10
    kofs = list(range(1, K + 1))

    def run(op_speed, plan_y):
        tk = np.arange(0, K + 1) * 0.1
        me = np.zeros((T, 1, K + 1, 2)); me[0, 0, :, 0] = 3.0 * tk; me[0, 0, 1:, 1] = plan_y
        mp = np.zeros((T, 1, K + 1))
        op = np.zeros((K + 2, 1, 2)); op[:, 0, 0] = 2.0 + op_speed * np.arange(0, K + 2) * 0.1
        # the other car's poses at each decision (only T + 1 = 2 rows are a rollout; give the whole second as 'rows')
        TT = K
        me_ = np.repeat(me, TT, 0); mp_ = np.repeat(mp, TT, 0)
        args = [me_, mp_, op[:TT + 1], np.zeros((TT + 1, 1)), np.ones((TT + 1, 1), bool), np.zeros((TT, 1), bool)]
        if xp is torch:
            args = [torch.as_tensor(x) for x in args]
        o = C.moving_read(xp, *args, kofs, L, W, MARGIN)
        return {k_: bool(np.asarray(v)[0, 0]) for k_, v in o.items()}
    r = run(3.0, 0.0)
    assert r["ahead"] and r["now"] and not r["fut"] and r["want"]
    r = run(0.0, 0.0)
    assert r["ahead"] and r["now"] and r["fut"] and not r["want"]
    r = run(3.0, 1.0)
    assert r["ahead"] and not r["now"] and not r["want"]


def test_pred_headings_from_points():
    pts = torch.zeros(1, PRED_N, 2)
    pts[0, :, 0] = torch.arange(PRED_N).float(); pts[0, :, 1] = torch.arange(PRED_N).float()
    assert torch.allclose(pred_headings(pts), torch.full((1, PRED_N), math.pi / 4))
    psi = torch.randn(1, PRED_N)
    assert torch.equal(pred_headings(pts, psi), psi)


# ------------------------------------------------------------------------------------------------ a copy at the other wheel
def test_a_pred_clear_copy_drives_the_other_car_and_its_scene_state_comes_back(monkeypatch):
    """--self-opp with a --pred-clear policy: its frozen copies read their own last plan (`SelfOpponents.plan`), and a scene
    puts that back (`snapshot` / `restore`: the headings and whether there is a last plan)"""
    B = 8

    class Obs(PlanObs2):
        def __init__(self, env, lim, goal=None, rows=None):
            n = int(rows.numel())
            self.env, self.rows, self.lid = env, rows, rows
            self.rel = torch.zeros(n, PS.N_STEP - 1, 2); self.rel_v = torch.zeros(n, PS.N_STEP)
            self.rel_psi = torch.zeros(n, PS.N_STEP - 1); self.prev_ok = torch.zeros(n, dtype=torch.bool)
            self.cmd_d = torch.zeros(n); self.cmd_v = torch.zeros(n); self.prev_branch = None; self.pace_idx = None
            self.plan_xy = torch.zeros(n, PS.N_STEP, 2); self.plan_v = torch.zeros(n, PS.N_STEP); self.mem_h = None
            self.scans = torch.zeros(n, N_SCANS, BEAMS)

        def _scan(self):
            return self.env.sim.state[self.rows, :1].expand(-1, BEAMS).clone().clamp(0, 1)

        def push_scan(self, rows=None):
            self.scans = torch.cat([self._scan()[:, None, :], self.scans[:, :-1]], 1)

        def state(self):
            return torch.zeros(self.scans.shape[0], STATE_DIM2)

    monkeypatch.setattr(PR, "PlanObs2", Obs)
    sim = types.SimpleNamespace(state=torch.zeros(B, 7), ideal_vcap=torch.full((B,), 5.0), absent_any=False)
    sim.state[:, 0] = torch.arange(B).float() * 10.0; sim.state[:, 3] = 3.0
    learner = torch.zeros(B, dtype=torch.bool); learner[0::2] = True
    env = types.SimpleNamespace(sim=sim, learner=learner, device=torch.device("cpu"), B=B, ideal_cmd=torch.zeros(B, 2),
                                level_seed=torch.full((B,), -1, dtype=torch.long),
                                last_result=types.SimpleNamespace(imu=torch.randn(B, 2, 6)))
    a = _actor()
    so = PR.SelfOpponents(env, _lim(), 1.0, seed=3)
    so.add(a, 500)
    for t in range(3):
        so.plan()
        cp = so.model(500)
        assert cp.pc_out.shape == (4, PC_N)
        if t == 0:                                                       # a fresh start: no last plan -> far
            assert torch.equal(cp.pc_out, torch.full((4, PC_N), PC_FAR / PC_SCALE))
        for j in range(PS.SUB):
            so.command(j); so.stepped(torch.zeros(B, dtype=torch.bool))
        so.after()
        assert bool(so.obs.prev_ok.all())
    oi = torch.tensor([0, 2])
    keep = {k: v.clone() for k, v in so.snapshot(oi).items()}
    assert {"rel_psi", "prev_ok"} <= set(keep)
    so.obs.reset_rows(torch.arange(4))
    assert not bool(so.obs.prev_ok.any())
    so.restore(oi, keep)
    assert torch.equal(so.obs.rel_psi[oi], keep["rel_psi"]) and bool(so.obs.prev_ok[oi].all())
