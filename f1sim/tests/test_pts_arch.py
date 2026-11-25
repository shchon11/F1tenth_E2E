"""(11-25) plan_rl --arch pts (`learn.pts_policy`), the parts that need no simulator: the token build (points dead-reckoned into
the frame now, masks, a fixed token count), the anchors (the previous plan moved into the frame now; a zero-initialised head
drives it), the factorised relative attention against its direct form, the free-space target's geometry (walls, boxes,
a moving car time-aligned and gated), the rollout store read back bit for bit (restarts, a stored moment put back), the
checkpoint round trip, and the collective count of the head's own optimizer on two processes."""
import math
import os
import tempfile
import types

import pytest
import torch

from f1sim import map_controller as M
from f1sim import plan_steps as PS
from f1sim.learn import pts_policy as T
from f1sim.learn.plan_policy import STATE_DIM2, N_SCANS

BEAMS = 40
ANG = torch.linspace(-2.356, 2.356, BEAMS)
MOUNT = (0.12, 0.03, 0.0)


def _lim():
    lut, nominal = M.config_lut()
    return PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu")


def _env(n, beams=BEAMS):
    """just what `PtsObs` reads: the learner rows, the scan, the car's state and odometry, the IMU"""
    ns = types.SimpleNamespace
    sim = ns(lidar=ns(angles=ANG[:beams] if beams == BEAMS else torch.linspace(-2.356, 2.356, beams)),
             cfg=ns(lidar=ns(range_max=10.0)), state=torch.zeros(n, 8), odom=ns(state=torch.zeros(n, 5)))
    return ns(learner_ids=torch.arange(n), device=torch.device("cpu"), sim=sim,
              last_result=ns(scan=torch.full((n, beams), 10.0), imu=torch.zeros(n, 2, 6)))


def _rand_in(n, beams=BEAMS, K=T.K_FRAMES, g=None):
    g = g or torch.Generator().manual_seed(1)
    return T.PtsIn(torch.rand(n, K, beams, generator=g), torch.randn(n, K, 3, generator=g) * 0.2,
                   torch.rand(n, K, generator=g) > 0.2, torch.randn(n, 10, 3, generator=g), torch.rand(n, 10, generator=g) * 5,
                   torch.rand(n, generator=g) > 0.3, torch.randn(n, 20, generator=g))


# ------------------------------------------------------------------------------------------------------------- tokens
def test_points_are_dead_reckoned_into_the_frame_now():
    """a return seen by frame j (pose T_j) lands where that world point is in the frame now (T_0), through rel_j = T_0^-1 T_j"""
    g = torch.Generator().manual_seed(0)
    geom = T.PtsGeom(ANG, MOUNT, 10.0)
    n, K = 3, T.K_FRAMES
    poses = torch.cumsum(torch.randn(n, K, 3, generator=g) * torch.tensor([0.4, 0.1, 0.1]), 1)    # world poses, age 0..4
    rel = T.se2_rel(poses[:, :1].expand(-1, K, -1), poses)
    frames = torch.ones(n, K, BEAMS)
    i = torch.randint(BEAMS, (n, K), generator=g)
    r = 1.0 + 7.0 * torch.rand(n, K, generator=g)
    frames.scatter_(2, i[..., None], (r / 10.0)[..., None])
    X, Y, rr, ok = geom.points(frames, rel, torch.ones(n, K, dtype=torch.bool))
    assert X.shape == (n, K, T.pad_beams(BEAMS)) and int(ok.sum()) == n * K       # exactly one return per frame
    a = ANG[i] + MOUNT[2]
    loc = torch.stack([MOUNT[0] + r * torch.cos(a), MOUNT[1] + r * torch.sin(a), torch.zeros_like(r)], -1)   # car frame of j
    world = T.se2_compose(poses, loc)
    want = T.se2_rel(poses[:, :1].expand(-1, K, -1), world)
    got = torch.stack([X.gather(2, i[..., None])[..., 0], Y.gather(2, i[..., None])[..., 0]], -1)
    assert torch.allclose(got, want[..., :2], atol=1e-5)


def test_token_masks_and_fixed_count():
    """N is fixed (by beams and K, never by the returns); padded beams and absent frames give no valid token; an all-empty
    decision still has one valid token (no all-masked softmax row)"""
    geom = T.PtsGeom(ANG, MOUNT, 10.0)
    front = T.PtsFront()
    n, K = 4, T.K_FRAMES
    P = T.pad_beams(BEAMS)
    assert P % T.RUN_OLD == 0 and P >= BEAMS
    frames = torch.rand(n, K, BEAMS) * 0.5
    frames[1] = 1.0                                                     # no return at all
    fok = torch.ones(n, K, dtype=torch.bool); fok[2, 1:] = False        # a restart: only the newest frame
    X, Y, r, ok = geom.points(frames, torch.zeros(n, K, 3), fok)
    tok, pos, age, val = front(X, Y, r, ok)
    N = T.n_tokens(BEAMS, K)
    assert tok.shape == (n, N, T.D_TOK) and val.shape == (n, N) and N == P // T.RUN_NEW + (K - 1) * (P // T.RUN_OLD)
    assert int(val[1].sum()) == 1                                       # the never-empty stand-in
    n_new = P // T.RUN_NEW
    assert not bool(val[2, n_new:].any()) and bool(val[2, :n_new].any())   # absent frames: no token
    # the padded tail of the newest frame (beams BEAMS..P-1): their run holds only padding when a run is all padding
    tail_runs = (P - BEAMS) // T.RUN_NEW
    if tail_runs:
        assert not bool(val[0, n_new - tail_runs:n_new].any())
    assert torch.isfinite(tok).all() and torch.allclose(age[0, :n_new], torch.zeros(n_new))
    assert torch.allclose(age[0, n_new:n_new + P // T.RUN_OLD], torch.full((P // T.RUN_OLD,), 0.1))


def test_relative_attention_factorised_equals_direct():
    torch.manual_seed(0)
    m = T.RelCross()
    with torch.no_grad():
        for p_ in m.parameters():
            p_.add_(torch.randn_like(p_) * 0.1)
    n, Q, N = 3, 11, 37
    q = torch.randn(n, Q, T.D_Q); anc = torch.randn(n, Q, 2) * 2; tok = torch.randn(n, N, T.D_TOK)
    pos = torch.randn(n, N, 2) * 4; age = torch.rand(n, N) * 0.4; val = torch.rand(n, N) > 0.4; val[:, 0] = True
    a, b = m(q, anc, tok, pos, age, val), T.rel_cross_naive(m, q, anc, tok, pos, age, val)
    assert torch.allclose(a, b, atol=2e-4, rtol=1e-4), float((a - b).abs().max())


# ------------------------------------------------------------------------------------------------------------- observations
def test_obs_chain_from_odometry_and_restart():
    """each push composes the odometry increment onto the older frames' poses: rel_j = pose(t)^-1 pose(t - j); a restart leaves
    only the newest frame"""
    n = 2
    env = _env(n)
    env.last_result.scan = torch.zeros(n, BEAMS)                        # the scan at the start: frame t = 0
    o = T.PtsObs(env, _lim())
    poses = [torch.zeros(n, 3)]
    g = torch.Generator().manual_seed(3)
    for t in range(1, 7):
        poses.append(T.se2_compose(poses[-1], torch.randn(n, 3, generator=g) * torch.tensor([0.5, 0.05, 0.1])))
        env.sim.odom.state[:, :3] = poses[-1]
        env.last_result.scan = torch.full((n, BEAMS), float(t))
        o.push_scan()
        for j in range(min(t, T.K_FRAMES - 1) + 1):
            assert torch.allclose(o.pts_rel[:, j], T.se2_rel(poses[t], poses[t - j]), atol=1e-5)
            assert bool(o.pts_fok[:, j].all())
            assert torch.allclose(o.pts_frames[:, j], torch.full((n, BEAMS), float(t - j)) / 10.0)
        assert torch.equal(o.pts_frames[:, :N_SCANS], o.scans)          # the newest frames are the scan stack's own
    o.reset_rows(torch.tensor([1]))
    assert bool(o.pts_fok[1, 0]) and not bool(o.pts_fok[1, 1:].any()) and torch.equal(o.pts_rel[1], torch.zeros(T.K_FRAMES, 3))
    assert bool(o.pts_fok[0].all())


def test_anchors_are_the_previous_plan_moved_into_the_frame_now():
    """driven exactly, the car is at the plan's 0.1 s point at the next decision: anchor k = the plan's point k + 1 there; the
    last carried on 0.1 s; the anchor action = the mean one step on (the last step neutral)"""
    lim = _lim()
    n = 3
    env = _env(n)
    o = T.PtsObs(env, lim)
    g = torch.Generator().manual_seed(5)
    u = torch.randn(n, PS.ACT_DIM, generator=g) * 0.5
    v0 = torch.tensor([2.0, 4.0, 6.0]); d0 = torch.tensor([0.0, 0.1, -0.15])
    p = PS.roll(torch.tanh(u), v0, d0, lim)
    o.u_pending = u
    o.set_plan2(p)
    assert torch.equal(o.u_prev, T.shift_anchor(u)) and torch.equal(o.u_prev[:, 9], torch.zeros(n))
    s1 = torch.stack([p["x"][:, PS.SUB], p["y"][:, PS.SUB], p["psi"][:, PS.SUB]], -1)        # where the car is at t + 0.1
    env.sim.odom.state[:, :3] = s1
    o.push_scan()
    P = o.pts_input()
    ii = T.PT_IDX
    pts = torch.stack([p["x"][:, ii], p["y"][:, ii], p["psi"][:, ii]], -1)                # the 10 points, frame t
    want = T.se2_rel(s1[:, None].expand(-1, 9, -1), pts[:, 1:])
    assert torch.allclose(P.anc[:, :9], want, atol=1e-5)
    # the same as `PlanObs2`'s own record of the plan in its 0.1 s pose's frame (what --pred-clear reads)
    assert torch.allclose(P.anc[:, :9, :2], o.rel, atol=1e-5) and torch.allclose(P.anc[:, :9, 2], o.rel_psi, atol=1e-5)
    last = pts[:, -1]
    ext = torch.stack([last[:, 0] + p["pv"][:, -1] * 0.1 * torch.cos(last[:, 2]),
                       last[:, 1] + p["pv"][:, -1] * 0.1 * torch.sin(last[:, 2]), last[:, 2]], -1)
    assert torch.allclose(P.anc[:, 9], T.se2_rel(s1, ext), atol=1e-5)
    assert bool(P.anc_ok.all()) and torch.equal(P.u_anc, T.shift_anchor(u))
    assert torch.allclose(P.anc_v, torch.cat([p["pv"][:, 1:], p["pv"][:, -1:]], 1))
    # a restart: no previous plan
    o.reset_rows(torch.tensor([0]))
    P = o.pts_input()
    assert not bool(P.anc_ok[0]) and float(P.anc[0].abs().sum()) == 0.0 and float(P.u_anc[0].abs().sum()) == 0.0


def test_zero_init_drives_the_anchor():
    """a fresh actor's mean IS the anchor action (whatever it reads, the free-space input included); rolled from the previous
    plan's 0.1 s state it is the previous plan's next 0.8 s exactly (then the neutral last step)"""
    torch.manual_seed(0)
    a = T.PtsActor(ANG, MOUNT, 10.0, memory=16)
    P = _rand_in(4)
    s = torch.randn(4, STATE_DIM2)
    m1, _, _ = a(P, s)
    assert torch.equal(m1, P.u_anc)
    fs = torch.randn(4, 10, 2)
    m2, _, _ = a(P, s, fs=fs)
    assert torch.equal(m2, P.u_anc)
    b = T.PtsActor(ANG, MOUNT, 10.0, memory=16, residual=False)
    assert torch.equal(b(P, s)[0], torch.zeros(4, PS.ACT_DIM))
    lim = _lim()
    u = torch.randn(3, PS.ACT_DIM) * 0.6
    p0 = PS.roll(torch.tanh(u), torch.tensor([1.0, 3.0, 5.0]), torch.tensor([0.05, 0.0, -0.1]), lim)
    p1 = PS.roll(torch.tanh(T.shift_anchor(u)), p0["v"][:, PS.SUB], p0["delta"][:, PS.SUB], lim)
    ii = T.PT_IDX
    q0 = torch.stack([p0["x"][:, ii], p0["y"][:, ii], p0["psi"][:, ii]], -1)
    s1 = torch.stack([p0["x"][:, PS.SUB], p0["y"][:, PS.SUB], p0["psi"][:, PS.SUB]], -1)
    want = T.se2_rel(s1[:, None].expand(-1, 9, -1), q0[:, 1:])                            # previous points 2..10
    got = torch.stack([p1["x"][:, ii], p1["y"][:, ii], p1["psi"][:, ii]], -1)
    assert torch.allclose(got[:, :8], want[:, :8], atol=1e-4)


def test_policy_parameters_leave_the_head_out_and_checkpoint_round_trip():
    torch.manual_seed(0)
    a = T.PtsActor(ANG, MOUNT, 10.0, memory=16, mem_value=True)
    with torch.no_grad():
        a.pi.lin.weight.normal_(); a.fs_in.weight.normal_()
    pol = {id(p_) for p_ in a.policy_parameters()}
    assert not any(id(p_) in pol for p_ in a.free_parameters())
    assert len(pol) + len(a.free_parameters()) == len(list(a.parameters()))
    ck = {"pts": a.config(), "actor": a.state_dict(), "args": {"std0": 0.6}}
    b = T.load_pts_actor(ck, "cpu")
    a.eval()
    P = _rand_in(2); s = torch.randn(2, STATE_DIM2)
    assert torch.equal(a(P, s)[0], b(P, s)[0]) and torch.equal(b.geom.angles, ANG) and T.is_pts(ck)


# ------------------------------------------------------------------------------------------------------------- the store
def test_store_reads_back_what_the_rollout_read():
    """the update's frames / poses / anchors / free-space inputs are bit-identical to the rollout's, through restarts and a stored
    moment put back mid-rollout (older frames from the store's table), and across the rollout's start"""
    lim = _lim()
    n, Tn = 3, 9
    env = _env(n)
    o = T.PtsObs(env, lim, odom_noise=1.0)
    st = T.PtsStore(n, o.K, "cpu")
    g = torch.Generator().manual_seed(7)
    pose = torch.zeros(n, 3)

    def drive(t):
        nonlocal pose
        pose = T.se2_compose(pose, torch.randn(n, 3, generator=g) * torch.tensor([0.4, 0.05, 0.1]))
        env.sim.odom.state[:, :3] = pose
        env.last_result.scan = torch.rand(n, BEAMS, generator=g)
        o.push_scan()
        u = torch.randn(n, PS.ACT_DIM, generator=g)
        o.u_pending = u
        o.set_plan2(PS.roll(torch.tanh(u), torch.full((n,), 3.0), torch.zeros(n), lim))
    for t in range(3):                                                  # a previous rollout: history before the start
        drive(t)
    saved = o.pts_snapshot(); saved_scans = o.scans.clone()
    st.begin(o)
    X, seen = [], []
    for t in range(Tn):
        P = o.pts_input()
        fs = torch.randn(n, 10, 2, generator=g)
        st.record(P, fs)
        X.append(o.scans.clone()); seen.append((P, fs))
        drive(t)
        if t == 2:                                                       # row 1 restarts
            rs = torch.tensor([False, True, False])
            o.reset_rows(torch.nonzero(rs)[:, 0]); st.restarted(rs, t)
        if t == 5:                                                       # row 2: a stored moment put back
            rs = torch.tensor([False, False, True])
            st.restarted(rs, t)
            idx = torch.tensor([2])
            o.scans[idx] = saved_scans[idx]
            o.pts_restore(idx, {k: v[idx] for k, v in saved.items()})
            st.loaded(idx, o, t)
    st.finish(torch.cat(X))
    for t, (P, fs) in enumerate(seen):
        rows = torch.arange(n) + t * n
        Q = st.gather(rows)
        for a_, b_ in zip(P, Q):
            assert torch.equal(a_, b_)          # every field, frames included (absent frames too: the gather is exact)
        assert torch.equal(st.fs_in(rows), fs)


# ------------------------------------------------------------------------------------------------------------- targets
def test_rect_signed_distance_against_sampling():
    g = torch.Generator().manual_seed(11)
    a = torch.tensor([[0.0, 0.0, 0.3]]); ha = torch.tensor([[0.29, 0.155]])
    for _ in range(20):
        b = torch.cat([torch.randn(1, 2, generator=g) * 0.8, torch.rand(1, 1, generator=g) * 3], 1)
        hb = 0.05 + torch.rand(1, 2, generator=g) * 0.2
        sd = float(T.rect_rect_sd(a, ha, b, hb))
        u = torch.linspace(-1, 1, 81)
        def boundary(p, h):
            e = torch.cat([torch.stack([u * h[0, 0], torch.full_like(u, h[0, 1])], -1), torch.stack([u * h[0, 0], -torch.full_like(u, h[0, 1])], -1),
                           torch.stack([torch.full_like(u, h[0, 0]), u * h[0, 1]], -1), torch.stack([-torch.full_like(u, h[0, 0]), u * h[0, 1]], -1)])
            c, s = math.cos(float(p[0, 2])), math.sin(float(p[0, 2]))
            return torch.stack([p[0, 0] + c * e[:, 0] - s * e[:, 1], p[0, 1] + s * e[:, 0] + c * e[:, 1]], -1)
        pa, pb = boundary(a, ha), boundary(b, hb)
        inside_b = T.point_rect_sd(pa[None], b, hb)[0] < 0
        inside_a = T.point_rect_sd(pb[None], a, ha)[0] < 0
        if bool(inside_a.any() or inside_b.any()):
            assert sd < 0
        else:
            brute = float(torch.cdist(pa, pb).min())
            assert abs(sd - brute) < 0.01, (sd, brute)


def test_wall_and_box_clearance():
    """our 0.58 x 0.31 footprint: a straight wall at y = 1 (signed field 1 - y), heading along it and across it; a box seen or not"""
    L, W = 0.58, 0.31
    sdf = lambda xy: 1.0 - xy[..., 1]
    poses = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, math.pi / 2], [0.0, 0.9, 0.0]]])
    w = T.wall_clear(sdf, poses, L, W)[0]
    assert torch.allclose(w, torch.tensor([1.0 - W / 2, 1.0 - L / 2, 0.1 - W / 2]), atol=1e-5)    # the last: into the wall
    bp = torch.tensor([[[1.0, 0.0, 0.0]]]); bh = torch.tensor([[[0.1, 0.1]]])
    c = T.box_clear(poses[:, :1], L, W, bp, bh, torch.tensor([[True]]))[0, 0]
    assert abs(float(c) - (1.0 - 0.1 - L / 2)) < 1e-5
    assert math.isinf(float(T.box_clear(poses[:, :1], L, W, bp, bh, torch.tensor([[False]]))[0, 0]))   # not seen: not knowable


def test_moving_car_time_aligned_and_gated():
    """anchors at x = 5 for every k; the other car drives +1 m per decision along x: point k is measured against where it is at
    t + k, not where it is at t; beyond the rollout or a restart the point is left out; a car not 'there' does not count"""
    L, W = 0.58, 0.31
    Tn, B, K = 6, 1, 10
    anc = torch.zeros(Tn, B, K, 3); anc[..., 0] = 5.0
    op = torch.zeros(Tn + 1, B, 3); op[:, 0, 0] = torch.arange(Tn + 1).float()
    ok = torch.ones(Tn + 1, B, dtype=torch.bool)
    know = torch.ones(Tn, B, dtype=torch.bool)
    brk = torch.zeros(Tn, B, dtype=torch.bool)
    car, unk = T.car_clear(anc, op, ok, know, brk, L, W)
    for k in range(1, K + 1):
        if k <= Tn:                                      # decision 0, point k: the car at x = k
            gap = abs(5.0 - k) - L
            assert not bool(unk[0, 0, k - 1])
            if gap > 0:
                assert abs(float(car[0, 0, k - 1]) - gap) < 1e-5
            else:
                assert float(car[0, 0, k - 1]) < 0        # nose to tail overlap: negative
        else:
            assert bool(unk[0, 0, k - 1])                # its pose then is not in this rollout
    brk[2, 0] = True                                     # the other car put elsewhere during decision 2
    car, unk = T.car_clear(anc, op, ok, know, brk, L, W)
    assert not bool(unk[0, 0, :2].any()) and bool(unk[0, 0, 2:].all())
    know[:] = False
    car, unk = T.car_clear(anc, op, ok, know, brk, L, W)
    assert torch.isinf(car).all() and not bool(unk.any())
    y, m = T.free_target(torch.full((Tn, B, K), 7.0), car, unk)
    assert torch.equal(y, torch.full_like(y, T.FS_HI)) and bool(m.all())


def test_free_loss_masks_and_stats():
    m = torch.tensor([[0.2, -0.1, 1.0]]); lb = torch.zeros(1, 3); y = torch.tensor([[0.1, 0.2, 5.0]])
    msk = torch.tensor([[True, True, False]])
    loss = T.free_loss(m, lb, y, msk)
    assert torch.isfinite(loss)
    st = T.free_stats(m, y, msk)
    assert st["n"] == 2 and st["n_near"] == 2 and abs(st["sign_near"] - 0.5) < 1e-6
    assert float(T.free_loss(m, lb, y, torch.zeros_like(msk))) == 0.0


# ------------------------------------------------------------------------------------------------------------- two processes
def _dist_worker(rank, world, init, out):
    import torch.distributed as dist
    from f1sim.learn.plan_rl import _CountingDist
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=world)
    cd = _CountingDist(dist)
    torch.manual_seed(0)
    a = T.PtsActor(ANG, MOUNT, 10.0, memory=8)
    opt = torch.optim.Adam(a.free_parameters(), lr=1e-3)
    n = 7 if rank == 0 else 3                            # each process its own number of decisions

    def reduce(ps):
        flat = torch.cat([p_.grad.reshape(-1) for p_ in ps]) * (0.7 if rank == 0 else 0.3)
        cd.all_reduce(flat)
        o_ = 0
        for p_ in ps:
            p_.grad = flat[o_:o_ + p_.numel()].view_as(p_).clone(); o_ += p_.numel()
    g = torch.Generator().manual_seed(100 + rank)
    for step in range(3):
        P = _rand_in(n, g=g)
        y = torch.rand(n, 10, generator=g) * 2
        msk = torch.rand(n, 10, generator=g) > 0.3
        if rank == 1 and step == 1:
            msk[:] = False                               # nothing to learn in this minibatch: it still takes part
        T.free_head_step(a, opt, P, torch.randn(n, STATE_DIM2, generator=g), y, msk, reduce_grads=reduce)
    chk = float(sum(p_.double().abs().sum() for p_ in a.free_parameters()))
    torch.save({"n": cd.n, "chk": chk}, os.path.join(out, f"r{rank}.pt"))
    dist.destroy_process_group()


def test_free_head_collectives_match_across_processes():
    """two gloo processes with different decision counts (one minibatch with no target at all): the same number of collectives,
    the same weights after"""
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory() as d:
        init = "file://" + os.path.join(d, "rdv")
        mp.spawn(_dist_worker, args=(2, init, d), nprocs=2, join=True)
        r0, r1 = (torch.load(os.path.join(d, f"r{k}.pt")) for k in (0, 1))
    assert r0["n"] == r1["n"] == 3
    assert abs(r0["chk"] - r1["chk"]) < 1e-9 * max(1.0, abs(r0["chk"]))


def test_minibatch_split_is_exact():
    """plan_rl's --dist-world split: exactly MB minibatches whatever the number of sequences (fewer sequences than MB included)"""
    for n_seq in (1, 5, 19, 32, 33, 304):
        for MB in (8, 32, 128):
            assert len(torch.randperm(n_seq).tensor_split(MB)) == MB
