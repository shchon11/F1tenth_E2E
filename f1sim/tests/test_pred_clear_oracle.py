"""(11-25) plan_rl --pred-clear-oracle (the ORACLE gate), the parts that need no simulator: the 10 clearances measured against
the other car's OWN plan (`pred_oracle`) -- equal to the geometric clearance between our previous plan and the copy's plan
points at the same times; the pace stretch (a copy at pace f is at its plan's pose f t after t s); the 'there' / previous-plan
gates and which rows take the oracle; the trunk reading it through `Actor2.forward(..., pc_over=)`; a fork that drives as its
parent whatever the oracle says."""
import copy
import math
import types

import pytest
import torch

from f1sim import map_controller as M
from f1sim import plan_steps as PS
from f1sim.learn import plan_car_cost as C
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import (Actor2, N_SCANS, STATE_DIM2, PRED_N, PRED_SCALE, PC_N, PC_FAR, PC_SCALE, PC_LO, PC_HI,
                                     PT_IDX, plan_world)
from f1sim.learn.pred_oracle import copy_plan_future, to_frame, oracle_clear, copy_oracle

BEAMS = 90
MEM = 16
L, W, MARGIN = 0.58, 0.31, 0.05
GEOM = (L, W, MARGIN)
R = C.disc_radius(L, W)


def _lim():
    lut, nominal = M.config_lut()
    return PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu")


def _kin():
    v = torch.linspace(0.0, 12.0, 241)
    return v, torch.minimum(torch.full_like(v, 2.0), 6.0 / v.clamp_min(1e-3) ** 2)


def _plans(n, seed, v0=None):
    """n copies' plans (`plan_steps.roll`) from random actions, and the world poses they were drawn at"""
    g = torch.Generator().manual_seed(seed)
    act = torch.tanh(torch.randn(n, PS.ACT_DIM, generator=g) * 0.7)
    v0 = torch.rand(n, generator=g) * 6.0 + 1.0 if v0 is None else v0
    p = PS.roll(act, v0, torch.zeros(n), _lim())
    st = torch.stack([torch.rand(n, generator=g) * 4.0 - 2.0, torch.rand(n, generator=g) * 4.0 - 2.0,
                      (torch.rand(n, generator=g) - 0.5) * 2.0 * math.pi], 1)
    return p, st


def _opp(p, st, pace, live=None):
    """a `SelfOpponents` as `copy_oracle` reads it: this decision's plans, where they start, the pace, which a copy drives"""
    n = st.shape[0]
    return types.SimpleNamespace(p=p, st_plan=st, pace=pace, live=torch.ones(n, dtype=torch.bool) if live is None else live)


def _prev(n, seed):
    """our previous plan's poses at t_1..t_10 in our frame now (as `PlanObs2.prev_poses`), all there"""
    g = torch.Generator().manual_seed(seed)
    k = torch.arange(1, PC_N + 1, dtype=torch.float32)
    v = torch.rand(n, 1, generator=g) * 6.0
    kap = (torch.rand(n, 1, generator=g) - 0.5) * 0.6
    s = v * PS.DT * k                                        # an arc at speed v, curvature kap
    th = kap * s
    x = torch.where(kap.abs() > 1e-6, torch.sin(th) / kap, s)
    y = torch.where(kap.abs() > 1e-6, (1.0 - torch.cos(th)) / kap, torch.zeros_like(s))
    return torch.stack([x, y, th], -1), torch.ones(n, dtype=torch.bool)


def _discs(xy, psi):
    c, s = torch.cos(psi)[..., None], torch.sin(psi)[..., None]
    off = torch.tensor([-L / 3.0, 0.0, L / 3.0])
    return torch.stack([xy[..., 0:1] + off * c, xy[..., 1:2] + off * s], -1)        # (..., 3, 2)


def _brute_clear(me, them_xy, them_psi):
    """the clearance by hand: every pair of our 3 discs and theirs, min |d| - 2 r + margin, clipped"""
    a, b = _discs(me[..., :2], me[..., 2]), _discs(them_xy, them_psi)              # (n, K, 3, 2)
    d = (a[..., :, None, :] - b[..., None, :, :]).norm(dim=-1).flatten(-2).amin(-1)  # (n, K)
    return (d - 2.0 * R + MARGIN).clamp(PC_LO, PC_HI)


def _ours(n, seed):
    """our cars now: world poses near the copies"""
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.rand(n, generator=g) * 4.0 - 2.0, torch.rand(n, generator=g) * 4.0 - 2.0,
                        (torch.rand(n, generator=g) - 0.5) * 2.0 * math.pi], 1)


# ------------------------------------------------------------------------------------------------ the geometry
def test_the_oracle_is_the_geometric_clearance_to_the_copys_plan_points():
    n = 64
    p, st = _plans(n, 1)
    me = _ours(n, 2)
    prev = _prev(n, 3)
    there = torch.ones(n, dtype=torch.bool)
    vals, use = copy_oracle(_opp(p, st, torch.ones(n)), torch.arange(n), me, there, torch.zeros(n, dtype=torch.bool), prev, GEOM)
    assert vals.shape == (n, PC_N) and bool(use.all())
    # at its own pace the copy is at its plan's 10 points at t_1..t_10: those, in the world, then in our frame
    xw, yw, pw = plan_world(p, st)
    ii = PT_IDX
    them_w = torch.stack([xw[:, ii], yw[:, ii]], -1)
    dx, dy = them_w[..., 0] - me[:, 0:1], them_w[..., 1] - me[:, 1:2]
    c, s = torch.cos(me[:, 2:3]), torch.sin(me[:, 2:3])
    them = torch.stack([dx * c + dy * s, -dx * s + dy * c], -1)
    them_psi = pw[:, ii] - me[:, 2:3]
    want = _brute_clear(prev[0], them, them_psi)
    assert torch.allclose(vals * PC_SCALE, want, atol=1e-4)
    assert bool((want < PC_HI).any()) and bool((want < 0).any())                   # (the test covers close calls and overlaps)
    # an overlap by more than the margin <=> the plan-car cost's overlap test, at every t_k
    ov = C.overlap(torch, prev[0][..., :2], prev[0][..., 2], them, them_psi, L, W, MARGIN)
    assert torch.equal(ov, vals * PC_SCALE < 0)
    # and exactly `Actor2.clear_in` fed the same points with P(there) = 1 and their headings
    torch.manual_seed(0)
    a = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True, pred_clear=True, pc_geom=GEOM).eval()
    q = torch.cat([(them / PRED_SCALE).reshape(n, -1), torch.full((n, 1), 30.0)], 1)
    assert torch.allclose(a.clear_in(q, prev, psi=them_psi), vals, atol=1e-6)


def test_world_to_our_frame_round_trip():
    n = 8
    me = _ours(n, 5)
    xy = torch.randn(n, PC_N, 2) * 3.0
    psi = torch.randn(n, PC_N)
    loc, lpsi = to_frame(xy, psi, me)
    c, s = torch.cos(me[:, 2:3]), torch.sin(me[:, 2:3])
    back = torch.stack([me[:, 0:1] + loc[..., 0] * c - loc[..., 1] * s, me[:, 1:2] + loc[..., 0] * s + loc[..., 1] * c], -1)
    assert torch.allclose(back, xy, atol=1e-5) and torch.allclose(lpsi + me[:, 2:3], psi, atol=1e-6)


# ------------------------------------------------------------------------------------------------ the pace stretch
def test_a_slowed_copy_is_at_its_plans_pose_f_t_after_t():
    n = 6
    p, st = _plans(n, 7)
    pace = torch.tensor([1.0, 0.5, 0.75, 0.6, 0.95, 0.25])
    xy, psi = copy_plan_future(p, st, pace)
    xw, yw, pw = plan_world(p, st)
    D = xw.shape[1] - 1                                          # dense steps of 0.025 s (N SUB)
    for i, f in enumerate(pace.tolist()):
        for k in range(1, PC_N + 1):
            tau = f * k * PS.SUB                                 # dense index of the plan's time f t_k
            j0 = min(int(math.floor(tau)), D - 1); w = tau - j0
            want = torch.stack([xw[i, j0] * (1 - w) + xw[i, j0 + 1] * w, yw[i, j0] * (1 - w) + yw[i, j0 + 1] * w])
            assert torch.allclose(xy[i, k - 1], want, atol=1e-5), (f, k)
            assert abs(float(psi[i, k - 1] - (pw[i, j0] * (1 - w) + pw[i, j0 + 1] * w))) < 1e-5
    # whole dense steps land on the plan's poses exactly: f = 0.5 -> t_k is the plan's pose at 0.05 k s (dense index 2 k)
    assert torch.allclose(xy[1, :, 0], xw[1, 2 * torch.arange(1, PC_N + 1)], atol=1e-6)
    # a straight plan at constant speed v: a copy at pace f is f v t ahead of where it drew the plan
    v = torch.full((4,), 4.0)
    ps = PS.roll(torch.zeros(4, PS.ACT_DIM), v, torch.zeros(4), _lim())
    st0 = torch.zeros(4, 3)
    f4 = torch.tensor([1.0, 0.8, 0.6, 0.5])
    xy4, psi4 = copy_plan_future(ps, st0, f4)
    t = torch.arange(1, PC_N + 1, dtype=torch.float32) * PS.DT
    assert torch.allclose(xy4[..., 0], f4[:, None] * v[:, None] * t[None, :], atol=2e-3)
    assert torch.allclose(xy4[..., 1], torch.zeros_like(xy4[..., 1]), atol=1e-5) and torch.allclose(psi4, torch.zeros_like(psi4), atol=1e-6)


def test_a_slower_copy_ahead_reads_closer():
    """the same plan driven slower: the car ahead is further back at every t_k, so our plan comes closer to it"""
    n = 1
    v = torch.full((n,), 4.0)
    ps = PS.roll(torch.zeros(n, PS.ACT_DIM), v, torch.zeros(n), _lim())
    st = torch.tensor([[2.0, 0.0, 0.0]])                        # 2 m ahead of us, the same heading
    me = torch.zeros(n, 3)
    k = torch.arange(1, PC_N + 1, dtype=torch.float32)
    prev = (torch.stack([4.0 * PS.DT * k, torch.zeros(PC_N), torch.zeros(PC_N)], -1)[None], torch.ones(n, dtype=torch.bool))
    out = []
    for f in (1.0, 0.8, 0.6):
        vals, _ = copy_oracle(_opp(ps, st, torch.full((n,), f)), torch.arange(n), me, torch.ones(n, dtype=torch.bool),
                              torch.zeros(n, dtype=torch.bool), prev, GEOM)
        out.append(vals * PC_SCALE)
    gap0 = 2.0 - 2.0 * L / 3.0 - 2.0 * R + MARGIN               # nose to tail on one line, both at 4 m/s: constant
    assert torch.allclose(out[0], torch.full_like(out[0], gap0), atol=2e-3)
    want = lambda f: (2.0 + (f - 1.0) * 4.0 * PS.DT * k - 2.0 * L / 3.0 - 2.0 * R + MARGIN).clamp(PC_LO, PC_HI)
    assert torch.allclose(out[1][0], want(0.8), atol=2e-3) and torch.allclose(out[2][0], want(0.6), atol=2e-3)
    assert bool((out[2] <= out[1] + 1e-6).all()) and bool((out[1] <= out[0] + 1e-6).all())


# ------------------------------------------------------------------------------------------------ the gates
def test_gates_there_previous_plan_scripted_and_absent():
    n = 6
    p, st = _plans(n, 11)
    me = st.clone()                                              # right on top of each copy: every clearance an overlap
    prev = (torch.zeros(n, PC_N, 3), torch.ones(n, dtype=torch.bool))
    prev[0][..., 0] = 0.05 * torch.arange(1, PC_N + 1)
    there = torch.tensor([True, False, True, True, True, False])
    live = torch.tensor([True, True, False, True, True, False])  # row 2: a scripted driver; row 5: no car (absent)
    absent = torch.tensor([False, False, False, False, False, True])
    ok = torch.tensor([True, True, True, False, True, True])     # row 3: no previous plan (a start)
    vals, use = copy_oracle(_opp(p, st, torch.ones(n), live), torch.arange(n), me, there, absent, (prev[0], ok), GEOM)
    far = torch.full((PC_N,), PC_FAR / PC_SCALE)
    assert use.tolist() == [True, True, False, True, True, True]
    assert bool((vals[0] < 0).any()) and bool((vals[4] < 0).any())      # there, a copy, a previous plan: the real overlap
    assert torch.equal(vals[1], far)                             # not 'there' (out of range / unseen): far, P = 0
    assert torch.equal(vals[3], far)                             # no previous plan: far
    assert torch.equal(vals[5], far)                             # no other car in the race: far (the oracle knows)
    # an absent car is never there, even if a stale 'there' says so; a copy's row with there=False reads far, not the plan
    v2, _ = copy_oracle(_opp(p, st, torch.ones(n), live), torch.arange(n), me, torch.ones(n, dtype=torch.bool), absent,
                        (prev[0], ok), GEOM)
    assert torch.equal(v2[5], far) and bool((v2[1] < 0).any())
    # oracle_clear's own gate: P from `there` as 1 / 0
    xy, psi = to_frame(*copy_plan_future(p, st, torch.ones(n)), me)
    assert torch.equal(oracle_clear(xy, psi, torch.zeros(n, dtype=torch.bool), prev, GEOM), far.expand(n, -1))


def test_copy_oracle_picks_each_learners_own_other_car():
    """learner i's other car is opponent row oi[i]: its plan, start pose and pace -- not row i's"""
    n = 4
    p, st = _plans(n, 13)
    pace = torch.tensor([1.0, 0.6, 0.8, 0.95])
    oi = torch.tensor([2, 0, 3, 1])
    me = _ours(n, 14)
    prev = _prev(n, 15)
    t = torch.ones(n, dtype=torch.bool); a0 = torch.zeros(n, dtype=torch.bool)
    v_all, _ = copy_oracle(_opp(p, st, pace), oi, me, t, a0, prev, GEOM)
    p_sel = {k: p[k][oi] for k in ("x", "y", "psi")}
    v_one, _ = copy_oracle(_opp(p_sel, st[oi], pace[oi]), torch.arange(n), me, t, a0, prev, GEOM)
    assert torch.equal(v_all, v_one)


# ------------------------------------------------------------------------------------------------ the trunk reads it
def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    prev = (torch.randn(n, PC_N, 3, generator=g) * torch.tensor([2.0, 0.5, 0.2]), torch.ones(n, dtype=torch.bool))
    return torch.rand(n, N_SCANS, BEAMS, generator=g), torch.randn(n, STATE_DIM2, generator=g), prev


def test_the_trunk_reads_the_oracle_on_its_rows_and_the_head_elsewhere():
    torch.manual_seed(0)
    a = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True, pred_smooth=True, pred_kin=_kin(),
               pred_clear=True, pc_geom=GEOM).eval()
    with torch.no_grad():
        a.trunk[0].weight[:, -PC_N:].normal_(0.0, 0.5)            # the plan reads the clearances
    n = 5
    sc, st, prev = _inputs(n, 1)
    h = torch.zeros(n, MEM)
    with torch.no_grad():
        m_head, _, _ = a(sc, st, h, prev=prev)
        pc_head = a.pc_out.clone()
        orc = torch.full((n, PC_N), -0.2)
        use = torch.tensor([True, False, True, False, True])
        m_orc, _, _ = a(sc, st, h, prev=prev, pc_over=(orc, use))
    assert torch.equal(a.pc_out[use], orc[use]) and torch.equal(a.pc_out[~use], pc_head[~use])
    assert torch.allclose(m_orc[~use], m_head[~use], atol=1e-6)        # rows without the oracle drive as before
    assert float((m_orc[use] - m_head[use]).abs().max()) > 1e-4        # rows with it read it
    # what the update replays: the stored input (the oracle's) gives the rollout's plan again
    with torch.no_grad():
        f = a.features(sc, st)
        h1 = a.mem_cell(f, h)
        q = a.pred_q(sc, f, h1)
        m_rep = a.heads(a.mem_trunk(f, h1, a.pred_feed(q), a.pc_out))[0]
    assert torch.allclose(m_rep, m_orc, atol=1e-6)
    # a frozen copy (`SelfOpponents.add` deep-copies the learner) carries no oracle: it reads its own head
    cp = copy.deepcopy(a)
    with torch.no_grad():
        m_cp, _, _ = cp(sc, st, h, prev=prev)
    assert torch.equal(cp.pc_out, pc_head) and torch.allclose(m_cp, m_head, atol=1e-6)


def test_a_fork_with_the_oracle_drives_as_its_parent_at_first():
    """race33b-like parent (--pred-opp --pred-smooth, no clearances) -> the gate's fork (+ --pred-clear, oracle input): the
    clearances' trunk columns start at zero, so whatever the oracle says the fork plans as its parent did"""
    torch.manual_seed(0)
    par = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True, pred_smooth=True, pred_kin=_kin()).eval()
    with torch.no_grad():
        for q in par.parameters():
            q.add_(0.05 * torch.randn_like(q))
    fork = Actor2(BEAMS, memory=MEM, mem_front=True, mem_value=True, pred_opp=True, pred_smooth=True, pred_kin=_kin(),
                  pred_clear=True, pc_geom=GEOM)
    assert PR.init_actor(fork, par.state_dict()) == ""
    fork.eval()
    hp = hf = torch.zeros(4, MEM)
    for k in range(4):
        sc, st, prev = _inputs(4, 20 + k)
        orc = (torch.rand(4, PC_N) * 4.0 - 1.0) / PC_SCALE
        with torch.no_grad():
            mp, _, vp = par(sc, st, hp); hp = par.h_out
            mf, _, vf = fork(sc, st, hf, prev=prev, pc_over=(orc, torch.ones(4, dtype=torch.bool))); hf = fork.h_out
        assert torch.allclose(mp, mf, rtol=0, atol=1e-6) and torch.allclose(vp, vf, rtol=0, atol=5e-6) and torch.equal(hp, hf)
        assert torch.equal(fork.pc_out, orc)


def test_arg_checks(capsys):
    """--check-args (the console's queue) refuses the oracle without --pred-clear / --self-opp and passes it with both"""
    import os
    world = os.path.join(os.path.dirname(PR.__file__), "configs", "plan_world_korea26scan_race.json")
    base = ["--name", "t", "--format", "v2", "--world", world, "--memory", "16", "--mem-front", "--pred-opp", "--check-args"]
    for extra, why in ((["--pred-clear-oracle"], "--pred-clear-oracle needs --pred-clear and --self-opp"),
                       (["--pred-clear", "--pred-clear-oracle"], "--pred-clear-oracle needs --pred-clear and --self-opp"),
                       (["--self-opp", "1", "--pred-clear-oracle"], "--pred-clear-oracle needs --pred-clear and --self-opp")):
        with pytest.raises(SystemExit) as e:
            PR.main(base + extra)
        assert str(e.value) == why
    with pytest.raises(SystemExit) as e:
        PR.main(base + ["--self-opp", "1", "--pred-clear", "--pred-clear-oracle"])
    assert e.value.code == 0 and '"pred_clear_oracle": true' in capsys.readouterr().out
