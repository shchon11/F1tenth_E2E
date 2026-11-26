"""The race phase (11-24, agreed with the user topic by topic), the parts that need no simulator:
  * `Actor2.pred_opp`: the other car's predicted next second -- a fork from a policy without it drives exactly as that policy
    (`plan_rl.init_actor`: zero trunk columns), the policy's gradient never reaches the head, the head's loss never reaches the
    policy, an old checkpoint loads without it;
  * `SelfOpponents` pace (--self-opp-pace): the per-race factor and what it does to the plan driven; a race keeps its copy when
    the pool moves on; `snapshot` / `restore` put a copy back exactly;
  * `PlanReward` --car-fault: a car contact costs only when the env's rule puts it on this car."""
import types

import pytest
import torch

from f1sim import map_controller as M
from f1sim import plan_steps as PS
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import Actor2, N_SCANS, STATE_DIM2, PRED_IN, PRED_OUT, load_actor2
from f1sim.learn.plan_reward import PlanReward, CONTACT_S, CONTACT_S_PER_MPS, T_REF

BEAMS = 90


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N_SCANS, BEAMS, generator=g), torch.randn(n, STATE_DIM2, generator=g)


def _pair():
    torch.manual_seed(0)
    base = Actor2(BEAMS, memory=16, mem_front=True, mem_value=True).eval()
    with torch.no_grad():
        for p in base.parameters():
            p.add_(0.05 * torch.randn_like(p))              # not the init: a trained-looking policy
        base.norm.mean.normal_(); base.norm.var.uniform_(0.5, 2.0)
    fork = Actor2(BEAMS, memory=16, mem_front=True, mem_value=True, pred_opp=True)
    added = PR.init_actor(fork, base.state_dict())
    return base, fork.eval(), added


def test_a_fork_with_the_prediction_head_drives_exactly_as_its_parent():
    base, fork, added = _pair()
    assert added and fork.trunk[0].in_features == base.trunk[0].in_features + PRED_IN
    assert bool((fork.trunk[0].weight[:, -PRED_IN:] == 0).all())
    for (n, p), q in zip(fork.pred.enc.state_dict().items(), fork.enc.state_dict().values()):
        assert torch.equal(p, q), n                         # the head's encoder starts as the policy's
    hb = hf = torch.zeros(5, 16)
    for k in range(6):                                      # decisions in a row, the memory carried
        sc, st = _inputs(5, k)
        mb, lb, vb = base(sc, st, hb); hb = base.h_out
        mf, lf, vf = fork(sc, st, hf); hf = fork.h_out
        assert torch.allclose(mb, mf, atol=1e-6) and torch.allclose(lb, lf) and torch.allclose(vb, vf, atol=1e-6)
        assert torch.allclose(hb, hf, atol=1e-6)
        assert fork.pred_out.shape == (5, PRED_OUT)
    # once the plan's weights on the prediction move, the prediction matters (it is wired in)
    with torch.no_grad():
        fork.trunk[0].weight[:, -PRED_IN:].normal_(0.0, 0.5)
    mf2, _, _ = fork(sc, st, torch.zeros(5, 16))
    mb2, _, _ = base(sc, st, torch.zeros(5, 16))
    assert (mf2 - mb2).abs().max() > 1e-4


def test_the_policy_gradient_never_reaches_the_head_and_the_head_loss_never_the_policy():
    _, fork, _ = _pair()
    fork.train()
    sc, st = _inputs(4, 1)
    mean, log_std, v_hat = fork(sc, st, torch.zeros(4, 16))
    pol = fork.policy_parameters()
    head = list(fork.pred.parameters())
    assert len(pol) + len(head) == len(list(fork.parameters())) and not any(p is h for p in pol for h in head)
    g = torch.autograd.grad((mean.pow(2).sum() + v_hat.sum()), head, allow_unused=True, retain_graph=True)
    assert all(x is None or float(x.abs().max()) == 0.0 for x in g)
    q = fork.pred_out
    g2 = torch.autograd.grad(q.pow(2).sum(), pol, allow_unused=True)
    assert all(x is None or float(x.abs().max()) == 0.0 for x in g2)


def test_an_old_checkpoint_loads_without_the_head_and_a_new_one_with_it():
    base, fork, _ = _pair()
    args = {"memory": 16, "mem_front": True, "std0": 0.6}
    a0 = load_actor2({"actor": base.state_dict(), "args": args, "beams": BEAMS}, "cpu", check_mount=False)
    a1 = load_actor2({"actor": fork.state_dict(), "args": args, "beams": BEAMS}, "cpu", check_mount=False)
    assert not a0.pred_opp and a0.pred is None and a1.pred_opp and a1.pred is not None


# ------------------------------------------------------------------ SelfOpponents (stand-in env, as test_plan_self_opponents)
B = 16                                                      # 8 races of two cars: even rows learn


class _Obs:
    def __init__(self, env, lim, goal=None, rows=None):
        self.env, self.rows = env, rows
        n = int(rows.numel())
        self.scans = torch.zeros(n, N_SCANS, BEAMS)
        self.rel = torch.zeros(n, PS.N_STEP - 1, 2); self.rel_v = torch.zeros(n, PS.N_STEP)
        self.cmd_d = torch.zeros(n); self.cmd_v = torch.zeros(n)
        self.plan_xy = torch.zeros(n, PS.N_STEP, 2); self.plan_v = torch.zeros(n, PS.N_STEP)
        self.rel_psi = torch.zeros(n, PS.N_STEP - 1); self.prev_ok = torch.zeros(n, dtype=torch.bool)   # (11-25, --pred-clear)
        self.mem_h = None
        self.reset_rows(torch.arange(n))

    def _scan(self):
        return self.env.sim.state[self.rows, :1].expand(-1, BEAMS).clone()

    def reset_rows(self, rows):
        self.scans[rows] = self._scan()[rows][:, None, :].expand(-1, N_SCANS, -1)
        self.rel[rows] = 0.0; self.cmd_d[rows] = 0.0; self.rel_psi[rows] = 0.0; self.prev_ok[rows] = False
        if self.mem_h is not None:
            self.mem_h[rows] = 0.0

    def push_scan(self):
        self.scans = torch.cat([self._scan()[:, None, :], self.scans[:, :-1]], 1)

    def state(self):
        return torch.zeros(self.scans.shape[0], STATE_DIM2)

    def set_plan2(self, p):
        self.cmd_d = p["delta"][:, PS.SUB]; self.cmd_v = p["v"][:, PS.SUB]
        self.rel_psi = p["psi"][:, 2 * PS.SUB::PS.SUB] - p["psi"][:, PS.SUB:PS.SUB + 1]; self.prev_ok = torch.ones_like(self.prev_ok)


def _env():
    sim = types.SimpleNamespace(state=torch.zeros(B, 7), ideal_vcap=torch.full((B,), 5.0), absent_any=False)
    sim.state[:, 0] = torch.arange(B).float() * 10.0
    sim.state[:, 3] = 3.0                                     # every car at 3 m/s
    learner = torch.zeros(B, dtype=torch.bool); learner[0::2] = True
    return types.SimpleNamespace(sim=sim, learner=learner, device=torch.device("cpu"), B=B, ideal_cmd=torch.zeros(B, 2),
                                 level_seed=torch.full((B,), -1, dtype=torch.long),
                                 last_result=types.SimpleNamespace(imu=torch.randn(B, 2, 6)))


@pytest.fixture()
def made(monkeypatch):
    monkeypatch.setattr(PR, "PlanObs2", _Obs)
    lut, nominal = M.config_lut()
    lim = PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu")
    torch.manual_seed(0)
    actor = Actor2(BEAMS, memory=16, mem_front=True)
    with torch.no_grad():
        actor.pi.weight.normal_(0.0, 0.3)
    return _env(), lim, actor


def test_the_pace_factor_is_drawn_per_race_and_slows_the_plan_driven(made):
    env, lim, actor = made
    so = PR.SelfOpponents(env, lim, 1.0, seed=3, pace=(0.6, 0.95, 0.25))
    so.add(actor, 500)
    for _ in range(30):
        so.draw(torch.arange(so.on.numel()))
        f = so.pace
        assert bool(((f == 1.0) | ((f >= 0.6) & (f <= 0.95))).all())
    full = torch.stack([(so.draw(torch.arange(so.on.numel())), so.pace == 1.0)[1] for _ in range(400)]).float().mean()
    assert 0.18 < float(full) < 0.32                         # ~P of the races at the copy's own pace
    so.pace[:] = torch.tensor([1.0, 0.6, 0.8, 0.95, 1.0, 0.7, 0.6, 0.9])
    so.plan()
    slow = so.pace < 1.0
    v_true = env.sim.state[so.rows, 3]
    assert torch.allclose(so.p["v"][slow, 0], (v_true / so.pace)[slow], atol=1e-5)   # it plans from its own-pace speed
    assert torch.allclose(so.cmd_v_driven, so.p["cmd_v"] * so.pace[:, None])          # and drives the plan's speeds x f
    so.command(0)
    assert torch.allclose(env.ideal_cmd[so.rows, 1], so.p["cmd_v"][:, 0] * so.pace)


def test_without_pace_nothing_changes(made):
    env, lim, actor = made
    a = PR.SelfOpponents(env, lim, 0.5, seed=3)
    a.add(actor, 25)
    env2 = _env()
    b = PR.SelfOpponents(env2, lim, 0.5, seed=3, pace=None)
    b.add(actor, 25)
    assert torch.equal(a.on, b.on) and torch.equal(a.which, b.which) and bool((a.pace == 1.0).all())
    a.plan()
    assert torch.equal(a.cmd_v_driven, a.p["cmd_v"])


def test_a_race_keeps_its_copy_when_the_pool_moves_on(made):
    env, lim, actor = made
    so = PR.SelfOpponents(env, lim, 1.0, seed=3, max_pool=2, p_newest=1.0)
    so.add(actor, 500)
    tag0 = so.row_tag.clone()
    assert bool((tag0 == 500).all())
    import copy
    a2 = copy.deepcopy(actor)
    so.add(a2, 525); so.add(a2, 550)                         # 500 leaves the pool
    assert 500 not in so.tags and so.model(500) is not None  # ... but the races under way still drive it
    assert bool((so.row_tag == 500).all())
    so.plan()                                                # and it plans with it
    so.on[:] = False                                         # those races end: nothing needs it any more
    so.prune()
    assert so.model(500) is None
    so.on[:] = True; so.row_tag[:] = 525
    so.keep_tags = lambda: {525}
    so.add(a2, 575)
    assert so.model(525) is not None                         # a stored scene keeps it


def test_snapshot_and_restore_put_the_copy_back(made):
    env, lim, actor = made
    so = PR.SelfOpponents(env, lim, 1.0, seed=3, max_pool=2, p_newest=0.5, pace=(0.6, 0.95, 0.2))
    so.add(actor, 500); so.add(actor, 525)
    for _ in range(3):
        so.plan()
        for j in range(PS.SUB):
            so.command(j); so.stepped(torch.zeros(B, dtype=torch.bool))
        so.after()
    oi = torch.tensor([1, 3, 6])
    snap = so.snapshot(oi)
    assert set(snap) == set(PR.SelfOpponents.PACE_KEYS)
    keep = {k: v.clone() for k, v in snap.items()}
    so.on[:] = False; so.pace[:] = 0.5; so.row_tag[:] = 525; so.obs.mem_h.zero_(); so.obs.scans.zero_()
    env.last_result.imu.zero_()
    so.restore(oi, keep)
    again = so.snapshot(oi)
    for k in PR.SelfOpponents.PACE_KEYS:
        assert torch.allclose(again[k].float(), keep[k].float()), k


# ------------------------------------------------------------------ PlanReward --car-fault
def _reward(fault_only, ahead_gap):
    """a PlanReward on two learner rows racing two others; the other car `ahead_gap` m ahead along the lane (+ ahead)"""
    from f1sim.gym_env import F1VecEnv
    pr = PlanReward.__new__(PlanReward)
    env = types.SimpleNamespace()
    env.B = 4
    env.sim = types.SimpleNamespace(other_idx=torch.tensor([[1], [0], [3], [2]]), tid=torch.zeros(4, dtype=torch.long),
                                    track=types.SimpleNamespace(length=torch.tensor([40.0])), control_dt=0.025)
    env._present_mask_gaps = False
    env.signed_gaps = types.MethodType(F1VecEnv.signed_gaps, env)
    env.car_contact_charge = types.MethodType(F1VecEnv.car_contact_charge, env)
    s = torch.tensor([10.0, 10.0 + ahead_gap, 20.0, 20.0 + ahead_gap])
    on = torch.zeros(4, 3, dtype=torch.bool); on[:, 2] = True
    env.last_result = types.SimpleNamespace(s=s, contact_onsets=on, contact_impacts=torch.full((4, 3), 1.0))
    pr.env, pr.lid = env, torch.tensor([0, 2])
    pr.phi = torch.zeros(2); pr.cost = torch.zeros(2); pr.seen_t = torch.zeros(2); pr.n_contacts = torch.zeros(2, 3)
    pr.stall_s = 0.0; pr.contact_scale = 0.82; pr.car_fault_only = fault_only; pr.car_charged = torch.zeros(2)
    pr.cost_kind = torch.zeros(2, 4)                    # (11-25) contacts by kind (wall, box, car, stall), --plan-car-cost's live tap
    pr.ww_on, pr.ww_k = False, 0.0                     # (11-25) --wrong-way-cost off
    pr.cond = None                                     # (11-26) --contact-cond off
    pr.ww_since = torch.full((2,), 1e9)                # (11-26) seconds since a car contact (--wrong-way-cost's car-push waiver)
    return pr


@pytest.mark.parametrize("gap,ours", [(2.0, True), (0.3, True), (-0.3, True), (-2.0, False)])
def test_a_car_contact_costs_only_when_it_is_ours(gap, ours):
    one = 0.82 * (CONTACT_S + CONTACT_S_PER_MPS * 1.0) / T_REF
    pr = _reward(True, gap)
    pr.control_step(torch.ones(2, dtype=torch.bool))
    assert torch.allclose(pr.cost, torch.full((2,), one if ours else 0.0))       # as large as a wall at that impact
    assert torch.equal(pr.car_charged, torch.full((2,), 1.0 if ours else 0.0))
    old = _reward(False, gap)                                                   # off: every seen car contact costs
    old.control_step(torch.ones(2, dtype=torch.bool))
    assert torch.allclose(old.cost, torch.full((2,), one))


def test_a_car_not_seen_still_costs_nothing():
    pr = _reward(True, 2.0)
    pr.seen_t = torch.full((2,), 5.0)
    pr.control_step(None)
    assert float(pr.cost.abs().max()) == 0.0
