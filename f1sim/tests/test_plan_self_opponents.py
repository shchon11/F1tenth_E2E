"""`plan_rl.SelfOpponents` (11-12): the other car of a race driven by a frozen copy of the policy. Its bookkeeping on a
stand-in env: which rows a copy drives and what reaches the env, a named race (level replay) drawing the same every
time, a car put somewhere else between two decisions planning from where it is, and the pool keeping its newest copy."""
import types

import pytest
import torch

from f1sim import map_controller as M
from f1sim import plan_steps as PS
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import Actor2, N_SCANS, STATE_DIM2

BEAMS, B = 90, 16                                             # 8 races of two cars: even rows learn


class _Obs:
    """`PlanObs2` on the opponent rows, without a simulator: a scan that tells where the car is"""

    def __init__(self, env, lim, goal=None, rows=None):
        self.env, self.rows = env, rows
        n = int(rows.numel())
        self.scans = torch.zeros(n, N_SCANS, BEAMS)
        self.cmd_d = torch.zeros(n)
        self.mem_h = None
        self.resets = []
        self.reset_rows(torch.arange(n))
        self.resets = []

    def _scan(self):
        return self.env.sim.state[self.rows, :1].expand(-1, BEAMS).clone()

    def reset_rows(self, rows):
        self.resets.append(rows.tolist())
        self.scans[rows] = self._scan()[rows][:, None, :].expand(-1, N_SCANS, -1)
        if self.mem_h is not None:
            self.mem_h[rows] = 0.0

    def push_scan(self):
        self.scans = torch.cat([self._scan()[:, None, :], self.scans[:, :-1]], 1)

    def state(self):
        return torch.zeros(self.scans.shape[0], STATE_DIM2)

    def set_plan2(self, p):
        self.cmd_d = p["delta"][:, PS.SUB]


def _env():
    sim = types.SimpleNamespace(state=torch.zeros(B, 7), ideal_vcap=torch.full((B,), 5.0), absent_any=False)
    sim.state[:, 0] = torch.arange(B).float() * 10.0
    learner = torch.zeros(B, dtype=torch.bool); learner[0::2] = True
    return types.SimpleNamespace(sim=sim, learner=learner, device=torch.device("cpu"), B=B,
                                 ideal_cmd=torch.zeros(B, 2), level_seed=torch.full((B,), -1, dtype=torch.long))


@pytest.fixture()
def made(monkeypatch):
    monkeypatch.setattr(PR, "PlanObs2", _Obs)
    lut, nominal = M.config_lut()
    lim = PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu")
    torch.manual_seed(0)
    actor = Actor2(BEAMS, memory=16, mem_front=True)
    with torch.no_grad():
        actor.pi.weight.normal_(0.0, 0.3)                   # the output layer starts near zero: make the plan readable
    env = _env()
    return env, lim, actor


def _decision(so):
    so.plan()
    for j in range(PS.SUB):
        so.command(j)
        so.stepped(torch.zeros(B, dtype=torch.bool))
    so.after()


def test_only_the_copy_driven_rows_reach_the_env(made):
    env, lim, actor = made
    so = PR.SelfOpponents(env, lim, 0.5, seed=3)
    so.plan()                                                # no copy yet: every car is the world's
    assert not bool(env.ideal_extra.any()) and so.share_on() == 0.0
    so.add(actor, 25)
    assert 0 < int(so.on.sum()) < so.on.numel()             # a share of the races, not all
    so.plan()
    assert torch.equal(env.ideal_extra[so.rows], so.on) and not bool(env.ideal_extra[env.learner].any())
    assert bool((env.sim.ideal_vcap[so.rows[so.on]] == 1e3).all()) and bool((env.sim.ideal_vcap[so.rows[~so.on]] == 5.0).all())
    so.command(0)
    assert bool((env.ideal_cmd[so.rows[so.on], 1] != 0).all())                  # the copy's plan is what is driven
    assert bool((env.ideal_cmd[so.rows[~so.on]] == 0).all()) and bool((env.ideal_cmd[env.learner] == 0).all())
    assert so.obs.mem_h is not None and bool((so.obs.mem_h[so.on].abs().sum(1) > 0).all())
    assert bool((so.obs.mem_h[~so.on] == 0).all())
    assert abs(so.share_on() - float(so.on.float().mean())) < 1e-6


def test_a_race_ended_draws_anew_and_starts_blank(made):
    env, lim, actor = made
    so = PR.SelfOpponents(env, lim, 1.0, seed=3)
    so.add(actor, 25)
    _decision(so)
    assert bool((so.obs.mem_h.abs().sum(1) > 0).all())
    so.plan()
    done = torch.zeros(B, dtype=torch.bool); done[so.rows[2]] = True
    so.command(0); so.stepped(done); so.command(1)
    assert bool((env.ideal_cmd[so.rows[2]] == 0).all())                         # the rest of the decision: no old plan in the new race
    so.after()
    assert so.obs.resets[-1] == [2] and bool((so.obs.mem_h[2] == 0).all()) and bool((so.obs.mem_h[3].abs().sum() > 0))


def test_a_named_race_draws_the_same_every_time(made):
    env, lim, actor = made
    so = PR.SelfOpponents(env, lim, 0.5, seed=3)
    so.add(actor, 25); so.add(actor, 50); so.add(actor, 75)
    rows = torch.arange(so.on.numel())
    env.level_seed[:] = torch.arange(B) // 2 + 1000                             # a race's two rows share its name
    so.draw(rows); a = (so.on.clone(), so.which.clone())
    so.draw(rows); b = (so.on.clone(), so.which.clone())
    assert torch.equal(a[0], b[0]) and torch.equal(a[1][a[0]], b[1][b[0]])
    assert 0 < int(a[0].sum()) < a[0].numel()
    env.level_seed[:] = -1                                                      # races without a name: a new draw each time
    seen = set()
    for _ in range(8):
        so.draw(rows); seen.add(tuple(so.on.tolist()))
    assert len(seen) > 1


def test_a_car_put_elsewhere_plans_from_where_it_is(made):
    env, lim, actor = made
    so = PR.SelfOpponents(env, lim, 1.0, seed=3)
    so.add(actor, 25)
    so.obs.resets = []                                                          # the first copy starts every row blank
    _decision(so); _decision(so)
    assert so.obs.resets == []
    k = 5
    env.sim.state[so.rows[k], 0] += 30.0                                        # `state_rl.OppRestart`, after the decision
    so.plan()
    assert so.obs.resets == [[k]]                                               # before the copy read its inputs
    assert bool((so.obs.scans[k] == env.sim.state[so.rows[k], 0]).all()) and bool((so.obs.mem_h[k] != 0).any())
    for j in range(PS.SUB):
        so.command(j); so.stepped(torch.zeros(B, dtype=torch.bool))
    so.after()
    assert so.obs.resets == [[k]]                                               # once: not again after the decision


def test_a_full_pool_keeps_its_newest_copy(made):
    env, lim, actor = made
    so = PR.SelfOpponents(env, lim, 1.0, seed=3, max_pool=3, p_newest=1.0)
    for u in range(25, 251, 25):
        so.add(actor, u)
        assert len(so.pool) <= 3 and so.tags[so.newest] == u
    assert len(set(so.tags)) == 3
    so.draw(torch.arange(so.on.numel()))
    assert bool((so.which == so.newest).all())                                  # p_newest 1: everyone meets the newest
    for a in so.pool:
        assert not any(q.requires_grad for q in a.parameters()) and not a.training


def test_a_copy_gets_its_race_top_speed(made):
    env, lim, actor = made
    so = PR.SelfOpponents(env, lim, 0.5, seed=3)                                # default: no cap, the same draws as before
    so.add(actor, 25)
    on0, which0 = so.on.clone(), so.which.clone()
    assert bool((so.cap == 1e3).all())
    env2 = _env()
    sv = PR.SelfOpponents(env2, lim, 0.5, seed=3, vcap=(3.0, 5.0))
    sv.add(actor, 25)
    assert bool(((sv.cap >= 3.0) & (sv.cap <= 5.0)).all()) and float(sv.cap.max() - sv.cap.min()) > 0.3
    sv.plan()
    assert torch.equal(env2.sim.ideal_vcap[sv.rows[sv.on]], sv.cap[sv.on])      # the cap is on the copy-driven cars only
    assert bool((env2.sim.ideal_vcap[sv.rows[~sv.on]] == 5.0).all()) and bool((env2.sim.ideal_vcap[env2.learner] == 5.0).all())
    assert 0 < int(sv.on.sum()) < sv.on.numel()
    rows = torch.arange(sv.on.numel())
    env2.level_seed[:] = torch.arange(B) // 2 + 1000                            # a named race: the same top speed every time
    sv.draw(rows); a = sv.cap.clone()
    sv.draw(rows); b = sv.cap.clone()
    assert torch.equal(a, b) and float(a.max() - a.min()) > 0.3
    env2.level_seed[:] = -1
    sv.draw(rows)
    assert not torch.equal(sv.cap, a)
    assert torch.equal(on0, so.on) and torch.equal(which0, so.which)
