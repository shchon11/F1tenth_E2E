"""(11-26) a held intent in a plan policy's action (`plan_intent`, plan_rl --intent K,H,EPS; RQ6 (c) / RQ3): the parts that
need no simulator --
  * off, nothing changes: an actor without it (or with '') is the same actor, an old checkpoint loads without it, and an
    intent actor's other weights start exactly as a plain one's;
  * z is held H decisions, drawn again at a start / restart, F1SIM_INTENT_FIX forces one; drive time takes the argmax;
  * the log-probability: log pi_high(z | h) + log pi_low(a | h, z) at a draw, log pi_low only while held, the mixture with
    the floor scored -- and the update's recomputation matches the rollout's (PPO ratio 1 before any step);
  * the intent and its hold round-trip through a stored moment (a copy's `SelfOpponents.snapshot` / `restore`), and a copy
    carries its own;
  * the structured start: a fresh net's 1 s plan points lie the intended distances apart (a steady steering offset, an arc);
  * --intent-trunk: the trunk reads z through first-layer columns that start at zero -- at the start the plan under every z is
    exactly the vectors-only actor's; moved, z changes the control law state by state while pi_high, the speed estimate and
    the memory stay z-free."""
import math
import types

import pytest
import torch

from f1sim import map_controller as M
from f1sim import plan_steps as PS
from f1sim.learn import plan_intent as IN
from f1sim.learn import plan_rl as PR
from f1sim.learn.plan_policy import Actor2, Critic, N_SCANS, STATE_DIM2, PlanObs2, load_actor2, plan_v2, tanh_logp

BEAMS = 90
K, H, EPS = 4, 15, 0.1
SPEC = f"{K},{H},{EPS}"


@pytest.fixture(scope="module")
def lim():
    lut, nominal = M.config_lut()
    return PS.Limits.from_nominal(nominal, lut, M.GRIP_LEVELS.index(1.0), "cpu", margin=0.95)


def _inputs(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, N_SCANS, BEAMS, generator=g), torch.randn(n, STATE_DIM2, generator=g)


# ------------------------------------------------------------------ off: nothing changes
def test_off_is_the_same_actor():
    torch.manual_seed(3)
    a = Actor2(BEAMS, memory=16, mem_front=True)
    torch.manual_seed(3)
    b = Actor2(BEAMS, memory=16, mem_front=True, intent="")
    assert a.intent_k == 0 and b.intent_k == 0 and not hasattr(b, "intent_head") and not hasattr(b, "intent_bias")
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    sc, st = _inputs(5, 0)
    ma, la, va = a(sc, st, torch.zeros(5, 16)); mb, lb, vb = b(sc, st, torch.zeros(5, 16))
    assert torch.equal(ma, mb) and torch.equal(la, lb) and torch.equal(va, vb) and b.intent_logits_out is None
    # an intent actor's own weights are made last: everything else starts exactly as the plain actor's
    torch.manual_seed(3)
    c = Actor2(BEAMS, memory=16, mem_front=True, intent=SPEC)
    sc_ = c.state_dict()
    assert all(torch.equal(sa[k], sc_[k]) for k in sa)
    assert set(sc_) - set(sa) == {"intent_head.weight", "intent_head.bias", "intent_bias"}
    # and its plan without an intent is the plain actor's plan (the logits are a separate head)
    mc, _, vc = c(sc, st, torch.zeros(5, 16))
    assert torch.equal(ma, mc) and torch.equal(va, vc) and c.intent_logits_out.shape == (5, K)


def test_old_checkpoint_loads_without_intent_and_a_new_one_with_it():
    a = Actor2(BEAMS)
    old = load_actor2({"beams": BEAMS, "actor": a.state_dict(), "args": {}}, "cpu", check_mount=False)
    assert old.intent_k == 0
    b = Actor2(BEAMS, intent="3,10,0.2")
    with torch.no_grad():
        b.intent_bias.normal_(); b.intent_head.weight.normal_()
    new = load_actor2({"beams": BEAMS, "actor": b.state_dict(), "args": {"intent": "3,10,0.2"}}, "cpu", check_mount=False)
    assert (new.intent_k, new.intent_hold, new.intent_eps) == (3, 10, 0.2)
    assert torch.equal(new.intent_bias, b.intent_bias) and torch.equal(new.intent_head.weight, b.intent_head.weight)


def test_parse_and_refusals():
    assert IN.parse("") is None and IN.parse("4,15,0.1") == (4, 15, 0.1)
    for bad in ("4,15", "1,15,0.1", "4,0,0.1", "4,15,1.0", "a,b,c"):
        with pytest.raises(ValueError):
            IN.parse(bad)
    with pytest.raises(ValueError):
        Actor2(BEAMS, branches=2, intent=SPEC)
    with pytest.raises(ValueError):
        Actor2(BEAMS, pace_caps="6.5", intent=SPEC)
    for extra, word in ((["--commit-plan", "1"], "--commit-plan"), (["--branches", "2"], "--branches 2"),
                        (["--format", "v3"], "--format v2"), (["--intent-lat", "-1"], "--intent-lat")):
        with pytest.raises(SystemExit) as e:
            PR.main(["--name", "x", "--intent", SPEC, "--check-args"] + extra)
        assert word in str(e.value.code), e.value.code
    with pytest.raises(SystemExit) as e:
        PR.main(["--name", "x", "--intent", SPEC, "--tail-consistency", "1", "--check-args"])
    assert e.value.code == 0


# ------------------------------------------------------------------ holding
def test_z_is_held_for_H_decisions_and_redrawn_at_a_restart():
    torch.manual_seed(0)
    B = 64
    z = torch.zeros(B, dtype=torch.long); left = torch.zeros(B, dtype=torch.long)
    hist, dues = [], []
    for t in range(3 * H + 4):
        logits = torch.randn(B, K) * 3.0                        # the head's answer moves every decision
        if t == 2 * H + 2:
            left[:8] = 0                                        # a restart: those rows draw at once
        z, left, due = IN.step(logits, z, left, H, EPS, "sample")
        hist.append(z.clone()); dues.append(due.clone())
    Z, Dz = torch.stack(hist), torch.stack(dues)
    # rows 8..: drawn at 0, H, 2H, ... only, held in between
    want = torch.zeros(3 * H + 4, dtype=torch.bool); want[::H] = True
    assert torch.equal(Dz[:, 8:], want[:, None].expand(-1, B - 8))
    for s in range(0, 3 * H, H):
        assert bool((Z[s:s + H, 8:] == Z[s, 8:]).all())
    assert bool((Z[H, 8:] != Z[0, 8:]).any())                   # and a draw does change it (somewhere)
    # rows 0..7 restarted at 2H + 2: drawn there, then held H again
    assert bool(Dz[2 * H + 2, :8].all()) and not bool(Dz[2 * H + 3:3 * H + 2, :8].any()) and bool(Dz[3 * H + 2, :8].all())
    assert bool((Z[2 * H + 2:3 * H + 2, :8] == Z[2 * H + 2, :8]).all())


def test_drive_time_takes_the_argmax_and_the_fix_forces(monkeypatch):
    logits = torch.tensor([[0.0, 2.0, 1.0, -1.0], [3.0, 0.0, 0.0, 0.0]])
    z0, l0 = torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)
    z, left, due = IN.step(logits, z0, l0, H, EPS, "argmax")
    assert z.tolist() == [1, 0] and left.tolist() == [H - 1, H - 1] and bool(due.all())
    z2, left2, due2 = IN.step(logits.flip(1), z, left, H, EPS, "argmax")     # held: the head's new answer is not read
    assert z2.tolist() == [1, 0] and left2.tolist() == [H - 2, H - 2] and not bool(due2.any())
    monkeypatch.setenv(IN.ENV_FIX, "3")
    z3, _, due3 = IN.step(logits, z2, left2, H, EPS, "argmax", IN.fixed())
    assert z3.tolist() == [3, 3] and bool(due3.all())
    monkeypatch.setenv(IN.ENV_FIX, "7")
    with pytest.raises(ValueError):
        IN.step(logits, z2, left2, H, EPS, "argmax", IN.fixed())


class _StubObs:
    """what `plan_v2` reads of `PlanObs2`, without a simulator"""

    def __init__(self, n):
        self.scans = torch.rand(n, N_SCANS, BEAMS)
        self.rel = torch.zeros(n, PS.N_STEP - 1, 2)
        self.cmd_d = torch.zeros(n)
        self.prev_branch = self.pace_idx = None
        self.intent_z = self.intent_left = None
        self._st = torch.randn(n, STATE_DIM2)

    intent_on = PlanObs2.intent_on

    def state(self):
        return self._st


def test_plan_v2_holds_the_intent_and_draws_the_plan_under_it(lim):
    torch.manual_seed(1)
    a = Actor2(BEAMS, intent=SPEC).eval()
    a.intent_init(lim, 0.3)
    with torch.no_grad():
        a.intent_head.weight.normal_(0.0, 1.0)                  # a head with opinions that change with the input
    obs = _StubObs(6)
    zs = []
    for t in range(2 * H):
        obs._st = torch.randn(6, STATE_DIM2)
        nom, _, _ = a(obs.scans, obs.state())
        u, mean, _, _, p = plan_v2(a, obs, lim)
        zs.append(obs.intent_z.clone())
        assert torch.allclose(mean, nom + a.intent_bias[obs.intent_z], atol=1e-6) and torch.equal(u, mean)
        if t in (0, H):
            assert torch.equal(obs.intent_z, a.intent_logits_out.argmax(1))   # a draw: the likeliest
    Z = torch.stack(zs)
    assert bool((Z[:H] == Z[0]).all()) and bool((Z[H:] == Z[H]).all())


# ------------------------------------------------------------------ the log-probability
def test_the_mixture_and_its_log_probability():
    lg = torch.randn(32, K, requires_grad=True)
    p = IN.mix_probs(lg, EPS).detach()
    assert torch.allclose(p.sum(1), torch.ones(32)) and float(p.min()) >= EPS / K - 1e-7
    p_sure = IN.mix_probs(torch.tensor([[60.0, 0.0, 0.0, 0.0]]), EPS)
    assert abs(float(p_sure[0, 0]) - (1 - EPS + EPS / K)) < 1e-6 and abs(float(p_sure[0, 1]) - EPS / K) < 1e-6
    z = torch.randint(0, K, (32,)); due = torch.arange(32) % 3 == 0
    lp = IN.logp(lg, z, due, EPS)
    want = torch.log((1 - EPS) * torch.softmax(lg, 1) + EPS / K).gather(1, z[:, None])[:, 0]
    assert torch.allclose(lp[due], want[due]) and bool((lp[~due] == 0).all())
    g, = torch.autograd.grad(lp.sum(), lg)
    assert bool((g[~due] == 0).all()) and bool((g[due].abs().sum(1) > 0).all())   # a held decision teaches pi_high nothing


def test_the_action_log_probability_at_a_draw_and_while_held(lim):
    """the rollout's (draw, then the plan under it, noise on the first step) and the update's recomputation from the stored
    z / draw flags: equal at the same weights (PPO's ratio 1), and = log pi_high + log pi_low at a draw, log pi_low held"""
    torch.manual_seed(2)
    a = Actor2(BEAMS, memory=16, mem_front=True, intent=SPEC).eval()
    a.intent_init(lim, 0.3)
    with torch.no_grad():
        a.intent_head.weight.normal_(0.0, 0.5)
    rl = torch.tensor([0, PS.N_STEP])
    n = 8
    z = torch.zeros(n, dtype=torch.long); left = torch.zeros(n, dtype=torch.long)
    left[4:] = 5                                                   # half the rows hold an intent already
    z[4:] = 2
    sc, st = _inputs(n, 5)
    h = torch.randn(n, 16) * 0.3
    with torch.no_grad():
        nom, log_std, _ = a(sc, st, h)
        lg = a.intent_logits_out.clone()
        z, left, due = IN.step(lg, z, left, H, EPS, "sample")
        mean = a.intent_mean(nom, z)
        eps = torch.randn_like(mean)
        u = mean + log_std.exp() * eps * torch.zeros(PS.ACT_DIM).index_fill_(0, rl, 1.0)
        lp_roll = tanh_logp(u[:, rl], mean[:, rl], log_std[:, rl]) + IN.logp(lg, z, due, EPS)
    assert due.tolist() == [True] * 4 + [False] * 4 and bool((z[4:] == 2).all())
    # the update: the same inputs, the stored z and draw flags
    nom2, ls2, _ = a(sc, st, h)
    lp_upd = tanh_logp(u[:, rl], a.intent_mean(nom2, z)[:, rl], ls2[:, rl]) + IN.logp(a.intent_logits_out, z, due, EPS)
    assert torch.allclose(lp_upd, lp_roll, atol=1e-5)
    # by hand: the Gaussian of the first step's two outputs (tanh-squashed) + the mixture's probability of z at a draw
    d = torch.distributions.Normal(mean[:, rl], log_std[:, rl].exp())
    a_ = torch.tanh(u[:, rl])
    low = (d.log_prob(u[:, rl]) - torch.log(1 - a_ ** 2)).sum(1)
    high = torch.log(IN.mix_probs(lg, EPS).gather(1, z[:, None])[:, 0])
    assert torch.allclose(lp_roll[:4], (low + high)[:4], atol=1e-4)
    assert torch.allclose(lp_roll[4:], low[4:], atol=1e-4)
    # the low level's log-probability reaches the intent's vector (the first step's columns only)
    g, = torch.autograd.grad(lp_upd.sum(), a.intent_bias)
    assert bool((g[:, rl].abs().sum() > 0)) and float(g[:, [k for k in range(PS.ACT_DIM) if k not in rl.tolist()]].abs().sum()) == 0.0


# ------------------------------------------------------------------ stored moments, copies
def test_intent_state_round_trips():
    """(z, left) is the intent's whole state: put back, the same draws follow (the learner's scene entries)"""
    torch.manual_seed(4)
    B = 16
    z = torch.zeros(B, dtype=torch.long); left = torch.zeros(B, dtype=torch.long)
    for _ in range(7):
        z, left, _ = IN.step(torch.randn(B, K), z, left, H, EPS, "sample")
    snap = {"pol_intent_z": z.clone(), "pol_intent_left": left.clone()}
    g = torch.Generator().manual_seed(9)
    logits = [torch.randn(B, K, generator=g) for _ in range(20)]

    def run(z_, l_):
        torch.manual_seed(11)
        out = []
        for lg in logits:
            z_, l_, _ = IN.step(lg, z_, l_, H, EPS, "sample")
            out.append(z_.clone())
        return torch.stack(out)
    first = run(z, left)
    z2 = torch.randint(0, K, (B,)); l2 = torch.zeros(B, dtype=torch.long)   # the rows moved on
    z2[:] = snap["pol_intent_z"]; l2[:] = snap["pol_intent_left"]
    assert torch.equal(run(z2, l2), first)


def test_critic_reads_the_held_intent():
    o = types.SimpleNamespace(intent_k=K, intent_hold=H, intent_z=torch.tensor([2, 1, 3]), intent_left=torch.tensor([5, 0, H - 1]))
    pv = PlanObs2.intent_priv(o)
    assert pv.shape == (3, K + 1)
    assert pv[0, :K].tolist() == [0, 0, 1, 0] and abs(float(pv[0, K]) - 5 / H) < 1e-6
    assert pv[1].tolist() == [0.0] * (K + 1)                      # running out now: the next draw is this decision's action
    assert pv[2, 3] == 1 and abs(float(pv[2, K]) - (H - 1) / H) < 1e-6


def test_a_fork_critic_widened_for_the_intent_values_as_before():
    torch.manual_seed(5)
    old = Critic(BEAMS, 20, state_dim=STATE_DIM2)
    with torch.no_grad():
        old.norm.mean.normal_(); old.norm.var.uniform_(0.5, 2.0)
    new = Critic(BEAMS, 20 + K + 1, state_dim=STATE_DIM2)
    sd = IN.widen_critic(old.state_dict(), new)
    new.load_state_dict(sd)
    sc, st = _inputs(6, 1)
    pv = torch.randn(6, 20)
    extra = torch.rand(6, K + 1)
    assert torch.allclose(old(sc, st, pv), new(sc, st, torch.cat([pv, extra], 1)), atol=1e-6)
    assert IN.widen_critic(old.state_dict(), Critic(BEAMS, 20 + K + 1, hidden=128, state_dim=STATE_DIM2)) is None


B = 16                                                      # 8 races of two cars: even rows learn


class _Obs:
    """`PlanObs2` on the opponent rows of a stand-in env (as test_race_phase's), with the intent's state"""

    def __init__(self, env, lim, goal=None, rows=None):
        self.env, self.rows = env, rows
        n = int(rows.numel())
        self.scans = torch.zeros(n, N_SCANS, BEAMS)
        self.rel = torch.zeros(n, PS.N_STEP - 1, 2); self.rel_v = torch.zeros(n, PS.N_STEP)
        self.cmd_d = torch.zeros(n); self.cmd_v = torch.zeros(n)
        self.plan_xy = torch.zeros(n, PS.N_STEP, 2); self.plan_v = torch.zeros(n, PS.N_STEP)
        self.rel_psi = torch.zeros(n, PS.N_STEP - 1); self.prev_ok = torch.zeros(n, dtype=torch.bool)
        self.mem_h = None
        self.intent_z = self.intent_left = None
        self.reset_rows(torch.arange(n))

    intent_on = PlanObs2.intent_on

    def _scan(self):
        return self.env.sim.state[self.rows, :1].expand(-1, BEAMS).clone()

    def reset_rows(self, rows):
        self.scans[rows] = self._scan()[rows][:, None, :].expand(-1, N_SCANS, -1)
        self.rel[rows] = 0.0; self.cmd_d[rows] = 0.0; self.rel_psi[rows] = 0.0; self.prev_ok[rows] = False
        if self.mem_h is not None:
            self.mem_h[rows] = 0.0
        if self.intent_z is not None:
            self.intent_z[rows] = 0; self.intent_left[rows] = 0

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
    sim.state[:, 3] = 3.0
    learner = torch.zeros(B, dtype=torch.bool); learner[0::2] = True
    return types.SimpleNamespace(sim=sim, learner=learner, device=torch.device("cpu"), B=B, ideal_cmd=torch.zeros(B, 2),
                                 level_seed=torch.full((B,), -1, dtype=torch.long),
                                 last_result=types.SimpleNamespace(imu=torch.randn(B, 2, 6)))


def test_a_copy_carries_its_own_intent_and_it_round_trips(monkeypatch, lim):
    monkeypatch.setattr(PR, "PlanObs2", _Obs)
    torch.manual_seed(0)
    actor = Actor2(BEAMS, memory=16, mem_front=True, intent=SPEC)
    actor.intent_init(lim, 0.3)
    with torch.no_grad():
        actor.pi.weight.normal_(0.0, 0.3); actor.intent_head.bias.copy_(torch.tensor([0.0, 0.0, 2.0, 0.0]))
    env = _env()
    so = PR.SelfOpponents(env, lim, 1.0, seed=3, max_pool=2, p_newest=0.5)
    actor.heads(torch.randn(2, 256, requires_grad=True))           # as after an update's epochs: pi_high's logits in a graph
    lg = actor.intent_logits_out
    assert lg.grad_fn is not None
    so.add(actor, 500); so.add(actor, 525)                       # (deepcopy refuses a graph tensor: it is left out of the copy)
    assert actor.intent_logits_out is lg
    assert so.obs.intent_z is not None
    for k in range(3):
        so.plan()
        assert bool((so.obs.intent_z == 2).all())                  # the likeliest, as a copy drives
        assert so.obs.intent_left.tolist() == [H - 1 - k] * so.on.numel()
        for j in range(PS.SUB):
            so.command(j); so.stepped(torch.zeros(B, dtype=torch.bool))
        so.after()
    oi = torch.tensor([1, 3, 6])
    snap = so.snapshot(oi)
    assert set(snap) == set(PR.SelfOpponents.PACE_KEYS)
    keep = {k: v.clone() for k, v in snap.items()}
    so.obs.intent_z.zero_(); so.obs.intent_left.fill_(9)
    so.restore(oi, keep)
    again = so.snapshot(oi)
    for k in PR.SelfOpponents.PACE_KEYS:
        assert torch.allclose(again[k].float(), keep[k].float()), k
    assert again["intent_z"].tolist() == [2, 2, 2] and again["intent_left"].tolist() == [H - 3] * 3
    # a restart: drawn again at the next plan
    done = torch.zeros(B, dtype=torch.bool); done[so.rows[2]] = True
    so.plan(); so.stepped(done); so.after()
    assert int(so.obs.intent_left[2]) == 0 and int(so.obs.intent_left[3]) == H - 4


# ------------------------------------------------------------------ the structured start
def test_a_fresh_net_puts_the_intents_1s_points_the_intended_distances_apart(lim):
    torch.manual_seed(6)
    a = Actor2(BEAMS, memory=16, mem_front=True, intent=SPEC)
    a.intent_init(lim, 0.3)
    want = torch.tensor([-0.3, -0.1, 0.1, 0.3])
    sc, st = _inputs(32, 7)
    with torch.no_grad():
        nom, _, _ = a(sc, st, torch.zeros(32, 16))                 # a fresh net's outputs: ~0.01
        v0, d0 = torch.full((32,), IN.V_REF), torch.zeros(32)
        L = IN.lat_1s(nom, a.intent_bias, v0, d0, lim)            # (K, 32)
        L0 = IN.lat_1s(nom, torch.zeros(1, PS.ACT_DIM), v0, d0, lim)[0]
    assert (L - L0 - want[:, None]).abs().max() < 0.015            # each intent its distance from the plan without one
    assert ((L - L[:1]) - (want - want[0])[:, None]).abs().max() < 0.015     # and from each other
    assert abs(float(IN.pair_spread(L).mean()) - float(IN.pair_spread(want[:, None]).mean())) < 0.01
    # the same speeds; the intents' vectors: one number on the 10 turn outputs (a steady steering offset), the speed outputs 0
    u = (nom[None] + a.intent_bias[:, None]).reshape(K * 32, -1)
    p = PS.roll(torch.tanh(u), v0.repeat(K), d0.repeat(K), lim)
    p0 = PS.roll(torch.tanh(nom), v0, d0, lim)
    assert torch.allclose(p["pv"], p0["pv"].repeat(K, 1), atol=1e-5)
    assert float(a.intent_bias.detach()[:, PS.N_STEP:].abs().max()) == 0.0
    assert torch.allclose(a.intent_bias[:, :PS.N_STEP], a.intent_bias[:, :1].expand(-1, PS.N_STEP))
    assert bool((a.intent_bias[:, 0].sign() == want.sign()).all())
    # large starts too, on the arc's outward branch (past ~2.4 m at 5 m/s the arc curls back: refused)
    big = IN.lat_1s(torch.zeros(1, PS.ACT_DIM), IN.init_bias(lim, 2, 2.0), torch.full((1,), IN.V_REF), torch.zeros(1), lim)[:, 0]
    assert (big - torch.tensor([-2.0, 2.0])).abs().max() < 0.01
    with pytest.raises(ValueError):
        IN.init_bias(lim, 2, 3.0)
    # about the same metres at other speeds (the turn output is a share of the tyres' grip): 3-7 m/s within 30 %
    for v in (3.0, 7.0):
        Lv = IN.lat_1s(torch.zeros(1, PS.ACT_DIM), a.intent_bias, torch.full((1,), v), torch.zeros(1), lim)[:, 0]
        assert ((Lv - want).abs() <= 0.3 * want.abs() + 0.01).all(), (v, Lv)


def test_the_update_log(lim):
    st = IN.IntentStats(K, "cpu", every=1, rows=8)
    bias = IN.init_bias(lim, K, 0.3)
    z = torch.tensor([0, 1, 2, 3, 3, 3]); due = torch.tensor([True, False, True, False, True, False])
    lg = torch.zeros(6, K); lg[:, 3] = 1.0
    mean_of = lambda idx, k: torch.zeros(idx.numel(), PS.ACT_DIM) + (0.0 if k is None else bias[k])
    st.decision(0, z, due, lg, torch.ones(6, dtype=torch.bool), mean_of, torch.full((6,), 5.0), torch.zeros(6), lim)
    r = st.row(bias)
    assert r["intent_redraws"] == 3 and abs(r["intent_off_argmax"] - 2 / 3) < 1e-6
    assert r["intent_share"] == [round(1 / 6, 4), round(1 / 6, 4), round(1 / 6, 4), 0.5]
    assert abs(r["intent_lat_spread"] - float(IN.pair_spread(torch.tensor([[-0.3], [-0.1], [0.1], [0.3]])))) < 0.01
    assert all(abs(a - b) < 0.01 for a, b in zip(r["intent_lat_ref"], [-0.3, -0.1, 0.1, 0.3]))
    assert "의도" in IN.log_line(r)


# ------------------------------------------------------------------ the trunk reads z (--intent-trunk)
def _trunk_pair(lim):
    torch.manual_seed(8)
    a = Actor2(BEAMS, memory=16, mem_front=True, intent=SPEC)
    torch.manual_seed(8)
    b = Actor2(BEAMS, memory=16, mem_front=True, intent=SPEC, intent_trunk=True)
    for x in (a, b):
        x.intent_init(lim, 0.3)
    with torch.no_grad():
        a.pi.weight.normal_(0.0, 0.3); b.pi.weight.copy_(a.pi.weight)     # a plan head with opinions
        a.intent_head.weight.normal_(0.0, 0.5); b.intent_head.weight.copy_(a.intent_head.weight)
    return a.eval(), b.eval()


def test_trunk_z_starts_as_the_vectors_only_actor(lim):
    a, b = _trunk_pair(lim)
    assert b.intent_trunk and float(b.intent_in.detach().abs().max()) == 0.0
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {"intent_in"} and all(torch.equal(sa[k], sb[k]) for k in sa)
    sc, st = _inputs(7, 3)
    h = torch.randn(7, 16) * 0.3
    ma, la, va = a(sc, st, h); mb, lb, vb = b(sc, st, h)
    assert torch.equal(ma, mb) and torch.equal(va, vb) and torch.equal(a.intent_logits_out, b.intent_logits_out)
    assert torch.equal(a.h_out, b.h_out)
    for k in range(K):
        z = torch.full((7,), k)
        assert torch.equal(a.intent_mean(ma, z), b.intent_mean(mb, z))      # bit for bit: a1 + 0


def test_the_trunk_reads_z_once_its_columns_move(lim):
    a, b = _trunk_pair(lim)
    with torch.no_grad():
        b.intent_in.normal_(0.0, 0.5)
    sc, st = _inputs(7, 4)
    h = torch.randn(7, 16) * 0.3
    ma, _, va = a(sc, st, h); mb, _, vb = b(sc, st, h)
    # z-free: the plan without z, pi_high, the speed estimate and the memory do not change with z's columns
    assert torch.allclose(ma, mb, atol=1e-6) and torch.allclose(va, vb, atol=1e-6)
    assert torch.allclose(a.intent_logits_out, b.intent_logits_out, atol=1e-6) and torch.allclose(a.h_out, b.h_out)
    # under z: beyond the vector, and differently from state to state (the control law is z-conditioned)
    d = torch.stack([b.intent_mean(mb, torch.full((7,), k)) - a.intent_mean(ma, torch.full((7,), k)) for k in range(K)])
    d = d.detach()
    assert float(d.abs().max()) > 1e-3
    assert float((d[:, :1] - d[:, 1:]).abs().max()) > 1e-4                  # not one offset for every state
    # the plan under z reads the trunk run it was drawn from (a1), not whatever ran last
    a1 = b.trunk_a1.clone()
    z = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    m1 = b.intent_mean(mb, z)
    b(*_inputs(7, 9), h)                                                     # another trunk run
    assert torch.allclose(b.intent_mean(None, z, a1=a1), m1)


def test_the_policy_gradient_reaches_z_s_columns_and_not_through_pi_high(lim):
    _, b = _trunk_pair(lim)
    b.train()
    sc, st = _inputs(6, 5)
    mean, log_std, _ = b(sc, st, torch.zeros(6, 16))
    z = torch.tensor([0, 1, 2, 3, 3, 0])
    rl = torch.tensor([0, PS.N_STEP])
    m = b.intent_mean(mean, z)
    u = (m + 0.1).detach()
    lp = tanh_logp(u[:, rl], m[:, rl], log_std[:, rl])
    g_in, g_head = torch.autograd.grad(lp.sum(), [b.intent_in, b.intent_head.weight], allow_unused=True)
    assert g_in is not None and float(g_in.abs().sum()) > 0
    assert g_head is None or float(g_head.abs().sum()) == 0.0               # pi_low's log-prob does not move pi_high


def test_load_and_copy_an_intent_trunk_actor(lim, monkeypatch):
    _, b = _trunk_pair(lim)
    with torch.no_grad():
        b.intent_in.normal_()
    ck = {"beams": BEAMS, "actor": b.state_dict(), "args": {"intent": SPEC, "memory": 16, "mem_front": True}}
    c = load_actor2(ck, "cpu", check_mount=False)
    assert c.intent_trunk and torch.equal(c.intent_in, b.intent_in)
    old = {k: v for k, v in b.state_dict().items() if k != "intent_in"}
    assert not load_actor2({"beams": BEAMS, "actor": old, "args": ck["args"]}, "cpu", check_mount=False).intent_trunk
    # a copy is made after an update's epochs: the logits and the trunk's pre-activation are graph tensors then
    b.train()
    sc, st = _inputs(3, 1)
    b.heads(b.mem_trunk(torch.cat([b.enc(sc), b.norm(st)], 1), torch.zeros(3, 16)))     # as the epochs run it
    assert b.trunk_a1.grad_fn is not None and b.intent_logits_out.grad_fn is not None
    monkeypatch.setattr(PR, "PlanObs2", _Obs)
    so = PR.SelfOpponents(_env(), lim, 1.0, seed=3)
    so.add(b, 100)
    assert b.trunk_a1 is not None and so.model(100).intent_trunk


def test_free_runs_the_z_free_plan(monkeypatch, lim):
    a, b = _trunk_pair(lim)
    with torch.no_grad():
        b.intent_in.normal_(0.0, 0.5)
    obs = _StubObs(5)
    monkeypatch.setenv(IN.ENV_FIX, "free")
    nom, _, _ = b(obs.scans, obs.state())
    _, mean, _, _, _ = plan_v2(b, obs, lim)
    assert torch.allclose(mean, nom, atol=1e-6) and bool((obs.intent_left == 0).all())
    monkeypatch.setenv(IN.ENV_FIX, "2")
    _, mean2, _, _, _ = plan_v2(b, obs, lim)
    assert bool((obs.intent_z == 2).all()) and float((mean2 - mean).detach().abs().max()) > 1e-4


def test_intent_trunk_refusals():
    with pytest.raises(ValueError):
        Actor2(BEAMS, memory=16, mem_front=False, intent=SPEC, intent_trunk=True)
    with pytest.raises(SystemExit) as e:
        PR.main(["--name", "x", "--intent", SPEC, "--memory", "16", "--check-args"])
    assert "--mem-front" in str(e.value.code)
    with pytest.raises(SystemExit) as e:
        PR.main(["--name", "x", "--intent", SPEC, "--memory", "16", "--no-intent-trunk", "--check-args"])
    assert e.value.code == 0
