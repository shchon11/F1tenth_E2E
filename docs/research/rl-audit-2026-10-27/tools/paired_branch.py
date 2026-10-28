"""Twin worlds (rl-audit 2026-10-28): how precisely does a common-random-numbers comparison measure what one action change
is worth, against the policy gradient's SNR 0.15 per 2048 samples for "faster" (snr_v2, section 2)?

ppo.main's own `make_env` is called twice with the same arguments, so the two envs have the same structure. Env A is the
master: the checkpoint drives it with its MEAN action, W0 steps first, then GAP steps between trials. At each trial env B is
made an exact copy of A (`twin_sync.sync`: every tensor, generator and number; the first trial checks it by stepping both
on the same command for VERIFY steps and listing what differs), then every learner row branches: A drives mean + DELTA on
one action column and B mean - DELTA for KP steps, then each its own mean action until H steps; both steps draw the same
global random numbers. Per row: the objective summed over the H steps (progress + steer + wrong-way - PRICE x charged hard
onsets, the reward `lagrange_step` hands PPO) in A minus in B. Rows share nothing across races, so every row gets its own
setting (SETS, cycled over rows); the DELTA 0 setting is the control (its spread is the non-determinism floor). Rows whose
race resets inside the branch are dropped.
usage: python paired_branch.py -- <ppo argv incl --init>
env: TRIALS (8), H (120), KP (20), W0 (400), GAP (200), VERIFY (40), PRICE (80),
     SETS (col:delta list, default "4:0,4:0.1,4:0.2,1:0.1,0:0.1,2:0.1"; col 4 = v1, 0..2 = kappa)"""
import os
import sys

import numpy as np
import torch

argv = sys.argv[sys.argv.index("--") + 1:]
sys.argv = ["ppo"] + argv
from f1sim.learn import common, ppo, ppo_v2  # noqa: E402
from f1sim.learn import conditioning as cond_mod  # noqa: E402
from f1sim.learn.memory import reset_hidden  # noqa: E402
from f1sim.learn.model import load_checkpoint  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1sim.learn.twin_sync import diff, sync  # noqa: E402

TRIALS = int(os.environ.get("TRIALS", "8"))
H = int(os.environ.get("H", "120"))
KP = int(os.environ.get("KP", "20"))
W0, GAP, VERIFY = int(os.environ.get("W0", "400")), int(os.environ.get("GAP", "200")), int(os.environ.get("VERIFY", "40"))
PRICE = float(os.environ.get("PRICE", "80"))
NOISE = os.environ.get("NOISE", "") == "1"     # drive with the policy's own exploration noise, the SAME draw in both twins
SETS = [(int(c), float(d)) for c, d in (s.split(":") for s in
        os.environ.get("SETS", "4:0,4:0.1,4:0.2,1:0.1,0:0.1,2:0.1").split(","))]

_twins = []
_make = common.make_env


def _make_twice(*args, **kw):
    env = _make(*args, **kw)
    _twins.append(_make(*args, **kw))          # the same arguments and seed: the same world
    return env


common.make_env = _make_twice


def _rng_state():
    return torch.get_rng_state(), torch.cuda.get_rng_state_all()


def _set_rng_state(s):
    torch.set_rng_state(s[0]); torch.cuda.set_rng_state_all(s[1])


def train(a, *, env, obs, device, names, init_ck, controller, graph_rt, applied, league_seats=()):
    envB = _twins[0]
    if a.sim_backend == "graphs":
        from f1sim.learn.graph_runtime import prepare_graph_runtime
        prepare_graph_runtime(envB)
    model, extra = load_checkpoint(a.init, device)
    actor = model.actor
    lid = env.learner_ids
    if not torch.equal(lid, envB.learner_ids):
        raise SystemExit("the twin env's learner rows differ")
    Bl = lid.numel()
    cspec = cond_mod.CondSpec.from_meta(model.meta["cond"])
    dial = cond_mod.ConstantDial(env, a.dial_constant)
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp)
    w_col = abs(env.ecfg.reward_collision)
    n_set = len(SETS)
    set_of = torch.arange(Bl, device=device) % n_set
    col_of = torch.tensor([c for c, _ in SETS], device=device)[set_of]
    dlt_of = torch.tensor([d for _, d in SETS], device=device)[set_of]
    pert = torch.zeros(Bl, env.act_dim, device=device)
    pert[torch.arange(Bl, device=device), col_of] = dlt_of

    dead = [int(x) for x in str(getattr(a, "pg_dead_dims", "") or "").split(",") if x.strip()]
    live_m = torch.ones(env.act_dim, dtype=torch.bool, device=device); live_m[dead] = False
    sd = actor.log_std.detach().exp().float()
    eps_now = [None]

    def mean_action(o, h):
        """the mean action -- plus, with NOISE=1, sd x the step's shared draw on the live columns (PPO's rollout)"""
        c = cond_mod.mu_to_c(dial.value(), cspec).to(o["ego"].dtype)[lid]
        with torch.no_grad(), ac:
            d, hn = actor.step_dist(o["scan"][lid], o["ego"][lid], c, h)
        mu = d.mean.float()
        if NOISE and eps_now[0] is not None:
            mu = mu + torch.where(live_m, sd * eps_now[0], torch.zeros_like(mu))
        return mu, hn.float()

    def draw():
        eps_now[0] = torch.randn(Bl, env.act_dim, device=device) if NOISE else None

    def objective(rew, info, touching_prev):
        comp = info["reward_components"]
        col = comp["collision"][lid].float(); car = comp["car_contact"][lid].float()
        touching = car < 0
        onset = touching & ~touching_prev
        onw = (-col / max(w_col, 1e-9)).clamp_min(0)
        r = rew[lid].float() - col - car - PRICE * onw - PRICE * onset.float()
        return r, touching, onw + onset.float(), info["progress"][lid].float()

    from f1sim.learn.train_v2 import ValueNorm
    vn = ValueNorm(0.99)
    if extra.get("value_norm"):
        vn.load_state_dict(extra["value_norm"])

    def critic_value(e, o):
        """the checkpoint's own critic, de-normalised: what GAE would read at this state"""
        c = cond_mod.mu_to_c(dial.value(), cspec).to(o["ego"].dtype)[lid]
        with torch.no_grad(), ac:
            v_n, _ = model.critic(o["scan"][lid], o["ego"][lid], c, e.privileged_v2()[lid])
        return vn.denormalize(v_n.float()).reshape(-1)

    def contacts(e, prev_car):
        """this step's contact onsets per learner row with their impact speeds (wall, prop, car; 0 = none)"""
        r = e.last_result
        on, imp = r.contact_onsets[lid][:, :2], r.contact_impacts[lid][:, :2].float()
        car = r.car_collision[lid] if r.car_collision is not None else torch.zeros_like(prev_car)
        car_on = car & ~prev_car
        car_imp = r.car_impact[lid].float() if r.car_impact is not None else torch.zeros(Bl, device=device)
        imps = torch.cat([torch.where(on, imp, torch.zeros_like(imp)), torch.where(car_on, car_imp, 0 * car_imp)[:, None]], 1)
        return imps, car

    def step_both(fullA, fullB):
        s = _rng_state()
        outA = env.step(fullA.clamp(-1, 1))
        _set_rng_state(s)                       # the twin draws what A drew
        outB = envB.step(fullB.clamp(-1, 1))
        return outA, outB

    if os.environ.get("DIAG"):
        from f1sim.learn.twin_sync import covered, reach
        allA, cov = reach(env), covered(env, envB)
        miss = sorted((v for k, v in allA.items() if k not in cov), key=lambda x: x[0])
        print(f"DIAG: {len(allA)} tensors reachable from env A, {len(cov)} visited by sync, {len(miss)} missed:", flush=True)
        for m_ in miss:
            print("   ", m_, flush=True)
        raise SystemExit(0)
    for e in (env, envB):
        e.sim.warmup()
    fullA = torch.zeros(env.B, env.act_dim, device=device)
    fullB = torch.zeros(env.B, env.act_dim, device=device)
    oA = obs
    h = actor.initial_hidden(Bl, device=device)

    def drive_master(n, o, h):
        for _ in range(n):
            draw()
            mu, hn = mean_action(o, h)
            fullA[lid] = mu
            o, r, te, tr, i = env.step(fullA.clamp(-1, 1))
            h = reset_hidden(hn, (te | tr)[lid])
        return o, h

    # the twin walks the warm-up in lockstep on A's command, so every lazily created piece of state exists in both
    s0 = _rng_state(); envB.reset(seed=int(a.seed)); _set_rng_state(s0)
    for _ in range(W0):
        draw()
        mu, hn = mean_action(oA, h)
        fullA[lid] = mu
        (oA, rA, tA, trA, iA), _ = step_both(fullA, fullA)
        h = reset_hidden(hn, (tA | trA)[lid])
    rec = {k: [] for k in ("set", "dJ", "dprog", "dhits", "hitsA", "hitsB", "v0", "nearA", "nearB", "allA", "allB",
                           "maxA", "maxB", "hkA", "hkB")}
    for trial in range(TRIALS):
        n = sync(env, envB)
        if trial == 0:
            print(f"sync: {n}", flush=True)
            oB = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in oA.items()}
            d0 = diff(env, envB)
            print(f"right after the sync, {len(d0)} differing paths: {d0[:8]}", flush=True)
            hA_ = h.clone()
            for k in range(VERIFY):
                draw()
                mu, hn = mean_action(oA, hA_)
                fullA[lid] = mu
                (oA, rA, tA, trA, iA), (oB, rB, tB, trB, iB) = step_both(fullA, fullA)
                hA_ = reset_hidden(hn, (tA | trA)[lid])
                if k == 0:
                    print(f"  after the first lockstep step, differing paths: {diff(env, envB)[:16]}", flush=True)
                if k in (0, 1, 4, VERIFY - 1):
                    print(f"  verify step {k + 1}: state max diff {float((env.sim.state - envB.sim.state).abs().max()):.3e}, "
                          f"scan {float((oA['scan'].float() - oB['scan'].float()).abs().max()):.3e}, "
                          f"reward {float((rA - rB).abs().max()):.3e}", flush=True)
            d1 = diff(env, envB)
            print(f"after {VERIFY} lockstep steps, {len(d1)} differing paths: {d1[:12]}", flush=True)
            h = hA_
            sync(env, envB)
        oB = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in oA.items()}
        v0 = env.sim.state[lid, 3].float().clone()
        hA, hB = h.clone(), h.clone()
        J = torch.zeros(2, Bl, device=device); P = torch.zeros(2, Bl, device=device); N = torch.zeros(2, Bl, device=device)
        tp = [torch.zeros(Bl, dtype=torch.bool, device=device) for _ in range(2)]
        reset_in = torch.zeros(Bl, dtype=torch.bool, device=device)
        pc = [torch.zeros(Bl, dtype=torch.bool, device=device) for _ in range(2)]
        NEAR = torch.zeros(2, Bl, device=device); ALL = torch.zeros(2, Bl, device=device); MX = torch.zeros(2, Bl, device=device)
        HK = torch.zeros(2, Bl, 3, device=device)               # raw hard (>= 1.5 m/s) onsets per kind: wall, prop, car
        for k in range(H):
            draw()                                           # one draw, both twins: common random numbers
            muA, hnA = mean_action(oA, hA)
            muB, hnB = mean_action(oB, hB)
            if k < KP:
                muA = muA + pert; muB = muB - pert
            fullA[lid] = muA; fullB[lid] = muB
            (oA, rA, tA, trA, iA), (oB, rB, tB, trB, iB) = step_both(fullA, fullB)
            for j, (r_, i_) in enumerate(((rA, iA), (rB, iB))):
                r, tp[j], hits, prog = objective(r_, i_, tp[j])
                J[j] += r; P[j] += prog; N[j] += hits
                imps, pc[j] = contacts(env if j == 0 else envB, pc[j])
                NEAR[j] += ((imps >= 1.0) & (imps < 1.5)).sum(1).float(); ALL[j] += (imps > 0).sum(1).float()
                MX[j] = torch.maximum(MX[j], imps.max(1).values)
                HK[j] += (imps >= 1.5).float()
            dA, dB = (tA | trA)[lid], (tB | trB)[lid]
            reset_in |= dA | dB
            tp = [tp[0] & ~dA, tp[1] & ~dB]
            pc = [pc[0] & ~dA, pc[1] & ~dB]
            hA, hB = reset_hidden(hnA, dA), reset_hidden(hnB, dB)
        ok = ~reset_in
        vA_end, vB_end = critic_value(env, oA), critic_value(envB, oB)
        for k_, v_ in (("dV", vA_end - vB_end),):
            rec.setdefault(k_, []).append(v_[ok].cpu().numpy())
        for k_, v_ in (("set", set_of), ("dJ", J[0] - J[1]), ("dprog", P[0] - P[1]), ("dhits", N[0] - N[1]),
                       ("hitsA", N[0]), ("hitsB", N[1]), ("v0", v0), ("nearA", NEAR[0]), ("nearB", NEAR[1]),
                       ("allA", ALL[0]), ("allB", ALL[1]), ("maxA", MX[0]), ("maxB", MX[1]),
                       ("hkA", HK[0]), ("hkB", HK[1])):
            rec[k_].append(v_[ok].cpu().numpy())
        print(f"trial {trial}: {int(ok.sum())}/{Bl} rows without a reset in the branch", flush=True)
        h = hA
        oA, h = drive_master(GAP, oA, h)
    R = {k: np.concatenate(v) for k, v in rec.items()}
    snr_pg = 0.15 / np.sqrt(2048) * np.sqrt(2 * H)
    print(f"\n== H {H} steps ({H / 40:.1f} s), perturbation for the first {KP} steps; objective with price {PRICE:g}")
    print("setting | rows | dJ mean +- se | dJ sd | dprog mean (m) | dprog sd | rows with a hit flip | dJ/(2 delta) | pair SNR")
    for i, (c, d) in enumerate(SETS):
        m = R["set"] == i
        dj, dp, dh = R["dJ"][m], R["dprog"][m], R["dhits"][m]
        n = m.sum()
        se = dj.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        slope = dj.mean() / (2 * d) if d else float("nan")
        snr = abs(dj.mean()) / dj.std(ddof=1) if n > 1 and dj.std() > 0 else float("nan")
        name = {4: "v1", 0: "k0", 1: "k1", 2: "k2", 3: "k3"}.get(c, str(c))
        print(f"{name} {d:+.2f} | {n} | {dj.mean():+.3f} +- {se:.3f} | {dj.std(ddof=1):.3f} | {dp.mean():+.4f} | {dp.std(ddof=1):.4f} | "
              f"{(dh != 0).mean() * 100:.1f} % | {slope:+.3f} | {snr:.2f}")
        if "dV" in R:
            dv = R["dV"][m]
            print(f"   critic at the branch end: V(A) - V(B) {dv.mean():+.3f} +- {dv.std(ddof=1) / np.sqrt(n):.3f} "
                  f"-> dJ + dV {(dj + dv).mean():+.3f} (what a critic-bootstrapped advantage would see)")
        if d:
            fast = R["v0"][m] > 6.0
            if fast.sum() > 2:
                print(f"   on straights (> 6 m/s, {fast.sum()} rows): dJ {dj[fast].mean():+.3f} +- "
                      f"{dj[fast].std(ddof=1) / np.sqrt(fast.sum()):.3f}, dprog {dp[fast].mean():+.4f} m"
                      + (f", critic dV {R['dV'][m][fast].mean():+.3f} +- "
                         f"{R['dV'][m][fast].std(ddof=1) / np.sqrt(fast.sum()):.3f}" if "dV" in R else ""))
    flip = R["dhits"] != 0
    if flip.any():
        lowA = R["hitsA"] < R["hitsB"]                                    # the twin with fewer hard hits ...
        near_low = np.where(lowA, R["nearA"], R["nearB"]) > 0
        all_low = np.where(lowA, R["allA"], R["allB"]) > 0
        mx_low = np.where(lowA, R["maxA"], R["maxB"])
        mx_high = np.where(lowA, R["maxB"], R["maxA"])
        f = flip
        print(f"hit flips: {f.sum()} rows of {len(f)} ({f.mean() * 100:.1f} %). In the twin with FEWER hard hits: a contact "
              f"of 1.0-1.5 m/s in {near_low[f].mean() * 100:.0f} %, any contact in {all_low[f].mean() * 100:.0f} %, none in "
              f"{(~all_low[f]).mean() * 100:.0f} %; its fastest onset median {np.median(mx_low[f]):.2f} m/s vs the other twin's "
              f"{np.median(mx_high[f]):.2f} m/s")
        extra_hits = np.abs(R["hkA"][f] - R["hkB"][f]).sum(0)          # the raw hard onsets the twins disagree on, per kind
        print(f"   what the flipped hits were (raw hard onsets the twins disagree on): wall {extra_hits[0]:.0f}, "
              f"prop {extra_hits[1]:.0f}, car {extra_hits[2]:.0f}")
        imp_all = np.concatenate([R["maxA"][R["allA"] > 0], R["maxB"][R["allB"] > 0]])
        if imp_all.size:
            q = np.percentile(imp_all, [10, 25, 50, 75, 90])
            print(f"fastest onset per row with any contact: p10/25/50/75/90 = {' / '.join(f'{x:.2f}' for x in q)} m/s; "
                  f"share within 1.2-1.8 m/s {((imp_all >= 1.2) & (imp_all < 1.8)).mean() * 100:.0f} %")
    print(f"policy gradient for comparison: SNR {snr_pg:.3f} for the {2 * H} learner samples one pair costs "
          f"(0.15 per 2048, snr_v2)")
    raise SystemExit(0)


ppo_v2.train = train
ppo.main()
