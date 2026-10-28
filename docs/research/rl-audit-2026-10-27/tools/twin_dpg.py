"""Twin-world policy gradient (rl-audit 2026-10-28): train the DEPLOYED (mean-action) policy from exact paired comparisons.

Why: PPO's per-sample signal for an action's worth is the return's noise plus a 2 cm effect (SNR ~0.003 per sample for
"faster", snr_v2); paired_branch.py measured that two twin envs that share every random number give the difference two
action choices make with no noise at all (a delta 0 pair differs by exactly 0) -- per pair SNR 1.3 for the same 80 samples
where the policy gradient has 0.03.

How: env A (the master) and its twin B are ppo.main's own world built twice. The checkpoint drives A with its mean action.
Every branch: B := A (`twin_sync.sync`), each learner row draws z ~ N(0, I) over the live action columns, A drives
mean + DELTA*z and B mean - DELTA*z for KP steps, then each its own mean action until H steps. dJ = the objective over the
H steps (progress + steer + wrong-way - PRICE x charged hard onsets, what `lagrange_step` hands PPO) in A minus in B, so
(dJ / 2) z / DELTA estimates the gradient of the H-step return w.r.t. the mean action at the branch's first state
(Gaussian smoothing; exact up to the shared future). The actor then takes Adam steps on
    loss = - mean_i  sum_d  g_id * mu_d(s_i),     g = (dJ/2) z / DELTA / sd(dJ/2)
i.e. a deterministic policy gradient whose action gradient is measured, not learned. No exploration noise, no critic.
A keeps driving from the branch's end (its next state), so the states are the deployed policy's own.
usage: python twin_dpg.py -- <ppo argv incl --init, --name>; run it through fixed_world.sh (the fixed training world)
env: UPDATES (400), H (120), KP (10), DELTAS (per live column, default 0.03,0.1,0.15,0.1 for kappa0..2, v1), LR (2e-5),
     EPOCHS (1), MAX_GRAD (0.5), PRICE (80), W0 (200), SAVE (25), OUT (~/f1sim_runs/<name>)"""
import json
import os
import sys
import time

import numpy as np
import torch

argv = sys.argv[sys.argv.index("--") + 1:]
sys.argv = ["ppo"] + argv
from f1sim.learn import common, ppo, ppo_v2  # noqa: E402
from f1sim.learn import conditioning as cond_mod  # noqa: E402
from f1sim.learn.memory import reset_hidden  # noqa: E402
from f1sim.learn.model import load_checkpoint  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1sim.learn.twin_sync import sync  # noqa: E402

UPDATES = int(os.environ.get("UPDATES", "400"))
H = int(os.environ.get("H", "120"))
KP = int(os.environ.get("KP", "10"))
LR = float(os.environ.get("LR", "2e-5"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
MAX_GRAD = float(os.environ.get("MAX_GRAD", "0.5"))
PRICE = float(os.environ.get("PRICE", "80"))
W0 = int(os.environ.get("W0", "200"))
SAVE = int(os.environ.get("SAVE", "25"))

_twins = []
_make = common.make_env


def _make_twice(*args, **kw):
    env = _make(*args, **kw)
    _twins.append(_make(*args, **kw))
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
    out = os.path.expanduser(os.environ.get("OUT", f"~/f1sim_runs/{a.name}"))
    os.makedirs(out, exist_ok=True)
    ck0 = torch.load(a.init, map_location="cpu", weights_only=False)
    model, extra = load_checkpoint(a.init, device)
    actor = model.actor
    dead = [int(x) for x in str(getattr(a, "pg_dead_dims", "") or "").split(",") if x.strip()]
    live = [d for d in range(env.act_dim) if d not in dead]
    deltas = [float(x) for x in os.environ.get("DELTAS", "0.03,0.1,0.15,0.1").split(",")]
    if len(deltas) != len(live):
        raise SystemExit(f"DELTAS needs one value per live column {live}")
    delta = torch.tensor(deltas, device=device)
    params = [p for n, p in actor.named_parameters() if n != "log_std" and p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR, eps=1e-8)
    lid = env.learner_ids
    if not torch.equal(lid, envB.learner_ids):
        raise SystemExit("the twin env's learner rows differ")
    Bl = lid.numel()
    cspec = cond_mod.CondSpec.from_meta(model.meta["cond"])
    dial = cond_mod.ConstantDial(env, a.dial_constant)
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp)
    w_col = abs(env.ecfg.reward_collision)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump({**{k: v for k, v in vars(a).items() if isinstance(v, (int, float, str, bool, type(None)))},
                   "twin_dpg": {"H": H, "KP": KP, "deltas": deltas, "live": live, "lr": LR, "epochs": EPOCHS,
                                "max_grad": MAX_GRAD, "price": PRICE}}, f, indent=1)
    log = open(os.path.join(out, "progress.jsonl"), "a")

    def inputs(o):
        c = cond_mod.mu_to_c(dial.value(), cspec).to(o["ego"].dtype)[lid]
        return o["scan"][lid], o["ego"][lid], c

    def mean_action(o, h):
        scan, ego, c = inputs(o)
        with torch.no_grad(), ac:
            d, hn = actor.step_dist(scan, ego, c, h)
        return d.mean.float(), hn.float()

    def objective(rew, info, touching_prev):
        comp = info["reward_components"]
        col = comp["collision"][lid].float(); car = comp["car_contact"][lid].float()
        touching = car < 0
        onset = touching & ~touching_prev
        onw = (-col / max(w_col, 1e-9)).clamp_min(0)
        r = rew[lid].float() - col - car - PRICE * onw - PRICE * onset.float()
        return r, touching, onw + onset.float(), info["progress"][lid].float()

    def step_both(fullA, fullB):
        s = _rng_state()
        outA = env.step(fullA.clamp(-1, 1))
        _set_rng_state(s)
        outB = envB.step(fullB.clamp(-1, 1))
        return outA, outB

    def save(u):
        ex = dict(ck0.get("extra") or {})
        ex.update(run=a.name, update=u, phase="ppo", twin_dpg={"H": H, "KP": KP, "deltas": deltas, "lr": LR})
        ex.pop("opt", None); ex.pop("opt_param_names", None)
        torch.save({"state_dict": model.state_dict(), "meta": model.meta, "extra": ex}, os.path.join(out, f"ppo_u{u}.pt"))

    for e in (env, envB):
        e.sim.warmup()
    fullA = torch.zeros(env.B, env.act_dim, device=device)
    fullB = torch.zeros(env.B, env.act_dim, device=device)
    oA = obs
    h = actor.initial_hidden(Bl, device=device)
    s0 = _rng_state(); envB.reset(seed=int(a.seed)); _set_rng_state(s0)
    for _ in range(W0):                      # lockstep: every lazily created piece of state exists in both
        mu, hn = mean_action(oA, h)
        fullA[lid] = mu
        (oA, rA, tA, trA, iA), _ = step_both(fullA, fullA)
        h = reset_hidden(hn, (tA | trA)[lid])
    save(0)
    t_start = time.time()
    live_t = torch.tensor(live, device=device)
    for u in range(1, UPDATES + 1):
        t0 = time.time()
        sync(env, envB)
        oB = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in oA.items()}
        scan0, ego0, c0 = (x.clone() for x in inputs(oA))
        h0 = h.clone()
        z = torch.randn(Bl, len(live), device=device)
        pert = torch.zeros(Bl, env.act_dim, device=device)
        pert[:, live_t] = z * delta
        hA, hB = h.clone(), h.clone()
        J = torch.zeros(2, Bl, device=device); P = torch.zeros(2, Bl, device=device); N = torch.zeros(2, Bl, device=device)
        tp = [torch.zeros(Bl, dtype=torch.bool, device=device) for _ in range(2)]
        reset_in = torch.zeros(Bl, dtype=torch.bool, device=device)
        actor.eval()
        for k in range(H):
            muA, hnA = mean_action(oA, hA)
            muB, hnB = mean_action(oB, hB)
            if k < KP:
                muA = muA + pert; muB = muB - pert
            fullA[lid] = muA; fullB[lid] = muB
            (oA, rA, tA, trA, iA), (oB, rB, tB, trB, iB) = step_both(fullA, fullB)
            for j, (r_, i_) in enumerate(((rA, iA), (rB, iB))):
                r, tp[j], hits, prog = objective(r_, i_, tp[j])
                J[j] += r; P[j] += prog; N[j] += hits
            dA, dB = (tA | trA)[lid], (tB | trB)[lid]
            reset_in |= dA | dB
            tp = [tp[0] & ~dA, tp[1] & ~dB]
            hA, hB = reset_hidden(hnA, dA), reset_hidden(hnB, dB)
        h = hA                               # A drives on from the branch's end
        t_roll = time.time() - t0
        ok = ~reset_in
        dJ = (J[0] - J[1])[ok]
        n = int(ok.sum())
        flips = float(((N[0] - N[1])[ok] != 0).float().mean()) if n else 0.0
        stats = {"update": u, "n": n, "dJ_mean": float(dJ.mean()) if n else 0.0, "dJ_sd": float(dJ.std()) if n > 1 else 0.0,
                 "hit_flip_frac": flips, "master_prog_ms": float(P[0].sum() / (H / 40.0) / Bl),
                 "master_hits_per_km": float(N[0].sum() / max(float(P[0].clamp_min(0).sum()), 1e-3) * 1000),
                 "master_J_per_step": float(J[0].mean() / H), "time_branch_s": t_roll}
        if n > 8:
            half = dJ / 2
            sd = float(half.std()) + 1e-6
            g = (half[:, None] * z[ok] / delta[None, :]) / sd                    # (n, live): d return / d mean action
            actor.train()
            mu_before = None
            for ep in range(EPOCHS):
                opt.zero_grad(set_to_none=True)
                with ac:
                    d_, _ = actor.step_dist(scan0[ok], ego0[ok], c0[ok], h0[:, ok])
                mu = d_.mean.float()[:, live_t]
                if mu_before is None:
                    mu_before = mu.detach().clone()
                loss = -(g * mu).sum(1).mean()
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(params, MAX_GRAD))
                opt.step()
            with torch.no_grad(), ac:
                d_, _ = actor.step_dist(scan0[ok], ego0[ok], c0[ok], h0[:, ok])
            move = (d_.mean.float()[:, live_t] - mu_before).abs().mean(0)
            # the step along the measured gradient: positive = the change went where the twins said J rises
            along = float(((d_.mean.float()[:, live_t] - mu_before) * g).sum(1).mean())
            stats.update(loss=float(loss), grad_norm=gn, along=along,
                         **{f"move_{['k0', 'k1', 'k2', 'k3', 'v1', 'v2', 'v3', 'v4'][d]}": float(m) for d, m in zip(live, move)})
        stats["time_update_s"] = time.time() - t0 - t_roll
        stats["elapsed_min"] = (time.time() - t_start) / 60
        log.write(json.dumps(stats) + "\n"); log.flush()
        if u % 5 == 0 or u == 1:
            print(f"upd {u}/{UPDATES} | n {n} | dJ {stats['dJ_mean']:+.3f} sd {stats['dJ_sd']:.3f} | flips {flips * 100:.1f} % | "
                  f"master {stats['master_prog_ms']:.2f} m/s, {stats['master_hits_per_km']:.2f} hits/km, J {stats['master_J_per_step']:.4f} | "
                  f"along {stats.get('along', 0):+.2e} | {t_roll:.1f}+{stats['time_update_s']:.1f} s", flush=True)
        if u % SAVE == 0:
            save(u)
    save(UPDATES)
    torch.save({"state_dict": model.state_dict(), "meta": model.meta, "extra": dict(ck0.get("extra") or {}, run=a.name)},
               os.path.join(out, "ppo_final.pt"))
    print(f"{a.name} exit 0", flush=True)
    raise SystemExit(0)


ppo_v2.train = train
ppo.main()
