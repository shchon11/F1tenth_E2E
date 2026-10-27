"""Does the critic see a hard hit coming? (rl-audit-2026-10-27, after rlc3: the deployed policy does not improve while PPO
learns from ~5 hard hits per update.) A checkpoint drives the training env (its own argv after `--`, with its training noise
unless F1SIM_ROLLOUT_MEAN=1) for STEPS steps; each learner step records the checkpoint's own critic value V (de-normalised)
and the reward PPO optimises (progress + steer + wrong-way - PRICE x charged hard onsets, as `lagrange_step`). Then the TD
error delta_t = r_t + gamma V_{t+1} - V_t is aligned on every charged hard onset (wall/prop or car): if the critic anticipates
the hit, the -PRICE is spread over the steps BEFORE it (V falls as danger rises) and delta at the onset is small; if not, it
lands at the onset and GAE can only smear it back over ~1/(1-gamma*lambda) steps. Also the contact head's P(car contact
within 1 s) before car onsets. usage: python hit_anticipation.py STEPS -- <ppo argv incl --init>   env: PRICE (80), GAMMA (0.997)"""
import os
import sys

import numpy as np
import torch

STEPS = int(sys.argv[1])
argv = sys.argv[sys.argv.index("--") + 1:]
sys.argv = ["ppo"] + argv
from f1sim.gym_env import REWARD_COMPONENT_KEYS  # noqa: E402
from f1sim.learn import ppo, ppo_v2  # noqa: E402
from f1sim.learn import conditioning as cond_mod  # noqa: E402
from f1sim.learn.memory import reset_hidden  # noqa: E402
from f1sim.learn.model import load_checkpoint  # noqa: E402
from f1sim.learn.train_v2 import ValueNorm  # noqa: E402

PRICE = float(os.environ.get("PRICE", "80"))
GAMMA = float(os.environ.get("GAMMA", "0.997"))
PRE, POST = 60, 10


def train(a, *, env, obs, device, names, init_ck, controller, graph_rt, applied, league_seats=()):
    model, extra = load_checkpoint(a.init, device)
    if float(getattr(a, "log_std_shift", 0.0) or 0.0):
        with torch.no_grad():
            model.actor.log_std.add_(float(a.log_std_shift))
    vn = ValueNorm(0.99)
    vn.load_state_dict(extra["value_norm"])
    actor, critic = model.actor, model.critic
    lid = env.learner_ids
    B = lid.numel()
    dial = cond_mod.ConstantDial(env, a.dial_constant)
    cspec = cond_mod.CondSpec.from_meta(model.meta["cond"])
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp)
    dead = [int(x) for x in str(getattr(a, "pg_dead_dims", "") or "").split(",") if x.strip()]
    live = torch.ones(env.act_dim, dtype=torch.bool, device=device); live[dead] = False
    h = actor.initial_hidden(B, device=device)
    act_full = torch.zeros(env.B, env.act_dim, device=device)
    w_col, w_car = abs(env.ecfg.reward_collision), abs(env.ecfg.reward_car_contact)
    touching_prev = torch.zeros(B, dtype=torch.bool, device=device)
    env.sim.warmup()
    V, R, D, ONW, ONC, PC = [], [], [], [], [], []
    for t in range(STEPS):
        c = cond_mod.mu_to_c(dial.value(), cspec).to(obs["ego"].dtype)[lid]
        scan, ego = obs["scan"][lid], obs["ego"][lid]
        priv = env.privileged_v2()[lid]
        with torch.no_grad(), ac:
            d, hn = actor.step_dist(scan, ego, c, h)
            act = d.mean if ppo_v2.ROLLOUT_MEAN else d.sample()
            if dead:
                act = torch.where(live, act, d.mean)
            v_n, logit = critic(scan, ego, c, priv)
        V.append(vn.denormalize(v_n.float()).reshape(-1))
        PC.append(torch.sigmoid(logit.float()).reshape(-1) if logit is not None else torch.zeros(B, device=device))
        act_full[lid] = act.float()
        obs, rew, term, trunc, info = env.step(act_full.clamp(-1, 1))
        comp = info["reward_components"]
        col = comp["collision"][lid].float()
        car = comp["car_contact"][lid].float()
        touching = car < 0
        onset = touching & ~touching_prev
        done = (term | trunc)[lid]
        touching_prev = touching & ~done
        onw = (-col / max(w_col, 1e-9)).clamp_min(0)
        r = rew[lid].float() - col - car - PRICE * onw - PRICE * onset.float()
        R.append(r); D.append(done); ONW.append(onw > 0); ONC.append(onset)
        h = reset_hidden(hn.float(), done)
    V, R, D = torch.stack(V).cpu().numpy(), torch.stack(R).cpu().numpy(), torch.stack(D).cpu().numpy()
    ONW, ONC, PC = torch.stack(ONW).cpu().numpy(), torch.stack(ONC).cpu().numpy(), torch.stack(PC).cpu().numpy()
    T = V.shape[0]
    delta = np.full_like(R, np.nan)
    delta[:-1] = R[:-1] + GAMMA * V[1:] * (~D[:-1]) - V[:-1]
    ep = np.cumsum(np.vstack([np.zeros((1, R.shape[1]), bool), D[:-1]]), 0)   # episode id per (t, row)
    base = np.nanmean(delta[np.isfinite(delta)])
    print(f"\nHIT ANTICIPATION {os.path.basename(os.path.dirname(a.init))}/{os.path.basename(a.init)} "
          f"({'mean action' if ppo_v2.ROLLOUT_MEAN else 'training noise'}): {T} steps x {R.shape[1]} rows | "
          f"mean delta {base:+.4f} | price {PRICE}")
    for name, ON in (("wall/prop", ONW), ("car", ONC)):
        ts, rs = np.nonzero(ON)
        rows = []
        for t0, r0 in zip(ts, rs):
            if t0 - PRE < 0 or t0 + POST >= T - 1:
                continue
            if (ep[t0 - PRE:t0 + POST + 1, r0] != ep[t0, r0]).any():
                continue
            rows.append((delta[t0 - PRE:t0 + POST + 1, r0] - base, V[t0 - PRE:t0 + POST + 1, r0], PC[t0 - PRE:t0 + POST + 1, r0]))
        if not rows:
            print(f"  {name}: no onsets with a full window"); continue
        dl = np.stack([x[0] for x in rows]); vv = np.stack([x[1] for x in rows]); pc = np.stack([x[2] for x in rows])
        at = dl[:, PRE].mean()
        before = dl[:, :PRE].sum(1).mean()
        before10 = dl[:, PRE - 10:PRE].sum(1).mean()
        after = dl[:, PRE + 1:].sum(1).mean()
        print(f"  {name}: {len(rows)} onsets | excess delta: at the onset step {at:+.1f}, summed over the {PRE} steps before {before:+.1f} "
              f"(last 10: {before10:+.1f}), {POST} after {after:+.1f}  (an unanticipated hit puts ~-{PRICE:.0f} at the onset)")
        print(f"     V at -60/-30/-10/-1/0/+5 steps: " + " ".join(f"{vv[:, PRE + k].mean():.1f}" for k in (-60, -30, -10, -1, 0, 5))
              + f" | P(car contact in 1 s) at -30/-10/-1: " + " ".join(f"{pc[:, PRE + k].mean():.2f}" for k in (-30, -10, -1)))
    raise SystemExit(0)


ppo_v2.train = train
ppo.main()
