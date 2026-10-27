"""Is the critic optimistic where the policy goes? (rl-audit-2026-10-27, after rlc_s3001 lowered its own training value 36 %
in 192 updates while getting faster solo.) A checkpoint's actor drives the training env WITH its training noise for STEPS steps;
each learner step records the checkpoint's own critic value V(s) (de-normalised with the checkpoint's value normaliser) and the
reward the PPO actually optimises (hard wall/prop onsets and charged car onsets at the price PRICE, as `lagrange_step` does).
After the run, the realised discounted return G_t = sum_k gamma^k r(t+k) is computed for steps with HORIZON steps of future in
the same episode (gamma^HORIZON ~ 0.03 at 1200 steps), and V - G is reported overall and by speed / situation.
V - G > 0: the critic expects more than the policy gets (optimistic). usage: python critic_bias.py STEPS -- <ppo argv incl --init>
env: PRICE (80), HORIZON (1200), GAMMA (0.997)."""
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
HORIZON = int(os.environ.get("HORIZON", "1200"))
GAMMA = float(os.environ.get("GAMMA", "0.997"))


def train(a, *, env, obs, device, names, init_ck, controller, graph_rt, applied, league_seats=()):
    model, extra = load_checkpoint(a.init, device)
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
    i_col, i_car = REWARD_COMPONENT_KEYS.index("collision"), REWARD_COMPONENT_KEYS.index("car_contact")
    w_col, w_car = abs(env.ecfg.reward_collision), abs(env.ecfg.reward_car_contact)
    touching_prev = torch.zeros(B, dtype=torch.bool, device=device)
    env.sim.warmup()
    V, R, D, SPD, NEAR = [], [], [], [], []
    for t in range(STEPS):
        c = cond_mod.mu_to_c(dial.value(), cspec).to(obs["ego"].dtype)[lid]
        scan, ego = obs["scan"][lid], obs["ego"][lid]
        priv = env.privileged_v2()[lid]
        with torch.no_grad(), ac:
            d, hn = actor.step_dist(scan, ego, c, h)
            act = d.sample()
            if dead:
                act = torch.where(live, act, d.mean)
            v_n, _ = critic(scan, ego, c, priv)
        V.append(vn.denormalize(v_n.float()))
        SPD.append(env.sim.state[lid, 3].float())
        near = env.nearest_visible_car_v2()[lid] if hasattr(env, "nearest_visible_car_v2") else None
        NEAR.append(near[:, 0].float() if near is not None else torch.zeros(B, device=device))
        act_full[lid] = act.float()
        obs, rew, term, trunc, info = env.step(act_full.clamp(-1, 1))
        comp = info["reward_components"]
        r = rew[lid].float()
        col = comp["collision"][lid].float()                  # -w_col x hard wall/prop onsets
        car = comp["car_contact"][lid].float()                # -w_car x charged contact, every step in contact
        touching = car < 0
        onset = touching & ~touching_prev
        done = (term | trunc)[lid]
        touching_prev = touching & ~done
        r = r - col - car + PRICE * col / max(w_col, 1e-9) - PRICE * onset.float()
        R.append(r); D.append(done)
        h = reset_hidden(hn.float(), done)
    V, R, D = torch.stack(V).cpu().numpy(), torch.stack(R).cpu().numpy(), torch.stack(D).cpu().numpy()
    SPD, NEAR = torch.stack(SPD).cpu().numpy(), torch.stack(NEAR).cpu().numpy()
    T = V.shape[0]
    G = np.zeros_like(R); ok = np.zeros_like(R, dtype=bool)
    acc = np.zeros(R.shape[1]); steps_left = np.zeros(R.shape[1])
    for t in reversed(range(T)):                          # G over the episode, and how many steps of future it saw
        acc = R[t] + GAMMA * acc * (~D[t])
        steps_left = np.where(D[t], 0, steps_left + 1)
        G[t] = acc
        ok[t] = steps_left >= HORIZON
    err = V - G
    sel = ok
    print(f"\nCRITIC BIAS {os.path.basename(os.path.dirname(a.init))}/{os.path.basename(a.init)}: {int(sel.sum())} steps with "
          f"{HORIZON} steps of future | reward/step {R.mean():+.4f} | V {V[sel].mean():.2f}  G {G[sel].mean():.2f}  "
          f"V-G {err[sel].mean():+.2f} (sd {err[sel].std():.2f}, se {err[sel].std() / np.sqrt(max(sel.sum(), 1) / 400):.2f})")
    for lo, hi in ((0, 3), (3, 5), (5, 6.5), (6.5, 8), (8, 11)):
        m = sel & (SPD >= lo) & (SPD < hi)
        if m.sum() > 200:
            print(f"  speed {lo}-{hi} m/s: {m.mean() * 100:4.1f} % of steps | V {V[m].mean():7.2f} G {G[m].mean():7.2f} V-G {err[m].mean():+6.2f}")
    raise SystemExit(0)


ppo_v2.train = train
ppo.main()
