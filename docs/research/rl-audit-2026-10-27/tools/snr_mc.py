"""Which advantage turns PPO's push the wrong way? (rl-audit 2026-10-28, after the twins: on straights kappa0 + is better
by 0.22 +- 0.06 J per 0.5 s, PPO's frozen gradient pushes kappa0 - with t -2.9.)

Run like ppo.py with the actor frozen (--lr 0 --lr-end 0) and a long rollout (--horizon >= 2 x W). Every update, the same
samples are scored with three advantages and the surrogate gradient along a global bias on each action mean is recorded
(mean of eps_j x the normalised advantage, as snr_v2.py):
  gae     what the run computes (its --horizon, --gamma, --lam, the critic everywhere);
  gae64   the same GAE cut into 64-step pieces bootstrapped from the critic -- the estimator rlc6..rlc9 trained with;
  mc      the reward actually collected over the next W steps (gamma^k, stopped at an episode's end), NO critic -- the
          twins' own quantity (their H = 120 steps).
Only steps with W steps of rollout ahead are scored. If mc agrees with the twins and gae64 does not, the critic's values
inside the advantage turn the push. env: SKIP (6), W (120)"""
import os
import sys

import numpy as np
import torch

from f1sim.learn import ppo, ppo_v2

SKIP = int(os.environ.get("SKIP", "6"))
W = int(os.environ.get("W", "120"))
REC = {"gae": [], "gae64": [], "mc": []}
_gae = ppo_v2.compute_gae


def window_return(rew, done, w, gamma):
    T = rew.shape[0]
    G = torch.zeros_like(rew); alive = torch.ones_like(rew)
    for k in range(w):
        n = T - k
        G[:n] += alive[:n] * (gamma ** k) * rew[k:k + n]
        alive[:n] = alive[:n] * (1.0 - done[k:k + n])
    return G


def _hook(*args, **kw):
    adv = _gae(*args, **kw)
    rew, val, term, trunc, final, gamma, lam = args[:7]
    f = sys._getframe(1).f_locals
    T = rew.shape[0]
    if T < 2 * W:
        raise SystemExit(f"--horizon {T} is too short for W {W}")
    model = f["model"]
    sd = model.actor.log_std.detach().exp().float()
    eps = ((f["buf_act"].float() - f["buf_ref_mu"].float()) / sd)[: T - W]
    mask = f["buf_mask"].float()[: T - W]
    prog = f["buf_prog"].float()[: T - W]
    pieces = [_gae(rew[s:s + 64], val[s:s + 65], term[s:s + 64], trunc[s:s + 64], final[s:s + 64], gamma, lam)
              for s in range(0, T, 64)]
    adv64 = torch.cat(pieces, 0)
    done = ((term.bool() | trunc.bool())).float()
    mc = window_return(rew.float(), done, W, float(gamma))
    for name, A in (("gae", adv), ("gae64", adv64), ("mc", mc)):
        A = A[: T - W].float()
        w = mask / mask.sum().clamp_min(1.0)
        m_ = (A * w).sum(); s_ = (((A - m_) ** 2) * w).sum().sqrt()
        an = (A - m_) / (s_ + 1e-8)
        REC[name].append(dict(eps=eps.cpu().numpy().astype(np.float32), an=an.cpu().numpy().astype(np.float32),
                              mask=mask.cpu().numpy(), prog=prog.cpu().numpy()))
    return adv


ppo_v2.compute_gae = _hook
sys.argv = ["ppo"] + sys.argv[1:]
try:
    ppo.main()
finally:
    names = ["k0", "k1", "k2", "k3", "v1", "v2", "v3", "v4"]
    live = [0, 1, 2, 4]
    dt = 0.025
    print(f"\nwhich advantage | W {W} steps scored per update")
    for est in ("gae", "gae64", "mc"):
        R = REC[est][SKIP:]
        for sel_name, sel in (("all", lambda v: v >= -1), ("straights v>6", lambda v: v > 6.0)):
            G = []
            for r in R:
                keep = (r["mask"] > 0) & sel(r["prog"] / dt)
                if keep.sum() < 50:
                    continue
                G.append((r["eps"][keep] * r["an"][keep][:, None]).mean(0))
            if len(G) < 4:
                print(f"  {est} {sel_name}: too few"); continue
            G = np.stack(G)
            t = G.mean(0) / (G.std(0, ddof=1) / np.sqrt(len(G)) + 1e-12)
            print(f"  {est:5s} {sel_name:13s} ({len(G)} upd) t: " + " ".join(f"{names[j]} {t[j]:+6.2f}" for j in live))
