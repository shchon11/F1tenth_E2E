"""Research 2026-10-27 (second pass, fresh measurements): how much signal does one PPO update carry for "drive faster",
and how much policy-gradient noise do the plan's dead dimensions add?

Run like ppo.py with the actor frozen (--lr 0 --lr-end 0); the critic trains as usual. Per update (after SKIP):
* the surrogate gradient at ratio 1 w.r.t. a global bias on each action mean, mean(eps_j * adv_n) (eps = sampled noise in
  sigma units, adv_n = the update's normalised advantage) -- its mean over updates / spread over updates = SNR per update;
  reported for all learner steps and for straights (speed > 6 m/s);
* every DEAD_EVERY updates (env, 0 = off): the norm of the policy-gradient on the actor trunk from the LIVE action dims and
  from the DEAD ones (MAP gives v2, v3 no authority and v4, kappa_4 ~none; measured 10-23), on the first CHUNKS env chunks.
env: SKIP (default 6), DEAD_EVERY (default 4), CHUNKS (default 64), DEAD (comma list of dims, default 3,5,6,7)."""
import os
import sys

import numpy as np
import torch

from f1sim.learn import ppo, ppo_v2

SKIP = int(os.environ.get("SKIP", "6"))
DEAD_EVERY = int(os.environ.get("DEAD_EVERY", "4"))
CHUNKS = int(os.environ.get("CHUNKS", "64"))
DEAD = [int(x) for x in os.environ.get("DEAD", "3,5,6,7").split(",")]
REC, GN = [], []
_gae = ppo_v2.compute_gae
_n = [0]


def _hook(*args, **kw):
    adv = _gae(*args, **kw)
    f = sys._getframe(1).f_locals
    _n[0] += 1
    model = f["model"]
    sd = model.actor.log_std.detach().exp().float()
    eps = (f["buf_act"].float() - f["buf_ref_mu"].float()) / sd
    mask = f["buf_mask"].float()
    w = mask / mask.sum().clamp_min(1.0)
    mu_ = (adv * w).sum(); sd_ = (((adv - mu_) ** 2) * w).sum().sqrt()
    an = (adv - mu_) / (sd_ + 1e-8)
    REC.append(dict(eps=eps.cpu().numpy().astype(np.float32), an=an.float().cpu().numpy(), mask=mask.cpu().numpy(),
                    prog=f["buf_prog"].float().cpu().numpy()))
    if DEAD_EVERY > 0 and _n[0] > SKIP and _n[0] % DEAD_EVERY == 0:
        actor = model.actor
        cols = torch.arange(min(CHUNKS, f["B"]), device=adv.device)
        trunk = [p for n_, p in actor.named_parameters() if not n_.startswith(("mu.", "b8", "near_car")) and n_ != "log_std"]
        was = actor.training
        actor.train()
        with torch.enable_grad():
            seq = actor.sequence(f["buf_scan"][:, cols].float(), f["buf_ego"][:, cols], f["buf_c"][:, cols],
                                 h0=f["buf_h0"][:, cols].contiguous(), keep=f["buf_keep"][:, cols])
            mu = seq["mu"].float()
            dist = torch.distributions.Normal(mu, sd.expand_as(mu))
            lp = dist.log_prob(f["buf_act"][:, cols].float())                       # (T, m, A)
            a_n = an[:, cols]; m_ = mask[:, cols]
            out = {}
            for name, dims in (("live", [j for j in range(lp.shape[-1]) if j not in DEAD]), ("dead", DEAD),
                               ("all", list(range(lp.shape[-1])))):
                loss = -((a_n * lp[..., dims].sum(-1)) * m_).sum() / m_.sum().clamp_min(1.0)
                g = torch.autograd.grad(loss, trunk, retain_graph=True)
                out[name] = torch.cat([x.reshape(-1) for x in g])
        actor.train(was)
        gl, gd, ga = out["live"], out["dead"], out["all"]
        GN.append((float(gl.norm()), float(gd.norm()), float(ga.norm()),
                   float((gl @ ga) / (gl.norm() * ga.norm() + 1e-12))))
    return adv


ppo_v2.compute_gae = _hook
sys.argv = ["ppo"] + sys.argv[1:]
try:
    ppo.main()
finally:
    R = REC[SKIP:]
    names = ["k1", "k2", "k3", "k4", "v1", "v2", "v3", "v4"]
    dt = 0.025
    print(f"\nSNR v2 | {len(R)} updates analysed (first {SKIP} dropped)")
    for sel_name, sel in (("all", lambda v: v >= -1), ("straights v>6", lambda v: v > 6.0)):
        G, pooled = [], []
        for r in R:
            keep = (r["mask"] > 0) & sel(r["prog"] / dt)
            if keep.sum() < 50:
                continue
            prod = r["eps"][keep] * r["an"][keep][:, None]
            G.append(prod.mean(0)); pooled.append(prod)
        if len(G) < 4:
            print(f"  {sel_name}: too few"); continue
        G = np.stack(G); P_ = np.concatenate(pooled)
        print(f"  {sel_name}: {P_.shape[0]} samples ({P_.shape[0] / len(G):.0f} an update), {len(G)} updates")
        print("    dim      " + " ".join(f"{n:>8s}" for n in names))
        print("    mean     " + " ".join(f"{x:+8.4f}" for x in G.mean(0)) + "   (per-update signal, mean over updates)")
        print("    SNR/upd  " + " ".join(f"{x:+8.2f}" for x in G.mean(0) / (G.std(0, ddof=1) + 1e-12)))
        print("    t(upd)   " + " ".join(f"{x:+8.2f}" for x in G.mean(0) / (G.std(0, ddof=1) / np.sqrt(len(G)) + 1e-12))
              + "   (t over updates: robust to correlation within an update)")
    if GN:
        g = np.array(GN)
        print(f"  trunk PG gradient norms over {len(g)} updates ({CHUNKS} env chunks each): live {g[:, 0].mean():.4f}  "
              f"dead {g[:, 1].mean():.4f}  all {g[:, 2].mean():.4f} | dead/live {np.mean(g[:, 1] / g[:, 0]):.2f} | "
              f"cos(live, all) {g[:, 3].mean():.3f}")
