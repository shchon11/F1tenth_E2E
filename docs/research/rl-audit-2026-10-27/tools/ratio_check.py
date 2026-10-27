"""Does the PPO ratio start at 1? Rollout log-probs (step-by-step, as the trainer computes them) vs the update's
recomputation (actor.sequence over the chunk) with the SAME weights. Also: which action dims carry the logp.

usage: python ratio_check.py WARM T -- <ppo argv>
"""
import sys

import torch

WARM, T = int(sys.argv[1]), int(sys.argv[2])
argv = sys.argv[sys.argv.index("--") + 1:]
sys.argv = ["ppo"] + argv
from f1sim.learn import ppo, ppo_v2  # noqa: E402
from f1sim.learn import conditioning as cond_mod  # noqa: E402
from f1sim.learn.model import load_checkpoint  # noqa: E402
from f1sim.learn.memory import reset_hidden  # noqa: E402


def train(a, *, env, obs, device, names, init_ck, controller, graph_rt, applied, league_seats=()):
    model, extra = load_checkpoint(a.init, device)
    actor = model.actor
    lid = env.learner_ids
    held = getattr(a, "dial_constant", None)
    dial = cond_mod.ConstantDial(env, held) if held is not None else cond_mod.DialDraw(env, a.dial_margin, a.dial_exact)
    cspec = cond_mod.CondSpec.from_meta(model.meta["cond"])
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp)
    h = actor.initial_hidden(env.B, device=device)
    env.sim.warmup()
    model.eval()
    B = lid.numel()
    for rep in range(3):
        bufs = {k: [] for k in ("scan", "ego", "c", "act", "logp", "keep", "mu")}
        for t in range(WARM + T):
            if t == WARM:
                h0 = h.detach()[:, lid].clone()
            c = cond_mod.mu_to_c(dial.value(), cspec).to(obs["ego"].dtype)
            with torch.no_grad(), ac:
                d, hn = actor.step_dist(obs["scan"], obs["ego"], c, h)
                act = d.sample()
                logp = d.log_prob(act).sum(1)
            if t >= WARM:
                bufs["scan"].append(obs["scan"][lid].half()); bufs["ego"].append(obs["ego"][lid]); bufs["c"].append(c[lid])
                bufs["act"].append(act[lid].float()); bufs["logp"].append(logp[lid].float()); bufs["mu"].append(d.mean[lid].float())
            obs, rew, term, trunc, info = env.step(act.clamp(-1, 1))
            done = term | trunc
            h = reset_hidden(hn.float(), done)
            if t >= WARM:
                bufs["keep"].append((~done[lid]).float())
        st = {k: torch.stack(v, 0) for k, v in bufs.items()}
        keep = torch.ones(T, B, device=device)
        keep[1:] = st["keep"][:-1]
        std = actor.log_std.exp().float()
        for label, use_ac in (("bf16 autocast (as the trainer)", True), ("fp32", False)):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
                seq = actor.sequence(st["scan"].float(), st["ego"], st["c"], h0=h0.contiguous(), keep=keep)
            mu = seq["mu"].float()
            dist = torch.distributions.Normal(mu, std.expand_as(mu))
            lp = dist.log_prob(st["act"]).sum(-1)
            dlp = lp - st["logp"]
            r = dlp.exp()
            dmu = (mu - st["mu"]) / std                      # mean shift in units of the noise std, per dim
            print(f"[rep {rep}] {label}: logp_new - logp_old  mean {dlp.mean():+.4f} sd {dlp.std():.4f} "
                  f"max|.| {dlp.abs().max():.3f} | ratio outside [0.8,1.2]: {((r < 0.8) | (r > 1.2)).float().mean()*100:.2f} % "
                  f"| |ratio-1| mean {(r - 1).abs().mean():.4f} | per-dim |dmu|/std mean "
                  + " ".join(f"{x:.3f}" for x in dmu.abs().mean((0, 1)).tolist()), flush=True)
        # also: the rollout's own mean in fp32 vs bf16 for the same step (single step, no recurrence difference)
    raise SystemExit(0)


ppo_v2.train = train
ppo.main()
