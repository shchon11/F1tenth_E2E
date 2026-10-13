"""`--vf-clip`: PPO's value clip is in return units, and the returns are not normalised.

With contact multipliers of 300-1000 the returns spread with sd ~125 while the clip (0.2, the policy
clip reused) lets the value move 0.2 per update: once the value has moved that far from the rollout's
estimate, a target further out in the same direction gives the critic no gradient at all.
"""
import copy

import torch

from f1sim.learn.model import ActorCritic
from f1sim.learn.ppo import PPOHyper, minibatch_losses


def _batch(b=16):
    torch.manual_seed(0)
    cfg = dict(n_stack=1, n_beams=32, proprio_dim=4, priv_dim=2, act_dim=2, scan_stem="plain")
    model = ActorCritic(**cfg)
    ref = copy.deepcopy(model.actor).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.rand(*s, generator=g)
    scan, pro, priv = r(b, 1, 32), r(b, 4) * 2 - 1, r(b, 2) * 2 - 1
    act = r(b, 2) * 2 - 1
    with torch.no_grad():
        d = model.actor.step_dist(scan, pro, None)[0]
        logp = d.log_prob(act).sum(1)
        val = model.critic.step(scan, pro, priv, None)[0].float()
    return model, ref, dict(scan=scan, pro=pro, priv=priv, act=act, logp=logp, val=val)


def _critic_grad(vf_clip):
    model, ref, x = _batch()
    # the value has already moved 1.0 below the rollout's estimate, and the target is 10 further down
    val_old = x["val"] + 1.0
    ret = x["val"] - 10.0
    out = minibatch_losses(model, ref, scan=x["scan"], pro=x["pro"], priv=x["priv"], act=x["act"],
                           logp_old=x["logp"], adv=torch.zeros_like(ret), ret=ret, val_old=val_old,
                           w=torch.ones_like(ret),
                           hyper=PPOHyper(clip=0.2, vf=0.5, ent=0.0, kl_coef=0.0, vf_clip=vf_clip))
    model.zero_grad()
    (out["vf"]).backward()
    return sum(float(p.grad.norm()) for n, p in model.named_parameters()
               if n.startswith("critic.") and p.grad is not None)


def test_the_default_clip_starves_the_critic_once_it_has_moved() -> None:
    assert _critic_grad(None) == 0.0


def test_vf_clip_zero_lets_the_critic_follow_the_target() -> None:
    assert _critic_grad(0.0) > 1e-3
