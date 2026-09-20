"""Freeze today's PPO minibatch loss as an oracle for the `--memory off` byte-identity test.

Run BEFORE the memory work lands (and never again): it records what `model.evaluate_aux` plus
the loss arithmetic inlined in `ppo.main` produce for one fixed batch and one fixed seed, so the
recurrent refactor can be held to producing the identical numbers when memory is off.

    python -m f1sim.scripts.make_ppo_loss_oracle        # writes tests/data/ppo_loss_oracle.json

The arithmetic below is a verbatim copy of the inline update in `ppo.main` as of ae1f4df; the
test compares the refactored `ppo.ppo_loss_terms` against it, bit for bit.
"""
from __future__ import annotations

import copy
import json
import os

import torch

# The oracle batch. Small enough to run on one CPU thread in a second, wide enough to exercise the
# resnet stem with deltas, the aux heads, the mask and the KL reference.
CFG = dict(n_stack=4, n_beams=256, proprio_dim=32, priv_dim=21, act_dim=8,
           scan_deltas=True, temporal_encoder="cnn", scan_stem="resnet")
BATCH = 24
SEED = 20260913
HYPER = dict(clip=0.2, vf=0.5, ent=0.01, kl_coef=0.3, aux_grip=0.5, aux_opp=0.25,
             priv_mu_index=16, aux_opp_range_m=6.0, m_gt_1=True)


def make_batch(device="cpu"):
    from f1sim.learn.model import ActorCritic
    torch.manual_seed(SEED)
    model = ActorCritic(**CFG).to(device)
    g = torch.Generator(device="cpu").manual_seed(SEED + 1)
    r = lambda *s: torch.rand(*s, generator=g)
    b = BATCH
    batch = dict(
        scan=r(b, CFG["n_stack"], CFG["n_beams"]),
        pro=r(b, CFG["proprio_dim"]) * 2 - 1,
        priv=r(b, CFG["priv_dim"]) * 2 - 1,
        act=(r(b, CFG["act_dim"]) * 2 - 1),
        logp=r(b) * 2 - 1,
        adv=(r(b) * 2 - 1),
        ret=(r(b) * 2 - 1),
        val=(r(b) * 2 - 1),
        mask=(r(b) > 0.25).float(),
    )
    # The stored log probability is the policy's own, jittered: a ratio of exp(+-0.3) puts samples
    # on both sides of the clip so the oracle pins `torch.min`'s two branches, not just one.
    with torch.no_grad():
        d = model.actor.dist(batch["scan"], batch["pro"], None)
        batch["logp"] = (d.log_prob(batch["act"]).sum(1) + (r(b) * 0.6 - 0.3)).float()
    return model, batch


def terms(model, ref, batch, device="cpu"):
    """Verbatim from `ppo.main`'s inner loop (no autocast, no optimiser)."""
    from f1sim.gym_env import PRIV_OPP_DIST_SCALE
    h = HYPER
    scan, pro = batch["scan"], batch["pro"]
    priv, act = batch["priv"], batch["act"]
    # `*_`, not `_h`: `evaluate_aux` grew a future-head slot after this recording was made.
    # The model here carries no such head, so the slot is None and the arithmetic below is
    # still the verbatim copy of `ppo.main` as of ae1f4df that the oracle was recorded from.
    logp, ent, val, d, grip, opp_pred, *_ = model.evaluate_aux(scan, pro, priv, act, None)
    logp, ent, val = logp.float(), ent.float(), val.float()
    w = batch["mask"]

    def wmean(x, w):
        return (x * w).sum() / w.sum().clamp_min(1.0)

    aux = torch.zeros((), device=device)
    if h["aux_grip"] > 0:
        mu_true = priv[:, h["priv_mu_index"]] - 1.0
        aux = wmean((grip - mu_true) ** 2, w)
    aux_o = torch.zeros((), device=device)
    if h["aux_opp"] > 0 and h["m_gt_1"]:
        o_true = priv[:, 8:11] / torch.tensor([3.0, 1.0, 2.0], device=device)
        near = (priv[:, 11] * PRIV_OPP_DIST_SCALE < h["aux_opp_range_m"]).to(w.dtype) * w
        aux_o = wmean(((opp_pred - o_true) ** 2).mean(1), near)
    ratio = (logp - batch["logp"]).exp()
    adv = batch["adv"]
    pg = -wmean(torch.min(ratio * adv, ratio.clamp(1 - h["clip"], 1 + h["clip"]) * adv), w)
    v_clipped = batch["val"] + (val - batch["val"]).clamp(-h["clip"], h["clip"])
    vf = 0.5 * wmean(torch.max((val - batch["ret"]) ** 2, (v_clipped - batch["ret"]) ** 2), w)
    with torch.no_grad():
        d_ref = ref.dist(scan, pro, None)
    d_ref = torch.distributions.Normal(d_ref.mean.float(), d_ref.stddev.float())
    d = torch.distributions.Normal(d.mean.float(), d.stddev.float())
    kl_ref = wmean(torch.distributions.kl_divergence(d_ref, d).sum(1), w)
    loss = (h["vf"] * vf + (pg - h["ent"] * wmean(ent, w) + h["kl_coef"] * kl_ref)
            + h["aux_grip"] * aux + h["aux_opp"] * aux_o)
    return {"pg": pg, "vf": vf, "ent": wmean(ent, w), "kl_ref": kl_ref,
            "aux_grip": aux, "aux_opp": aux_o, "loss": loss,
            "approx_kl": ((ratio - 1) - (logp - batch["logp"])).mean(),
            "clipfrac": ((ratio - 1).abs() > h["clip"]).float().mean()}


def main():
    torch.set_num_threads(1)
    model, batch = make_batch()
    ref = copy.deepcopy(model.actor).eval()
    # Nudged off the trained actor: an untouched deep copy gives KL exactly 0, which pins nothing.
    with torch.no_grad():
        ref.mu.weight.mul_(1.03); ref.log_std.add_(0.05)
    for p in ref.parameters():
        p.requires_grad_(False)
    out = terms(model, ref, batch)
    loss = out["loss"]
    model.zero_grad()
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    rec = {"config": CFG, "batch": BATCH, "seed": SEED, "hyper": HYPER,
           "torch": torch.__version__,
           "values": {k: float(v.detach()) for k, v in out.items()},
           "grad_norm": float(gnorm)}
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tests", "data", "ppo_loss_oracle.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec["values"], indent=1))
    print("grad_norm", rec["grad_norm"])
    print("wrote", path)


if __name__ == "__main__":
    main()
