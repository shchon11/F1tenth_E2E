"""Research (2026-10-27): is the PPO policy gradient for "drive faster" mostly noise?

Observed: PPO from the Korea entry raises the straight-line command ~0.5 m/s per 256 updates and then stalls near 7 m/s
(the car does 9+), races and solo alike; a 5x learning rate climbs a little faster; every fine-tune drifts toward more
car contacts. Candidate cause (Tallec et al. 2019, "time discretization"): one action lasts 25 ms, so its advantage is
O(dt) while the advantage noise (critic error, the other cars, props, the noise of all later actions) is not; with
2048 samples an update the "raise the speed" direction would then be buried, and Adam's step follows the noise.

Measured here with the actor FROZEN (run it with --lr 0 --lr-end 0; the critic trains as usual): per update, the
gradient of the PPO surrogate at ratio 1 w.r.t. a global bias on each action dimension's mean, which is
mean(eps_j * adv_n) / sigma_j with eps the sampled noise in sigma units and adv_n the update's normalised advantage
(what the loss uses). Its mean over updates is the signal, its spread the noise -> SNR per update, and the pooled
per-sample correlation -> how many samples one coherent step would need.
env HOLD=k: the env executes each sampled action for k control steps (a decision every k*25 ms; the GRU still reads
every step); only decision steps are analysed. env NPZ=npz, SKIP=updates to drop at the start (critic settling)."""
import os
import sys

import numpy as np
import torch

from f1sim import gym_env
from f1sim.learn import ppo, ppo_v2

HOLD = int(os.environ.get("HOLD", "1"))
REC, DEC = [], []
_gae = ppo_v2.compute_gae


def _gae_hook(*args, **kw):
    adv = _gae(*args, **kw)
    f = sys._getframe(1).f_locals
    T = adv.shape[0]
    lid = f["lid"]
    sd = f["model"].actor.log_std.detach().exp().float()
    eps = (f["buf_act"].float() - f["buf_ref_mu"].float()) / sd
    if "buf_dec" in f and int(getattr(f.get("a"), "decision_every", 1) or 1) > 1:
        dec = f["buf_dec"]                         # --decision-every: the env holds the plan on the track
    elif HOLD > 1:
        dec = torch.stack(DEC[-T:])[:, lid]        # the old body-frame hold (env patch below)
    else:
        dec = torch.ones_like(adv, dtype=torch.bool)
    rew, val, done, trunc, fval = args[:5]
    REC.append(dict(eps=eps.cpu().numpy().astype(np.float32), adv=adv.float().cpu().numpy(),
                    mask=f["buf_mask"].float().cpu().numpy(), prog=f["buf_prog"].float().cpu().numpy(),
                    dec=dec.cpu().numpy(), rew=rew.float().cpu().numpy(), val=val.float().cpu().numpy(),
                    done=done.float().cpu().numpy(), trunc=trunc.float().cpu().numpy(), fval=fval.float().cpu().numpy(),
                    comp=f["buf_comp"].float().cpu().numpy(), raw=f["buf_rew"].float().cpu().numpy()))
    return adv


ppo_v2.compute_gae = _gae_hook

if HOLD > 1:
    _step = gym_env.F1VecEnv.step
    ST = {}

    def _step_hold(self, action):
        if ST.get("held") is None or ST["held"].shape != action.shape:
            ST["held"] = action.clone()
            ST["cnt"] = torch.zeros(action.shape[0], dtype=torch.long, device=action.device)
        dec = (ST["cnt"] % HOLD) == 0
        ST["held"] = torch.where(dec[:, None], action, ST["held"])
        out = _step(self, ST["held"])
        done = out[2] | out[3]
        ST["cnt"] = torch.where(done, torch.zeros_like(ST["cnt"]), ST["cnt"] + 1)
        DEC.append(dec.clone())
        del DEC[:-4096]
        return out

    gym_env.F1VecEnv.step = _step_hold

sys.argv = ["ppo"] + sys.argv[1:]
ppo.main()

skip = int(os.environ.get("SKIP", "8"))
R = REC[skip:]
if os.environ.get("NPZ"):
    np.savez_compressed(os.environ["NPZ"], **{k: np.stack([r[k] for r in R]) for k in R[0]})
names = ["k1", "k2", "k3", "k4", "v1", "v2", "v3", "v4"]
dt = 0.025
print(f"\nHOLD {HOLD} | {len(R)} updates analysed (first {skip} dropped)")
for sel_name, sel in (("all", lambda v: v >= -1), ("v>6 (straights)", lambda v: v > 6.0), ("v<4 (corners)", lambda v: v < 4.0)):
    G, pooled = [], []
    for r in R:
        m = r["mask"] > 0
        w = r["mask"] / max(r["mask"].sum(), 1.0)
        mu_ = (r["adv"] * w).sum(); sd_ = np.sqrt((((r["adv"] - mu_) ** 2) * w).sum())
        an = (r["adv"] - mu_) / (sd_ + 1e-8)
        keep = m & r["dec"] & sel(r["prog"] / dt)
        if keep.sum() < 50:
            continue
        prod = r["eps"][keep] * an[keep][:, None]                 # (n, 8)
        G.append(prod.mean(0))
        pooled.append(prod)
    if len(G) < 4:
        print(f"  {sel_name}: too few samples"); continue
    G = np.stack(G); P_ = np.concatenate(pooled)
    n_per = P_.shape[0] / len(G)
    snr_u = G.mean(0) / (G.std(0, ddof=1) + 1e-12)
    corr = P_.mean(0) / (P_.std(0) + 1e-12)
    se = P_.std(0) / np.sqrt(P_.shape[0])
    print(f"  {sel_name}: {P_.shape[0]} samples ({n_per:.0f} an update)")
    print("    dim      " + " ".join(f"{n:>8s}" for n in names))
    print("    signal   " + " ".join(f"{x:+8.4f}" for x in P_.mean(0)) + "   (pooled mean of eps*adv_n)")
    print("    t-stat   " + " ".join(f"{x:+8.1f}" for x in P_.mean(0) / (se + 1e-12)))
    print("    SNR/upd  " + " ".join(f"{x:+8.2f}" for x in snr_u))
    print("    N* (SNR1)" + " ".join(f"{(1 / max(c * c, 1e-12)):8.0f}" for c in corr) + "   samples for one coherent step")
