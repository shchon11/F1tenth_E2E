"""Offline companion of grad_snr.py (2026-10-27): the same "raise the action mean" gradient SNR, with the advantages
recomputed from the saved rollout (rewards after the Lagrangian charge, values, terminations) for several GAE lambda.
Lower lambda leans on the critic (bias) and drops the noise of later events and later exploration (variance).
For a HOLD run the lambda inside a held block is 1 (the block's mid values cancel: the decision-level GAE).
usage: snr_lambda.py run.npz [gamma]"""
import sys

import numpy as np

Z = np.load(sys.argv[1]); gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.997
U, T, B = Z["adv"].shape
dt = 0.025
names = ["k1", "k2", "k3", "k4", "v1", "v2", "v3", "v4"]


def gae(u, lam):
    rew, val, done, trunc, fval, dec = Z["rew"][u], Z["val"][u], Z["done"][u], Z["trunc"][u], Z["fval"][u], Z["dec"][u]
    adv = np.zeros_like(rew); last = np.zeros(B, dtype=rew.dtype)
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]; cont = nonterm * (1.0 - trunc[t])
        nxt = np.where(trunc[t] > 0, fval[t], val[t + 1])
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        lam_t = lam if t + 1 >= T else np.where(dec[t + 1], lam, 1.0)
        last = delta + gamma * lam_t * cont * last
        adv[t] = last
    return adv


chk = np.abs(gae(0, 0.97) - Z["adv"][0]).max()
hold = "held blocks" if not Z["dec"].all() else "every step a decision"
print(f"{sys.argv[1]}: {U} updates x {T} x {B} ({hold}); lambda 0.97 recomputed vs logged max |diff| {chk:.2e}")
for sel_name, lo, hi in (("all", -1, 99), ("v>6 straights", 6, 99), ("v<4 corners", -1, 4)):
    print(f"  {sel_name}:   SNR per update (mean/sd over updates) of eps_j * adv_n, and pooled t")
    for lam in (0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.99):
        G, P_ = [], []
        for u in range(U):
            a = gae(u, lam) if lam != 0.97 or not Z["dec"].all() else Z["adv"][u]
            m = Z["mask"][u] > 0
            w = Z["mask"][u] / max(Z["mask"][u].sum(), 1.0)
            mu_ = (a * w).sum(); sd_ = np.sqrt((((a - mu_) ** 2) * w).sum())
            an = (a - mu_) / (sd_ + 1e-8)
            v = Z["prog"][u] / dt
            keep = m & Z["dec"][u] & (v > lo) & (v < hi)
            prod = Z["eps"][u][keep] * an[keep][:, None]
            G.append(prod.mean(0)); P_.append(prod)
        G = np.stack(G); P_ = np.concatenate(P_)
        snr = G.mean(0) / (G.std(0, ddof=1) + 1e-12)
        t = P_.mean(0) / (P_.std(0) / np.sqrt(len(P_)) + 1e-12)
        print(f"    lam {lam:4.2f}  SNR " + " ".join(f"{n}{x:+5.2f}" for n, x in zip(names, snr))
              + "  | t " + " ".join(f"{x:+5.1f}" for x in t))
