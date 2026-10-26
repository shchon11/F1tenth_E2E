"""Offline (2026-10-27): which reward term carries the "drive faster" signal, and which one drowns it?

GAE is linear in the rewards and the values: A = sum_c GAE(r_c, V=0) + GAE(r=0, V), with r_c each logged reward
component (`gym_env.REWARD_COMPONENT_KEYS`) and "other" = what the Lagrangian step and anything unlogged add. Per term:
its share of the advantage variance, the signal it carries for each action mean (cov(eps_j, A_c) / sd(A), the
part of the pooled eps*adv_n mean it explains), and the t of "raise v1 on straights" with that term removed (the
critic held fixed -- a first look, not a retrained critic).
usage: snr_components.py run.npz [gamma lam]"""
import sys

import numpy as np

KEYS = ("progress", "collision", "collision_speed", "steer_rate", "proximity", "plan_clearance", "wrong_way", "lap",
        "alive", "car_contact", "overtake", "lap_time", "car_proximity", "sideslip", "ttc", "overtake_hold",
        "grip_budget", "rear_end", "merge")
Z = np.load(sys.argv[1])
gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.997
lam = float(sys.argv[3]) if len(sys.argv) > 3 else 0.97
U, T, B = Z["rew"].shape
dt = 0.025


def gae(rew, val, done, trunc, fval, dec):
    adv = np.zeros_like(rew); last = np.zeros(rew.shape[1], dtype=rew.dtype)
    for t in reversed(range(rew.shape[0])):
        nonterm = 1.0 - done[t]; cont = nonterm * (1.0 - trunc[t])
        nxt = np.where(trunc[t] > 0, fval[t], val[t + 1])
        delta = rew[t] + gamma * nxt * nonterm - val[t]
        lam_t = lam if t + 1 >= rew.shape[0] else np.where(dec[t + 1], lam, 1.0)
        last = delta + gamma * lam_t * cont * last
        adv[t] = last
    return adv


parts = {}
zv, zf = np.zeros((T + 1, B), np.float32), np.zeros((T, B), np.float32)
A_tot = np.stack([gae(Z["rew"][u], Z["val"][u], Z["done"][u], Z["trunc"][u], Z["fval"][u], Z["dec"][u]) for u in range(U)])
for i, k in enumerate(KEYS):
    r = Z["comp"][..., i]
    if np.abs(r).max() == 0:
        continue
    parts[k] = np.stack([gae(r[u], zv, Z["done"][u], Z["trunc"][u], zf, Z["dec"][u]) for u in range(U)])
other = Z["rew"] - Z["comp"].sum(-1)
if np.abs(other).max() > 0:
    parts["other (Lagrangian)"] = np.stack([gae(other[u], zv, Z["done"][u], Z["trunc"][u], zf, Z["dec"][u]) for u in range(U)])
parts["critic (-V + gamma V')"] = np.stack([gae(np.zeros((T, B), np.float32), Z["val"][u], Z["done"][u], Z["trunc"][u],
                                                Z["fval"][u], Z["dec"][u]) for u in range(U)])
chk = np.abs(sum(parts.values()) - A_tot).max()
v = Z["prog"] / dt
sel = {"all": (Z["mask"] > 0) & Z["dec"], "v>6 straights": (Z["mask"] > 0) & Z["dec"] & (v > 6)}
print(f"{sys.argv[1]}: {U} updates; decomposition check max |sum - A| {chk:.2e}")
for sname, m in sel.items():
    eps = Z["eps"][m]                                         # (n, 8)
    At = A_tot[m]; sd = At.std()
    print(f"\n  {sname}: {m.sum()} samples, sd(A) {sd:.3f}")
    print(f"    {'term':24s} {'var share':>9s} {'sd':>7s}  signal for v1 (cov/sd(A))  k1      | t(v1) without the term")
    base_t = (eps[:, 4] * (At - At.mean()) / sd).mean() / ((eps[:, 4] * (At - At.mean()) / sd).std() / np.sqrt(len(At)))
    print(f"    {'(all terms)':24s} {1.0:9.3f} {sd:7.3f}  {np.cov(eps[:, 4], At)[0, 1] / sd:+.4f}                   "
          f"{np.cov(eps[:, 0], At)[0, 1] / sd:+.4f} | t(v1) {base_t:+.2f}")
    for k, Pk in parts.items():
        a = Pk[m]
        rest = At - a
        rn = (rest - rest.mean()) / (rest.std() + 1e-9)
        prod = eps[:, 4] * rn
        t = prod.mean() / (prod.std() / np.sqrt(len(prod)))
        print(f"    {k:24s} {a.var() / At.var():9.3f} {a.std():7.3f}  {np.cov(eps[:, 4], a)[0, 1] / sd:+.4f}"
              f"                   {np.cov(eps[:, 0], a)[0, 1] / sd:+.4f} | {t:+.2f}")
