"""Training-log objective against how far the policy has moved (rl-audit 2026-10-28, rlc8 vs rlc9): per bin of kl_ref (the
KL from the run's starting policy) and per block of actor updates -- progress m/s, hard wall/prop and car per km, J/step
(progress + steer + wrong-way - 80 x charged hard onsets), approx_kl. usage: python kl_bins.py RUN:FIRST_ACTOR_UPDATE ...
env: BLOCK (64), EDGES (kl_ref bin edges, default 0,0.25,0.5,1,1.5,2,3,5,8)"""
import json
import os
import sys

import numpy as np

BLOCK = int(os.environ.get("BLOCK", "64"))
EDGES = [float(x) for x in os.environ.get("EDGES", "0,0.25,0.5,1,1.5,2,3,5,8").split(",")]


def table(rows):
    g = lambda k: np.array([x.get(k, np.nan) for x in rows], float)
    prog = g("reward/progress_per_step"); w = g("lagrange/wall_rate_per_km"); c = g("lagrange/car_rate_per_km")
    J = prog + g("reward/steer_rate_per_step") + g("reward/wrong_way_per_step") - 80 * (w + c) * prog / 1000
    return prog * 40, w, c, J, g("loss/approx_kl"), g("kl_ref")


for arg in sys.argv[1:]:
    run, first = arg.split(":")
    rows = [json.loads(l) for l in open(os.path.expanduser(f"~/f1sim_runs/{run}/progress.jsonl"))]
    rows = [r for r in rows if r["update"] > int(first)]
    v, w, c, J, kl, ref = table(rows)
    print(f"== {run}: {len(rows)} actor updates")
    print("  actor block | progress m/s | hard wall+prop | hard car | J/step | approx_kl | kl_ref at the end")
    for a in range(0, len(rows), BLOCK):
        s = slice(a, a + BLOCK)
        print(f"  {a:4d}-{min(a + BLOCK, len(rows)) - 1:4d} | {np.nanmean(v[s]):.2f} | {np.nanmean(w[s]):.2f} | {np.nanmean(c[s]):.2f} | "
              f"{np.nanmean(J[s]):.4f} | {np.nanmean(kl[s]):.4f} | {ref[s][-1]:.2f}")
    print("  kl_ref bin | updates | progress m/s | hard wall+prop | hard car | J/step")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (ref >= lo) & (ref < hi)
        if m.sum():
            print(f"  {lo:4.2f}-{hi:4.2f} | {m.sum():4d} | {np.nanmean(v[m]):.2f} | {np.nanmean(w[m]):.2f} | {np.nanmean(c[m]):.2f} | "
                  f"{np.nanmean(J[m]):.4f}")
