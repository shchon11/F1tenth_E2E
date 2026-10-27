"""Training-log objective per block of ACTOR updates, for runs in a stationary world (rl-audit 2026-10-28): J/step = progress +
steer + wrong-way - 80 x charged hard onsets, hard wall/prop and car rates, approx_kl. usage: python actor_blocks.py RUN:FIRST_ACTOR_UPDATE ...
env: BLOCK (16)"""
import json
import os
import sys

import numpy as np

BLOCK = int(os.environ.get("BLOCK", "16"))
for arg in sys.argv[1:]:
    run, first = arg.split(":")
    rows = [json.loads(l) for l in open(os.path.expanduser(f"~/f1sim_runs/{run}/progress.jsonl"))]
    rows = [r for r in rows if r["update"] > int(first)]
    print(f"== {run} (액터 업데이트 {len(rows)}회) 액터 구간 | 진행 m/s | 강한 벽+상자 | 강한 차 | J/스텝 | kl | clip%")
    for a in range(0, len(rows), BLOCK):
        b = rows[a:a + BLOCK]
        g = lambda k: np.array([x.get(k, np.nan) for x in b], float)
        prog = g("reward/progress_per_step"); w = g("lagrange/wall_rate_per_km"); c = g("lagrange/car_rate_per_km")
        J = prog + g("reward/steer_rate_per_step") + g("reward/wrong_way_per_step") - 80 * (w + c) * prog / 1000
        print(f"  {a:3d}-{a + len(b) - 1:3d} | {np.nanmean(prog) * 40:.2f} | {np.nanmean(w):.2f} | {np.nanmean(c):.2f} | {np.nanmean(J):.4f} | "
              f"{np.nanmean(g('loss/approx_kl')):.4f} | {np.nanmean(g('loss/clipfrac')) * 100:.1f}")
