"""The objective PPO optimises, per learner step, from a run's progress.jsonl (rl-audit-2026-10-27): progress (1 per metre) +
steer + wrong-way - PRICE x charged hard onsets (`--lagrange` wall/car rates per km x metres per step). Skips the first SKIP
updates. usage: python fixed_world_j.py NAME [NAME ...]   env: PRICE (80), SKIP (2)"""
import json
import os
import sys

import numpy as np

PRICE = float(os.environ.get("PRICE", "80")); SKIP = int(os.environ.get("SKIP", "2"))
print("이름 | 업데이트 | 진행 m/s | 강한 벽+상자/km | 강한 차(책임)/km | 목표 J/스텝 (±표준오차)")
for name in sys.argv[1:]:
    p = os.path.expanduser(f"~/f1sim_runs/{name}/progress.jsonl")
    if not os.path.exists(p):
        print(name, "없음"); continue
    rows = [json.loads(l) for l in open(p)][SKIP:]
    g = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    prog = g("reward/progress_per_step"); w = g("lagrange/wall_rate_per_km"); c = g("lagrange/car_rate_per_km")
    J = prog + g("reward/steer_rate_per_step") + g("reward/wrong_way_per_step") - PRICE * (w + c) * prog / 1000
    print(f"{name} | {len(rows)} | {np.nanmean(prog) * 40:.2f} | {np.nanmean(w):.2f} | {np.nanmean(c):.2f} | "
          f"{np.nanmean(J):.4f} ± {np.nanstd(J) / np.sqrt(max(len(rows), 1)):.4f}")
