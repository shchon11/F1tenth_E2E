"""twin_dpg.py's log per block of updates (rl-audit 2026-10-28): the master drives the deployed (mean-action) policy in the
fixed training world, so its numbers are the deterministic objective's trend. usage: python twin_blocks.py RUN ... env: BLOCK (20)"""
import json
import os
import sys

import numpy as np

BLOCK = int(os.environ.get("BLOCK", "20"))
for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(os.path.expanduser(f"~/f1sim_runs/{run}/progress.jsonl"))]
    print(f"== {run}: 업데이트 {len(rows)}회 | 구간 | 기준 세계 진행 m/s | 충돌/km | J/스텝 | 목표 차 표준편차 | 충돌 갈린 행 | "
          f"평균 이동 κ0 κ1 κ2 v1 (x1e-3) | 측정 방향 이동 | 초/업데이트")
    for a in range(0, len(rows), BLOCK):
        b = rows[a:a + BLOCK]
        g = lambda k: np.array([x.get(k, np.nan) for x in b], float)
        mv = " ".join(f"{np.nanmean(g('move_' + d)) * 1e3:.2f}" for d in ("k0", "k1", "k2", "v1"))
        print(f"  {a:4d}-{a + len(b) - 1:4d} | {np.nanmean(g('master_prog_ms')):.2f} | {np.nanmean(g('master_hits_per_km')):.2f} | "
              f"{np.nanmean(g('master_J_per_step')):.4f} | {np.nanmean(g('dJ_sd')):.2f} | {np.nanmean(g('hit_flip_frac')) * 100:.1f} % | "
              f"{mv} | {np.nanmean(g('along')):+.2e} | {np.nanmean(g('time_branch_s') + g('time_update_s')):.1f}")
