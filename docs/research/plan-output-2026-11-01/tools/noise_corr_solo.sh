#!/bin/bash
# 상관된 탐색 잡음이 넓은 선을 벽에 박게 하나 (11-03): 혼자 정책(넓은 선)을 깨끗한 맵에서 잡음 상관 0 / 0.9로 (5060, 512대 x 40 s)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
for c in 0.0 0.9; do
  CUDA_VISIBLE_DEVICES=1 timeout 1800 python3 -u /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/noise_margin.py ~/f1sim_runs/srp25h2_solo2/state_u300.pt 512 40 781 0.0 all $c 2>&1 | grep -v "Warn\|warn"
done > $D/noise_corr_solo25.log
echo done > $D/noise_corr_solo25.done
