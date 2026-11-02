#!/bin/bash
# 지도 대 실제 여유 (11-03): Korea 2025 선생, 30배치 x 40 s, 시드 777 (5060)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=1 timeout 3600 python3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/grid_vs_true.py ~/f1sim_runs/pick_srp25h2/state_latest.pt real/iccas25 40 30 777 2>&1 | grep -v "Warn\|warn" > $D/grid_vs_true25.log
echo done > $D/grid_vs_true25.done
