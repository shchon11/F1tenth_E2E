#!/bin/bash
# 용량 판별 (11-03, 4070): 합성 행동(빈 길 = 혼자, 상자 근처 = 상자)을 폭 256 / 768 MLP가 담나
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 -u /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/capacity_probe.py ~/f1sim_runs/srp25h2_props/state_u200.pt ~/f1sim_runs/srp25h2_solo2/state_u300.pt 1024 60 779 > $D/capacity25.log 2>&1
echo done > $D/capacity25.done
