#!/bin/bash
# 빈 길에서만 속도 올림 (11-03): 잡음 목표 탓인지 학습 실패(용량·신호) 탓인지 판별 (5060, 512대 x 40 s)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=1 timeout 5400 python3 -u /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/noise_margin.py ~/f1sim_runs/srp25h2_props/state_u200.pt 512 40 778 0.0,0.15,0.3 clear > $D/noise_margin25_clear.log 2>&1
echo done > $D/noise_margin25_clear.done
