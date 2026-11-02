#!/bin/bash
# 상자 정책 느림 원인 (11-03): 학습 세계 1024대 x 40 s, 속도 오프셋 d, 잡음 없음/있음, 빈 길/상자 근처 나눔 (4070)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=0 timeout 5400 python3 -u /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/noise_margin.py ~/f1sim_runs/srp25h2_props/state_u200.pt 1024 40 777 -0.1,0.0,0.1,0.2,0.3 2>&1 | grep -v "Warn\|warn" > $D/noise_margin25.log
echo done > $D/noise_margin25.done
