#!/bin/bash
# 학습 세계 도구 하나 돌리기: GPU(0 = 4070, 1 = 5060) 도구.py 인자...  (출력 경로는 절대 경로로 줄 것: 아래에서 폴더를 옮김)
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
GPU=$1; TOOL=$2; shift 2
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 -u /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/$TOOL "$@"
