#!/bin/bash
# 닿는 장면 분류 한 번 돌리기: checkpoint 출력폴더 환경수 초 [GPU, 기본 0 = 4070]
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
OUT=$(realpath -m "$2")                                   # 아래에서 폴더를 옮기므로 출력 폴더는 절대 경로로
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=${5:-0} timeout 2400 python3 -u /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/hit_scenes.py "$1" "$OUT" "$3" "$4"
