#!/bin/bash
# 2026 srp26gwt 벽 세게 7(배치 10: 오른쪽 직선 튀어나온 벽을 매 바퀴 6.7 m/s로 긁음)이 어느 단계에서 생겼나: 중간 체크포인트 30배치. 인자: 이름 GPU
N=$1; GPU=$2; MC=real/korea26
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py /home/shchon11/f1sim_runs/$N/state_u100.pt $MC 40 30 777 $D/layouts/t_$N 2>&1 | grep "배치\|합계\|Error" > $D/layouts_t_$N.log
echo done > $D/layouts_t_$N.done
