#!/bin/bash
# 학습 없이 그대로 상대 차와(시작점 기준): 인자 체크포인트 이름 GPU
C=$1; NAME=$2; GPU=$3; MC=real/iccas25
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/races_${NAME}_u0.log
: > $L
for x in "forzaeth 0.6" "forzaeth 1.0" "forzaeth_pred 0.8" "lane_switch 0.8"; do
  set -- $x
  CUDA_VISIBLE_DEVICES=$GPU timeout 1800 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/${NAME}_u0_$1_x$2 2>&1 | grep -v Warning | grep "합계" | sed "s/^/u0 $1 x$2 /" >> $L
done
echo done > $D/races_${NAME}_u0.done
