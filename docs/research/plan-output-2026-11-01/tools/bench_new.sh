#!/bin/bash
# 새 30배치(라인 위 상자 30 % 들어간 콘솔 학습 장애물, 11-03 21:15~, 시드 777)로 다시 재기. 인자: 이름 체크포인트 GPU 맵
N=$1; C=$2; GPU=$3; MC=${4:-real/iccas25}
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $C $MC 40 30 777 $D/layouts/n_$N 2>&1 | grep "배치\|합계\|Error" > $D/layouts_n_$N.log
echo done > $D/layouts_n_$N.done
