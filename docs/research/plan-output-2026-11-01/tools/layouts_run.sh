#!/bin/bash
# 배치 여러 개 (5060): 인자 이름 체크포인트 맵 초 배치수 시드
NAME=$1; C=$2; MAP=$3; SECS=$4; N=$5; S=$6
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=1 timeout 3600 python3 $T/console_layouts.py $C $MAP $SECS $N $S $D/layouts/${NAME} 2>&1 | grep "배치\|합계\|Error\|error"
echo done > $D/layouts_${NAME}.done
