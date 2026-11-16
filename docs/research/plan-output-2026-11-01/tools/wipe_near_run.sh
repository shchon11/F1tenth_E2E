#!/bin/bash
# (11-16) 상대 차가 가까울 때 기억을 지우면 차 관련 결과가 나빠지나: 같은 frozen exam을 '3 m 안에 차가 있을 때 지움'과
# 대조 '3 m 안에 차가 없을 때 결정의 5 %를 무작위로 지움'으로 다시 돌림 (지우지 않은 기록 = 그 checkpoint의 exam1 기록).
# 인자: checkpoint 이름(예 kr26e3/plan_u1600) GPU
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/probe
C=$1; G=${2:-0}; N=$(echo $C | tr / _)
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SLOT=slot4070.sh; [ "$G" = 1 ] && SLOT=slot5060.sh
cd /home/shchon11/F1tenth_E2E/f1sim
for w in near:3.0 rand:0.05:3.0; do
  F1SIM_EXAM_WIPE=$w CUDA_VISIBLE_DEVICES=$G bash $T/$SLOT timeout 5400 python3 -u -W ignore $T/exam_record.py \
    /home/shchon11/f1sim_runs/$C.pt $O/wipe_${N}_${w//:/_}.npz ideal 256 uniform > $O/wipe_${N}_${w//:/_}.log 2>&1
done
