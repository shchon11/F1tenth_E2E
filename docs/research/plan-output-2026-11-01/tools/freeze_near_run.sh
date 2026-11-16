#!/bin/bash
# (11-16) 상대 차가 가까울 때 기억을 멈춰 두면(갱신 안 함) 차 관련 결과가 달라지나 (지우기는 기억에 든 내 차 속도까지 날려 섞임):
# 대조 = 상대 차 없는 경주에서 같은 빈도·길이(시작 확률 0.0114, 34결정)로 멈춤 (멈추지 않은 기록 = 그 checkpoint의 exam1 기록).
# 인자: checkpoint 이름(예 kr26e3/plan_u1600) GPU
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/probe
C=$1; G=${2:-0}; N=$(echo $C | tr / _)
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SLOT=slot4070.sh; [ "$G" = 1 ] && SLOT=slot5060.sh
cd /home/shchon11/F1tenth_E2E/f1sim
for w in near:3.0 rand:0.0114:34:3.0; do
  F1SIM_EXAM_FREEZE=$w CUDA_VISIBLE_DEVICES=$G bash $T/$SLOT timeout 5400 python3 -u -W ignore $T/exam_record.py \
    /home/shchon11/f1sim_runs/$C.pt $O/freeze_${N}_${w//:/_}.npz ideal 256 uniform > $O/freeze_${N}_${w//:/_}.log 2>&1
done
