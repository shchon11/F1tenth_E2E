#!/bin/bash
# (11-17) 상대 차 시험: frozen exam(상대 차는 학습 세계 그대로 0.4-1.2, 상자에 박혀 다시 놓이는 차가 많음) 옆에, 실제 대회에 가까운
# 상대 차로 고정한 두 번째 시험 -- 모든 세션에 상대 차(배율 0.9-1.1 = 우리 차의 0.7-0.82배), 상자는 대회처럼 2개, 상대 차 레이싱 라인
# 띠 밖에만(F1SIM_EXAM_CORRIDOR=1). 같은 시드라 checkpoint끼리 세션 짝 비교가 됨. 인자: checkpoint 이름(예 kr26j/plan_u1900) ...
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/probe
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_EXAM_OPP_SLOT='{"speed_scale": [0.9, 1.1], "present": 1.0}' F1SIM_EXAM_CORRIDOR=1
cd /home/shchon11/F1tenth_E2E/f1sim
for C in "$@"; do
  N=$(echo $C | tr / _ | sed 's/plan_//')
  [ -e $O/oppexam_$N.npz ] && continue
  CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 3600 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/$C.pt $O/oppexam_$N.npz ideal 256 max2 > $O/oppexam_$N.log 2>&1
done
