#!/bin/bash
# (11-16) 상대 차가 속도 배율을 올리면 실제로 얼마나 빨리 달리나(우리 차의 70-110 %를 낼 수 있나): 모든 세션에 상대 차, 배율 고정.
# 인자: 배율 [배율 ...]   출력 out/probe/oppscale_<배율>.npz
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/probe
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
for s in "$@"; do
  F1SIM_EXAM_OPP_SLOT="{\"speed_scale\": [$s, $s], \"present\": 1.0}" CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 3600 python3 -u -W ignore \
    $T/exam_record.py /home/shchon11/f1sim_runs/kr26i/plan_u1200.pt $O/oppscale_$s.npz ideal 128 uniform 60 30 > $O/oppscale_$s.log 2>&1
done
