#!/bin/bash
# (11-14 21:50) 질문: critic이 좁게 지나는 통과(틈 < 10 cm)와 넓게 지나는 통과(> 20 cm)를 지나기 0.3~1.5 s 전에 값으로 구분하나.
#   구분하면 -> "닿을 뻔한 통과의 예상 접촉 값"(후보 가)은 이미 학습 신호에 있음 -> 버림. 구분 못 하면 -> 그 정보가 학습에 없음.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; O=$R/out/edge1114
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
F1SIM_EXAM_SAVE_VALUE=1 CUDA_VISIBLE_DEVICES=0 timeout 2400 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/plan26l/plan_u150.pt $O/val_plan26l_u150.npz ideal 256 uniform > $O/val_plan26l_u150.log 2>&1
echo "rc $?" >> $O/val_plan26l_u150.log
touch $O/val26l.done
