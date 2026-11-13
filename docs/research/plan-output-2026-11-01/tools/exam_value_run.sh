#!/bin/bash
# plan25y u125의 시험 기록을 critic의 입력·값까지 남기며 다시 만듦 (ideal 실행, 상자 uniform, 256 세션 x 90 s, 4070).
# 질문: 세게 닿는 상자를 critic(상자 위치를 직접 받음)은 처음 보인 때부터 값에 치는가, actor처럼 0.5 s 앞에서야 치는가.
cd /home/shchon11/F1tenth_E2E/f1sim && source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
export CUDA_VISIBLE_DEVICES=0 F1SIM_EXAM_SAVE_MEM=1 F1SIM_EXAM_SAVE_SCAN=1 F1SIM_EXAM_SAVE_VALUE=1
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
for sd in 20261112 20261113 20261114; do
  O=$R/out/exam/examv_plan25y_u125_ideal_uniform_n256_s$sd
  F1SIM_EXAM_SEED=$sd python3 -W ignore $R/tools/exam_record.py /home/shchon11/f1sim_runs/plan25y/plan_u125.pt $O.npz ideal 256 uniform 90 50 > $O.log 2>&1
  echo "rc $?" >> $O.log
done
touch $R/out/exam/examv_plan25y_u125.done
