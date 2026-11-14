#!/bin/bash
# (11-14 17:50, 사용자: "분석을 좀 심도깊게 하고 정해") plan26i u150/u200/u250/u300을 frozen exam1 같은 세계에서
#   (a) 잡음 없이(frozen exam과 같은 행동) + 결정마다 학습 보상 기록 -> 보상이 어느 checkpoint를 더 좋다고 보는가
#   (b) 학습 rollout과 같은 탐색 잡음으로 -> 잡음 섞인 행동은 세게 닿음이 어떻게 변했나
# 4070에서 하나씩(그동안 gpu_guard가 plan26j를 멈춤). 결과 out/rwexam/
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; O=$R/out/rwexam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
for job in "150 0" "300 0" "150 1" "300 1" "200 0" "250 0" "200 1" "250 1"; do
  set -- $job; U=$1; SM=$2
  N=$O/plan26i_u${U}_s${SM}
  [ -e $N.npz ] && continue
  EXAM_SAMPLE=$SM EXAM_SAMPLE_SEED=1 CUDA_VISIBLE_DEVICES=0 timeout 2400 python3 -u -W ignore $T/exam_reward.py /home/shchon11/f1sim_runs/plan26i/plan_u${U}.pt $N.npz ideal 256 uniform > $N.log 2>&1
  echo "rc $?" >> $N.log
done
touch $O/all.done
