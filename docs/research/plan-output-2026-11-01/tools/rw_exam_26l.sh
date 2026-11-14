#!/bin/bash
# (11-14 21:40) 질문: 학습 보상(실제 지도, 잡음 없이 / 학습 잡음으로)이 plan26l u150(빠르고 더 박음)을 plan26i u150·plan26l u50/u100보다 좋다고 매기나.
#   좋다고 매기면 -> 보상·세계가 "박아도 빠른 쪽"을 원함(학습은 시킨 대로 함). 나쁘다고 매기면 -> 학습이 보상을 못 따라감(신호·탐색 쪽).
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; O=$R/out/rwexam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
for job in "150 0" "150 1" "100 0" "50 0"; do
  set -- $job; U=$1; SM=$2
  N=$O/plan26l_u${U}_s${SM}
  [ -e $N.npz ] && continue
  EXAM_SAMPLE=$SM EXAM_SAMPLE_SEED=1 CUDA_VISIBLE_DEVICES=0 timeout 2400 python3 -u -W ignore $T/exam_reward.py /home/shchon11/f1sim_runs/plan26l/plan_u${U}.pt $N.npz ideal 256 uniform > $N.log 2>&1
  echo "rc $?" >> $N.log
done
cd $T
python3 -W ignore rw_sum.py 26i_u150=$O/plan26i_u150_s0.npz 26l_u50=$O/plan26l_u50_s0.npz 26l_u100=$O/plan26l_u100_s0.npz 26l_u150=$O/plan26l_u150_s0.npz > $O/sum_26l_s0.txt 2>&1
python3 -W ignore rw_sum.py 26i_u150=$O/plan26i_u150_s1.npz 26l_u150=$O/plan26l_u150_s1.npz > $O/sum_26l_s1.txt 2>&1
touch $O/rw26l.done
