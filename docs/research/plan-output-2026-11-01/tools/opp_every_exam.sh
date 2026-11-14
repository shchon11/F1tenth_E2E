#!/bin/bash
# 상대 차 planner 계산 간격 시험 (11-14): plan26f u150을 frozen exam(ideal, 상자 uniform, 256)으로, 상대 차 계산 간격 2와 4 (1은 이미 있음).
# 볼 것: 상대 차 자체(속도, 선 시간, 다시 출발, 세게 닿음)와 우리 policy의 결과가 간격 1과 같은가.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
C=/home/shchon11/f1sim_runs/plan26f/plan_u150.pt
for k in 2 4; do
  O=$E/exam1_plan26f_u150_ideal_uniform_n256_opp$k
  [ -e $O.npz ] || { F1SIM_OPP_EVERY=$k CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 3000 python3 -u -W ignore $T/exam_record.py $C $O.npz ideal 256 uniform > $O.log 2>&1; echo "rc $?" >> $O.log; }
  python3 -W ignore $T/exam_table.py $O.npz > $O.txt 2>&1
done
python3 -W ignore $T/exam_table.py $E/exam1_plan26f_u150_ideal_uniform_n256.npz > $E/exam1_plan26f_u150_ideal_uniform_n256_opp1.txt 2>&1
touch $E/opp_every.done
