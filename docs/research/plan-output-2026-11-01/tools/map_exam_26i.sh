#!/bin/bash
# (11-14 17:00) plan26i u150(대표)과 plan25y u150을 MAP 실행(실차 방식: MAP이 plan을 따라감)으로 frozen exam -- 이정표마다 MAP 기준 한 줄
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
for rn in plan26i plan25y; do
  O=$E/exam1_${rn}_u150_map_uniform_n256
  [ -e $O.npz ] || { CUDA_VISIBLE_DEVICES=0 timeout 3000 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/$rn/plan_u150.pt $O.npz map 256 uniform > $O.log 2>&1; echo "rc $?" >> $O.log; }
done
touch $E/map_exam_26i.done
