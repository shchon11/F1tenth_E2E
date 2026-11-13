#!/bin/bash
# 5060: 추적 측정 run들의 u8 checkpoint를 frozen exam(ideal, 상자 uniform, 256)으로 -> u150 기준과 짝지어 비교 (8 update가 주행을 얼마나 바꿨나)
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/gstep150_chain3.done ]; do sleep 15; done
for run in plan25y_trk150 plan25y_trk150_acc16 plan25y_trk150_c7; do
  until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/track/$run/done ]; do sleep 15; done
  C=/home/shchon11/f1sim_runs/$run/plan_u8.pt; O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/exam1_${run}_u8_ideal_uniform_n256
  [ -e $C ] || { echo "$run: u8 없음"; continue; }
  CUDA_VISIBLE_DEVICES=1 bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/slot5060.sh timeout 3000 python3 -u -W ignore /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/exam_record.py $C $O.npz ideal 256 uniform > $O.log 2>&1
  echo "rc $?" >> $O.log
done
touch /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/track_exam_chain.done
