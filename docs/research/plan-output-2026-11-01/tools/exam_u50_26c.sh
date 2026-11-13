#!/bin/bash
# plan26c(닿는 값 2배 + 정말 서 있을 때만 벌점) u50: 박은 뒤 서 버리는 것이 줄었나(판정의 반증 2를 u50에서 먼저 봄).
# 같은 시험(exam1 ideal, 상자 uniform, 256세션 x 90 s)으로 plan25y·plan26a u50 기록(exam_pair_u50.sh)과 견줌.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
O=$R/out/exam/exam1_plan26c_u50_ideal_uniform_n256
if [ ! -e $O.npz ]; then
  CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 5400 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/plan26c/plan_u50.pt $O.npz ideal 256 uniform > $O.log 2>&1
  echo "rc $?" >> $O.log
fi
python3 -W ignore $T/exam_table.py $O.npz > $R/out/analysis/exam_u50_26c.txt 2>&1
echo done > $R/out/analysis/exam_u50_26c.done
