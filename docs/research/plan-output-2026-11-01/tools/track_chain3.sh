#!/bin/bash
# 4070 (00:50): 잡음 0.7 측정 뒤 -> 16 덩어리 합침 + 속도 짐작 손실 무게 0.1로 8 update (u150에서, 측정용). 채점은 5060.
until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/track/chain2.done ]; do sleep 10; done
sleep 5
bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/track_steps2.sh 0 plan25y_trk150_acc16_v01 0 8 16 0.1
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
C=/home/shchon11/f1sim_runs/plan25y_trk150_acc16_v01/plan_u8.pt; O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/exam1_plan25y_trk150_acc16_v01_u8_ideal_uniform_n256
[ -e $C ] && { CUDA_VISIBLE_DEVICES=1 bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/slot5060.sh timeout 3000 python3 -u -W ignore /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/exam_record.py $C $O.npz ideal 256 uniform > $O.log 2>&1; echo "rc $?" >> $O.log; }
touch /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/track/chain3.done
