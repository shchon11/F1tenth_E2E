#!/bin/bash
# net_audit2.py (plan 뒷부분이 실제로 간 길을 맞히나)만 다시: 새 인코더를 가르치는 부분은 5060에서 (작은 일, 1 GB 미만)
cd /home/shchon11/F1tenth_E2E/f1sim && source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
O=$R/out/netaudit/plan25y_u125_b
python3 $R/tools/net_audit2.py /home/shchon11/f1sim_runs/plan25y/plan_u125.pt $R/out/exam/exam1_plan25y_u125_ideal_uniform_n256.npz ${O}_pairs.npz ${O}_forecast2 > ${O}_forecast2.log 2>&1
echo "rc $?" >> ${O}_forecast2.log
touch ${O}_forecast2.done
