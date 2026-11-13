#!/bin/bash
# 같은 장면(상자가 가는 길 위에 보이는 결정)을 plan25y의 checkpoint마다 넣어, 상자에 반응하는 거리가 학습이 가면서 늘어나는지 봄. CPU만.
cd /home/shchon11/F1tenth_E2E/f1sim && source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
Y=/home/shchon11/f1sim_runs/plan25y
O=$R/out/netaudit/ckpts_plan25y
NET_HITS_CKPTS=$R/out/netaudit/plan25y_u125_b_pairs.npz python3 -W ignore $R/tools/net_audit_hits.py $Y/plan_u125.pt $O $R/out/exam/exam1_plan25y_u125_ideal_uniform_n256.npz \
  $Y/plan_u25.pt $Y/plan_u50.pt $Y/plan_u75.pt $Y/plan_u100.pt $Y/plan_u125.pt $Y/plan_u150.pt > $O.log 2>&1
echo "rc $?" >> $O.log; touch $O.done
