#!/bin/bash
# 세게 닿은 장면마다 닿기 1.0 / 0.5 / 0.2 s 전에 낸 plan을 그대로 달렸다면 닿았는지, 차가 그 plan에서 어느 쪽으로 벗어났는지 셈. CPU만.
cd /home/shchon11/F1tenth_E2E/f1sim && source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
O=$R/out/netaudit/hits_plan25y_u125
E=$R/out/exam/exam1_plan25y_u125_ideal
NET_HITS_PLANGAP=1 python3 -W ignore $R/tools/net_audit_hits.py /home/shchon11/f1sim_runs/plan25y/plan_u125.pt $O ${E}_uniform_n256.npz ${E}_max2_n256.npz ${E}_max2_n256_s20261113.npz ${E}_max2_n256_s20261114.npz > ${O}_plangap.log 2>&1
echo "rc $?" >> ${O}_plangap.log; touch ${O}_plangap.done
