#!/bin/bash
# net_audit_ckpts_run.sh와 같은 장면을 아무 run의 checkpoint들에 넣음: net_audit_ckpts_any.sh RUN U1 U2 ... -> out/netaudit/ckpts_RUN.log. CPU만.
cd /home/shchon11/F1tenth_E2E/f1sim && source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
RUN=$1; shift
CK=(); for u in "$@"; do CK+=(/home/shchon11/f1sim_runs/$RUN/plan_u$u.pt); done
O=$R/out/netaudit/ckpts_$RUN
NET_HITS_CKPTS=$R/out/netaudit/plan25y_u125_b_pairs.npz python3 -W ignore $R/tools/net_audit_hits.py /home/shchon11/f1sim_runs/plan25y/plan_u125.pt $O $R/out/exam/exam1_plan25y_u125_ideal_uniform_n256.npz "${CK[@]}" > $O.log 2>&1
echo "rc $?" >> $O.log; touch $O.done
