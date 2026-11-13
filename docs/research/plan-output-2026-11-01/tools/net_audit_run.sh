#!/bin/bash
# 네트워크 구조 점검(net_audit.py, net_audit2.py)을 plan25y u125 기록으로 돌림 (CPU만). 결과: out/netaudit/plan25y_u125*.{txt,png}, 끝나면 .done
# NET_AUDIT_SKIP_CKS=1: checkpoint별 표(3부)는 건너뜀
cd /home/shchon11/F1tenth_E2E/f1sim && source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
Y=/home/shchon11/f1sim_runs/plan25y; C=/home/shchon11/f1sim_runs/plan26c
CKS="25y_u25=$Y/plan_u25.pt,25y_u50=$Y/plan_u50.pt,25y_u100=$Y/plan_u100.pt,25y_u125=$Y/plan_u125.pt,25y_u150=$Y/plan_u150.pt,26c_u25=$C/plan_u25.pt,26c_u50=$C/plan_u50.pt,26c_u100=$C/plan_u100.pt,26c_u125=$C/plan_u125.pt"
[ -n "$NET_AUDIT_SKIP_CKS" ] && CKS=""
REC=$R/out/exam/exam1_plan25y_u125_ideal_uniform_n256.npz
O=$R/out/netaudit/plan25y_u125${NET_AUDIT_TAG}
NET_AUDIT_CKS="$CKS" python3 $R/tools/net_audit.py $Y/plan_u125.pt $REC $O > $O.log 2>&1
echo "rc $?" >> $O.log
python3 $R/tools/net_audit2.py $Y/plan_u125.pt $REC ${O}_pairs.npz ${O}_forecast > ${O}_forecast.log 2>&1
echo "rc $?" >> ${O}_forecast.log
touch $O.done
