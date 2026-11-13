#!/bin/bash
# net_audit_hits.py: plan25y u125의 시험 기록 4개(스캔이 저장된 것)에서 세게 닿은 장면 따라가기. 네트워크는 CPU, 움직임 묶음의 닿음 판정은 4070. 기록마다 따로 돌리고 합친 뒤 장면 그림 두 장.
cd /home/shchon11/F1tenth_E2E/f1sim && source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
C=/home/shchon11/f1sim_runs/plan25y/plan_u125.pt
O=$R/out/netaudit/hits_plan25y_u125
E=$R/out/exam/exam1_plan25y_u125_ideal
i=0
for rec in ${E}_uniform_n256 ${E}_max2_n256 ${E}_max2_n256_s20261113 ${E}_max2_n256_s20261114; do
  i=$((i + 1))
  python3 -W ignore $R/tools/net_audit_hits.py $C ${O}_$i $rec.npz > ${O}_$i.log 2>&1 &
  [ $((i % 2)) -eq 0 ] && wait
done
wait
NET_HITS_MERGE=1 python3 -W ignore $R/tools/net_audit_hits.py $C $O ${O}_1.npz ${O}_2.npz ${O}_3.npz ${O}_4.npz > $O.log 2>&1
echo "rc $?" >> $O.log
for w in early late; do
  NET_HITS_SCENES=${O}_scenes_$w.png NET_HITS_WHICH=$w python3 -W ignore $R/tools/net_audit_hits.py $C $O ${E}_uniform_n256.npz ${E}_max2_n256.npz ${E}_max2_n256_s20261113.npz ${E}_max2_n256_s20261114.npz > ${O}_scenes_$w.log 2>&1
done
touch $O.done
