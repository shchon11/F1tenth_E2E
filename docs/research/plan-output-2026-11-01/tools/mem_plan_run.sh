#!/bin/bash
# (11-15 19:10) 기억이 plan을 바꾸나: plan26s u900(기억 비교에서 상자 이득이 처음 보인 checkpoint) 시험을 기억까지 기록 -> mem_plan.py
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; O=$R/out/diag1115
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
N=$O/memrec_26s_u900.npz
[ -e $N ] || F1SIM_EXAM_SAVE_MEM=1 CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/plan26s/plan_u900.pt $N ideal 256 uniform > ${N%.npz}.log 2>&1
cd $T
EXAM_ROWS=$O/memrec_26s_u900_rows.npy python3 -W ignore exam_boxes.py $N > $O/memrec_26s_u900_boxes.txt 2>&1
python3 -W ignore mem_plan.py /home/shchon11/f1sim_runs/plan26s/plan_u900.pt $N $O/memrec_26s_u900_rows.npy 2>&1
