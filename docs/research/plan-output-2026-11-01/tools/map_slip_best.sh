#!/bin/bash
# (11-15 16:45, 사용자: "Map 튜닝으로는 안돼?") 추천 policy(avg26s/plan_u900.pt)를 MAP / MAP + 미끄러짐 보정(속도 x 회전 속도 -> 표의 미끄럼 각;
#   11-12 plan25r u300에서 벽 세게 3.41 -> 0.72/세션)으로 시험. 같은 시드 128 세션, plan대로 정확히 기록과도 비교.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
CK=/home/shchon11/f1sim_runs/avg26s/plan_u900.pt
N1=$E/exam1_avg26s_u900_map_uniform_n256.npz
[ -e $N1 ] || CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 3600 python3 -u -W ignore $T/exam_record.py $CK $N1 map 256 uniform > ${N1%.npz}.log 2>&1
N2=$E/exam1_avg26s_u900_map_uniform_n256_sliptab.npz
[ -e $N2 ] || F1SIM_EXAM_SLIP_FRAME=table F1SIM_EXAM_SLIP_TABLE=$E/slip_table_plan25r_u300.npz CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 3600 python3 -u -W ignore $T/exam_record.py $CK $N2 map 256 uniform > ${N2%.npz}.log 2>&1
cd $T
python3 -W ignore grad_step_table.py $E/exam1_avg26s_u900_ideal_uniform_n256.npz "MAP=$N1" "MAP + 미끄러짐 보정=$N2" 2>&1 | tail -5
