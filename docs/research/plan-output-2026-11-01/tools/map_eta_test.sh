#!/bin/bash
# (11-15 21:00) MAP의 추종 각도를 차체 방향(지금) 대신 실제 진행 방향(L1 유도법 원래 정의)에서 재면 MAP 벽 충돌이 줄어드나
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
until [ -e $R/out/maprepro/exam1_plan26i_u150_map_uniform_n256_now.npz ]; do sleep 20; done
cd /home/shchon11/F1tenth_E2E/f1sim
N=$E/exam1_avg26s_u900_map_uniform_n256_etavel.npz
[ -e $N ] || F1SIM_MAP_ETA_REF=velocity CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 3600 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/avg26s/plan_u900.pt $N map 256 uniform > ${N%.npz}.log 2>&1
cd $T
python3 -W ignore grad_step_table.py $E/exam1_avg26s_u900_ideal_uniform_n256.npz "MAP (차체 방향)=$E/exam1_avg26s_u900_map_uniform_n256.npz" "MAP (진행 방향)=$N" 2>&1 | tail -4
