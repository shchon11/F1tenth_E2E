#!/bin/bash
# (11-15 13:37) 가장 좋은 policy(avg26s = plan26s u650·700·750 평균)를 MAP 제어기로(실차 방식) 시험 -- 이정표마다 MAP 기준 한 줄
#   비교: plan26i u150 MAP 기록(11-14, 같은 시험), 같은 policy의 plan대로 정확히(ideal) 기록
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
N=$E/exam1_avg26s_u700_map_uniform_n256.npz
[ -e $N ] || CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 3600 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/avg26s/plan_u700.pt $N map 256 uniform > ${N%.npz}.log 2>&1
cd $T
echo "== MAP: plan26i u150 기준"; python3 -W ignore grad_step_table.py $E/exam1_plan26i_u150_map_uniform_n256.npz "avg26s (MAP)=$N" 2>&1 | tail -4
echo "== 같은 policy: plan대로 정확히 기준"; python3 -W ignore grad_step_table.py $E/exam1_avg26s_u700_ideal_uniform_n256.npz "avg26s (MAP)=$N" 2>&1 | tail -4
