#!/bin/bash
# (11-15 00:00) plan26o u125 학습 조건 평가가 5060 메모리 부족(Korea 2026 상자 배치 작업이 같이 돎)으로 출력 없이 끝나 다시 돌림
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out; T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 $T/plan_validate.py /home/shchon11/f1sim_runs/plan26o/plan_u125.pt ideal 512 40 "  u125 " > $D/redo_u125_26o.raw 2>&1
grep "^  u125" $D/redo_u125_26o.raw >> $D/one_plan26o_i.log
touch $D/redo_u125_26o.done
