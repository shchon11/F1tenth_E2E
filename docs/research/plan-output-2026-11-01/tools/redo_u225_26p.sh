#!/bin/bash
# (11-15 01:58) plan26p u225 학습 조건 평가가 두 GPU 짧은 시험(01:38)과 겹쳐 5060 메모리 부족으로 출력 없이 끝나 다시 돌림
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out; T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 $T/plan_validate.py /home/shchon11/f1sim_runs/plan26p/plan_u225.pt ideal 512 40 "  u225 " > $D/redo_u225_26p.raw 2>&1
grep "^  u225" $D/redo_u225_26p.raw >> $D/one_plan26p_i.log
touch $D/redo_u225_26p.done
