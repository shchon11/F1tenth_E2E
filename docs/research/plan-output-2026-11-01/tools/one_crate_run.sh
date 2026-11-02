#!/bin/bash
# 상자 하나를 혼자 주행선 12곳에 (11-03, 5060): 전 선생(정답 상자 다 봄) / 새 선생(보이는 상자만) / 이어 학습 / 학생
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
CUDA_VISIBLE_DEVICES=1 python3 -u ../docs/research/plan-output-2026-11-01/tools/one_crate.py real/iccas25 30x34 \
  srp25h2=$(readlink -f $R/pick_srp25h2/state_latest.pt) srp25h3=$R/srp25h3_propsmix/state_u200.pt srp25h3b=$R/srp25h3b_propsmix/state_u200.pt \
  sst25h2=$R/sst_srp25h2_props/student_i80.pt 2>&1 | grep -v Warning > $O/one_crate25.log
touch $O/one_crate25.done
