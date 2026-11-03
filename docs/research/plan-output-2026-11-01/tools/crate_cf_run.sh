#!/bin/bash
# 멈추는 상자에 정책이 반응하나 (11-03, 5060): 콘솔 30배치 중 5, 11, 20, 시드 777. 인자: 이름(srp25h3g2_props 등)
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
CUDA_VISIBLE_DEVICES=1 python3 -u ../docs/research/plan-output-2026-11-01/tools/crate_cf.py /home/shchon11/f1sim_runs/$1/state_u100.pt real/iccas25 5,11,20 777 2>&1 | grep --line-buffered -v Warning > $O/crate_cf_$1.log
touch $O/crate_cf_$1.done
