#!/bin/bash
# 선생이 랩 위치로 선을 외웠나 (11-03, 5060): 혼합 선생 srp25h3g2 u100(혼자 정책 srp25h2_solo2 u300 포함), 깨끗한 Korea 2025
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
CUDA_VISIBLE_DEVICES=1 python3 -u ../docs/research/plan-output-2026-11-01/tools/lapfrac_probe.py /home/shchon11/f1sim_runs/srp25h3g2_props/state_u100.pt real/iccas25 2>&1 | grep --line-buffered -v Warning > $O/lapfrac_probe.log
touch $O/lapfrac_probe.done
