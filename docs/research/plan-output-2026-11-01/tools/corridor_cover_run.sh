#!/bin/bash
# 통로 칸이 주행선 위 상자를 담나 (11-03, 5060): 콘솔과 같은 30배치 중 0..20(멈추는 5, 11, 20 포함), 시드 777
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
CUDA_VISIBLE_DEVICES=1 python3 -u ../docs/research/plan-output-2026-11-01/tools/corridor_cover.py /home/shchon11/f1sim_runs/srp25h3g2_props/state_u100.pt real/iccas25 21 777 2>&1 | grep -v Warning > $O/corridor_cover.log
touch $O/corridor_cover.done
