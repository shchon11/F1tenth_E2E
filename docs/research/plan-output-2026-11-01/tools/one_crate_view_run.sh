#!/bin/bash
# 상자 하나에 박는 장면 대 상자 없는 같은 주행 (11-03, 5060)
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
LINE=$R/srp25h2_propsmix/state_u200.pt
{
CUDA_VISIBLE_DEVICES=1 python3 -u ../docs/research/plan-output-2026-11-01/tools/one_crate_view.py real/iccas25 30x34 $R/srp25h3_propsmix/state_u200.pt 3,7,11 $O/onecrate_h3 $LINE
CUDA_VISIBLE_DEVICES=1 python3 -u ../docs/research/plan-output-2026-11-01/tools/one_crate_view.py real/iccas25 30x34 $R/srp25h3b_propsmix/state_u200.pt 5,7 $O/onecrate_h3b $LINE
} 2>&1 | grep -v Warning > $O/one_crate_view25.log
touch $O/one_crate_view25.done
