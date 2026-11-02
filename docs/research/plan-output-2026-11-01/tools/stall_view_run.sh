#!/bin/bash
# 멈추는 상자 배치 20, 5 시간 순 보기 (11-03), srp25h3b_propsmix u200, 콘솔과 같은 시드 777
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
for L in 5 20; do
  CUDA_VISIBLE_DEVICES=0 python3 -u ../docs/research/plan-output-2026-11-01/tools/stall_view.py /home/shchon11/f1sim_runs/srp25h3b_propsmix/state_u200.pt real/iccas25 $L 777 $O/stall_h3b 2>&1 | grep -v Warning
done > $O/stall_view_h3b.log
touch $O/stall_view_h3b.done
