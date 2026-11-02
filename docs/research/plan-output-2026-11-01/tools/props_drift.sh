#!/bin/bash
# 상자 단계 동안 빈 길 속도가 어떻게 떨어졌나 (11-03): 깨끗한 맵 60 s, 체크포인트별 (5060)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
for C in srp25h2_solo2/state_u300.pt srp25h2_props/state_u50.pt srp25h2_props/state_u100.pt srp25h2_props/state_u150.pt srp25h2_props/state_u200.pt; do
  echo "$C: $(CUDA_VISIBLE_DEVICES=1 timeout 600 python3 $T/console_laps.py $R/$C real/iccas25 60 2>&1 | grep '주행' | cut -c1-170)"
done > $D/props_drift25.log
echo done > $D/props_drift25.done
