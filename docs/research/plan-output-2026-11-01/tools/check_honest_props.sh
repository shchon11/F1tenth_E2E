#!/bin/bash
# 정직한 연쇄 상자 200회 확인 (5060): 각 맵 혼자 60 s(속도 유지), 학습 장애물 60 s(벽·상자 충돌), 궤적 일관성
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
until [ -f ~/f1sim_runs/srp25h2_props/state_u200.pt ] && [ -f ~/f1sim_runs/srp26xh2_props/state_u200.pt ]; do sleep 30; done; sleep 20
for X in "srp25h2 real/iccas25" "srp26xh2 real/korea26x12"; do
  set -- $X
  echo "== $1 상자 200회 @ $2"
  echo "혼자: $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py ~/f1sim_runs/$1_props/state_u200.pt $2 60 2>&1 | grep '주행\|Error')"
  echo "학습 장애물: $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py ~/f1sim_runs/$1_props/state_u200.pt $2 60 - 1 2>&1 | grep '주행\|Error')"
  CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/plan_consistency.py ~/f1sim_runs/$1_props/state_u200.pt $2 30 2>&1 | grep -v Warning | tail -3
done
echo done > $D/check_honest_props.done
