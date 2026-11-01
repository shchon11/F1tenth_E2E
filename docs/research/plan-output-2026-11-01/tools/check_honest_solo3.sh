#!/bin/bash
# 정직한 연쇄 혼자 150회의 반증 확인 (5060): ×1.2 정직한 레이싱 라인 기준선, 그다음 두 선생을 콘솔 워커 60 s + 궤적 일관성 + 그림
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
until [ -f $D/chain_srp25h2.solo2_done ] && [ -f $D/chain_srp26xh2.solo2_done ]; do sleep 30; done; sleep 10
for X in "srp25h2 real/iccas25" "srp26xh2 real/korea26x12"; do
  set -- $X
  echo "== $1 solo2 300회 @ $2"
  CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py ~/f1sim_runs/$1_solo2/state_u300.pt $2 60 2>&1 | grep '주행\|Error'
  CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/plan_consistency.py ~/f1sim_runs/$1_solo2/state_u300.pt $2 30 2>&1 | grep -v Warning | tail -3
  CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/drive_pic.py ~/f1sim_runs/$1_solo2/state_u300.pt $2 20 $D/../drive_$1_solo2_u300.png 2>&1 | grep "그림\|Error"
done
echo done > $D/check_honest_solo3.done
