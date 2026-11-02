#!/bin/bash
# 경주 기준선 (11-02 09:20, 5060): Korea 2025, 상자 단계 선생(+혼합)과 상대 단계 u100(+혼합), forzaeth 0.6배속 / 1.0배속, 20판 x 30 s
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
for X in "props $R/srp25h2_propsmix/state_u200.pt" "raceb $R/srp25h2_raceb/state_u100.pt"; do
  set -- $X
  for SC in 0.6 1.0; do
    CUDA_VISIBLE_DEVICES=1 timeout 3000 python3 $T/console_races.py $2 real/iccas25 30 20 forzaeth $SC 777 $D/races/${1}_x$SC 2>&1 | grep "판\|합계\|Error" | sed "s/^/$1 x$SC /" >> $D/races_base.log
  done
done
echo done > $D/races_base.done
