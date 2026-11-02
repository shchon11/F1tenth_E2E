#!/bin/bash
# 차 접촉 세기 분포 (4070, 콘솔 워커): 학습한 경주 정책(racep, racel) vs 학습 안 한 앞차-통로(pmcst). 같은 20판(시드 777).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
for x in "srp25h2_racep/state_u100.pt 1.0" "srp25h2_pmcst/state_u200.pt 1.0" "srp25h2_racel/state_u100.pt 1.0" "srp25h2_racep/state_u100.pt 0.6"; do
  set -- $x
  CUDA_VISIBLE_DEVICES=0 timeout 1200 python3 $T/car_impacts.py $R/$1 real/iccas25 30 20 forzaeth $2 777 2>&1 | grep "차 접촉\|Error"
done > $D/car_impacts.log
echo done > $D/car_impacts.done
