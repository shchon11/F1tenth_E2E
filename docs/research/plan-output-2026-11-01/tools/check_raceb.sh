#!/bin/bash
# 혼합 상대 단계 U회 확인 (4070, 지킴이가 그동안 4070 학습을 멈춤): 맵마다 혼자 / 학습 장애물 / 1:1 vs forzaeth x0.7, x1.0 (각 60 s)
U=$1
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
until [ -f ~/f1sim_runs/srp25h2_raceb/state_u$U.pt ] && [ -f ~/f1sim_runs/srp26xh2_raceb/state_u$U.pt ]; do sleep 60; done; sleep 20
for X in "srp25h2 real/iccas25" "srp26xh2 real/korea26x12"; do
  set -- $X
  C=~/f1sim_runs/$1_raceb/state_u$U.pt
  echo "== $1 혼합 상대 단계 $U회 @ $2"
  echo "혼자: $(CUDA_VISIBLE_DEVICES=0 timeout 900 python3 $T/console_laps.py $C $2 60 2>&1 | grep '주행\|Error')"
  echo "학습 장애물: $(CUDA_VISIBLE_DEVICES=0 timeout 900 python3 $T/console_laps.py $C $2 60 - 1 2>&1 | grep '주행\|Error')"
  for SC in 0.7 1.0; do
    CUDA_VISIBLE_DEVICES=0 timeout 900 python3 $T/console_race.py $C $2 60 forzaeth $SC 2>&1 | grep '경주\|Error' | tail -1
  done
done
echo done > $D/check_raceb_u$U.done
