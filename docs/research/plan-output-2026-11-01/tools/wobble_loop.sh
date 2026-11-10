#!/bin/bash
# checkpoint마다 흔들림 (11-10, plan25n6 지그재그 비용의 판정): 학습 조건(wobble_probe, 빈 트랙 차 64대 x 30 s) +
# 콘솔 경로 새 제어기(console_wobble, 혼자 빈 트랙 30 s). 결과 out/analysis/NAME_wobble.log. 인자: 이름 시작 간격 끝
NAME=$1; U0=$2; STEP=$3; U1=$4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
S="bash $T/slot5060.sh"
L=$D/analysis/${NAME}_wobble.log
for UU in $(seq $U0 $STEP $U1); do
  C=$R/${NAME}/plan_u${UU}.pt
  [ -e $D/analysis/${NAME}_u${UU}_wobble.done ] && continue
  until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 10
  CUDA_VISIBLE_DEVICES=1 $S timeout 1500 python3 -u $T/wobble_probe.py $D/analysis/${NAME}_u${UU}_wobble.png $C 64 30 2>&1 | grep "흔들림" | sed "s/^/학습 조건 /" >> $L
  F1SIM_PLAN_MODE=track CUDA_VISIBLE_DEVICES=1 $S timeout 1500 python3 -u $T/console_wobble.py $C 30 2>&1 | grep "흔들림" >> $L
  echo done > $D/analysis/${NAME}_u${UU}_wobble.done
done
