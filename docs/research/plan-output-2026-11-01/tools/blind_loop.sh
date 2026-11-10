#!/bin/bash
# 사각지대 상자 기억 측정(blind_probe.py)을 checkpoint마다 (11-10, plan25n5 --mem-aux 판정 1). 결과 out/analysis/NAME_blind.log
# 인자: 이름 시작 간격 끝
NAME=$1; U0=$2; STEP=$3; U1=$4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
S="bash $T/slot5060.sh"
for UU in $(seq $U0 $STEP $U1); do
  C=$R/${NAME}/plan_u${UU}.pt
  [ -e $D/analysis/${NAME}_u${UU}_blind.done ] && continue
  until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 10
  CUDA_VISIBLE_DEVICES=1 $S timeout 1800 python3 -u $T/blind_probe.py $C 256 30 2>&1 | grep "맞힌 비율" >> $D/analysis/${NAME}_blind.log
  echo done > $D/analysis/${NAME}_u${UU}_blind.done
done
