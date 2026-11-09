#!/bin/bash
# 추이 표의 시작 행: run이 이어 받은 checkpoint를 같은 분석·평가로 (11-09, 고정 추이 표). 결과는 analysis/RUN_start.log(== u0 ...),
# out/one_RUN_start.log(u0). 인자: run 시작checkpoint 이름표
RUN=$1; C=$2; TAG=$3
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
A=$D/analysis
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
L=$A/${RUN}_start.log
echo "== u0 시작 $TAG" > $L
CUDA_VISIBLE_DEVICES=1 timeout 1500 python3 -u $T/mem_probe.py $C $A/${RUN}_u0_mem.png 60 256 2>&1 | grep "맞힘" >> $L
CUDA_VISIBLE_DEVICES=1 timeout 1200 python3 -u $T/accel_usage.py $C 64 60 2>&1 | grep "올릴 때\|회전이 막지\|내릴 때" >> $L
CUDA_VISIBLE_DEVICES=1 timeout 1200 python3 -u $T/line_mintime.py $C 64 2>&1 | grep "실제 바퀴\|plan 모델 한계" >> $L
CUDA_VISIBLE_DEVICES=1 timeout 1200 python3 -u $T/tail_bias.py $C 64 40 2>&1 | grep "plan - 실제" >> $L
for M in ideal track map; do
  CUDA_VISIBLE_DEVICES=1 timeout 3000 python3 $T/plan_validate.py $C $M 256 40 "  u0 " 2>&1 | grep "^  u0" >> $D/one_${RUN}_start.log
done
echo done > $A/${RUN}_start.done
