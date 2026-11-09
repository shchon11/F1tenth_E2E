#!/bin/bash
# 추이 표의 시작 행: run이 이어 받은 checkpoint를 같은 분석·평가로 (11-09, 고정 추이 표). 결과는 analysis/RUN_start.log(== u0 ...),
# out/one_RUN_start.log(u0). 인자: run 시작checkpoint 이름표
# (11-09 v2) 분석만, 5060 자리 2개로 -- 시작 checkpoint의 평가는 앞 단계 run의 100 update 평가에 이미 있음. 빈 트랙 라인 그림도.
RUN=$1; C=$2; TAG=$3
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
A=$D/analysis
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
S="bash $T/slot5060.sh"   # at most two heavy jobs on the 5060 (11-09 out-of-memory)
L=$A/${RUN}_start.log
echo "== u0 시작 $TAG" > $L
CUDA_VISIBLE_DEVICES=1 $S timeout 1500 python3 -u $T/mem_probe.py $C $A/${RUN}_u0_mem.png 60 256 2>&1 | grep "맞힘" >> $L
CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/accel_usage.py $C 64 60 2>&1 | grep "올릴 때\|회전이 막지\|내릴 때" >> $L
CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/line_mintime.py $C 64 2>&1 | grep "실제 바퀴\|plan 모델 한계" >> $L
CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/line_compare.py $C ideal 32 40 $A/${RUN}_u0_line.png 2>&1 | grep "png" >> $L
CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/tail_bias.py $C 64 40 2>&1 | grep "plan - 실제" >> $L
echo done > $A/${RUN}_start.done
