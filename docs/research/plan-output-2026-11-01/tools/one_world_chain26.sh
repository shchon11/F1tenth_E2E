#!/bin/bash
# 한국 2026 one-world 두 단계 (11-04 03:20): obstacle + 상대 차 평가의 비교 기준 두 줄이 나온 뒤(5060이 빔) correlated 0.9 100 -> white 100.
# 시작 = 2026 props correlated noise u100 (30 layout hard 26, 기각선 통과). 인자: 시작체크포인트 이름 GPU 시드
INIT=$1; NAME=$2; GPU=$3; S=$4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
until [ -e $D/bank_srp26bkc_props.done ] && [ $(cat $D/obsrace_night1.log 2>/dev/null | wc -l) -ge 2 ]; do sleep 30; done
bash $T/one_world_x26.sh $INIT ${NAME}c $GPU 1024 $S 100 0 0.9 || exit 1
bash $T/one_world_x26.sh $R/${NAME}c/state_u100.pt ${NAME}w $GPU 1024 $((S + 1)) 100 1 0
