#!/bin/bash
# one-world 두 단계 (11-04 03:00): correlated noise 0.9로 100 update(피하기 탐색) -> white noise로 100 update(다듬기, 평가). 예전 world의 gl -> gw 순서.
# 인자: 시작체크포인트 이름 GPU 시드
INIT=$1; NAME=$2; GPU=$3; S=$4
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
bash $T/one_world_x.sh $INIT ${NAME}c $GPU 1024 $S 100 0 0.9 || exit 1
bash $T/one_world_x.sh $R/${NAME}c/state_u100.pt ${NAME}w $GPU 1024 $((S + 1)) 100 1 0
