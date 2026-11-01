#!/bin/bash
# 콘솔 워커로 학습 장애물 여러 배치(시드) 재기 (11-02, 5060): 콘솔은 세션마다 장애물 시드를 무작위로 뽑으므로 한 배치로 판단하지 않는다.
# 인자: 이름표 체크포인트 맵 초 시드...
NAME=$1; C=$2; MAP=$3; SECS=$4; shift 4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
for S in "$@"; do
  echo "$NAME $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py $C $MAP $SECS - 1 $S 2>&1 | grep '주행\|Error')"
done
