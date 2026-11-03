#!/bin/bash
# 네 번째 경우 평가 (11-04 02:30, 사용자: "아예 장애물 없고 혼자/상대방 포함, 장애물 있고 혼자/상대방 포함 이렇게 네개"): obstacle(콘솔 기본
# 새 layout) + 상대 차 race. console_races.py 9번째 인자 1. 다른 셋(빈 트랙 solo lap, obstacle solo 30 layout, 빈 트랙 race)은 one_world.sh가 봄.
# 인자: 출력이름 GPU 체크포인트들... -- 기다릴 파일은 WAIT 환경변수(공백 구분)
NAME=$1; GPU=$2; shift 2
for f in $WAIT; do until [ -e "$f" ]; do sleep 30; done; done
MC=real/iccas25
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/obsrace_${NAME}.log
: > $L
for C in "$@"; do
  until [ -e "$C" ]; do sleep 60; done
  TAG=$(basename $(dirname $C))_$(basename $C .pt)
  for x in "forzaeth 0.6" "forzaeth 1.0"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/obs_${TAG}_$1_x$2 1 2>&1 | grep -v Warning | grep "합계" | sed "s/^/$TAG obstacle+$1 x$2 /" >> $L
  done
done
echo done > $D/obsrace_${NAME}.done
