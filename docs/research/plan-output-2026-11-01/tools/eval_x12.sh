#!/bin/bash
# forzaeth 1.2배 추가 (11-04 03:55, 사용자: "기존에 1.2배만 추가하는걸로 가자"): 학습 상대 속도 상한을 1.2배로 올렸으니 평가에도 1.2배.
# 기존 0.6/1.0배는 one_world_x.sh(obstacle 없음)와 eval_obs_race.sh(obstacle 있음)가 봄. 여기는 1.2배 20판, obstacle 없음/있음.
# 인자: 출력이름 GPU 체크포인트들... -- 기다릴 파일은 WAIT 환경변수(공백 구분)
NAME=$1; GPU=$2; shift 2
for f in $WAIT; do until [ -e "$f" ]; do sleep 30; done; done
MC=real/iccas25
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/x12_${NAME}.log
for C in "$@"; do
  until [ -e "$C" ]; do sleep 60; done
  TAG=$(basename $(dirname $C))_$(basename $C .pt)
  for P in 0 1; do
    CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 $T/console_races.py $C $MC 30 20 forzaeth 1.2 777 $D/races/x12_${TAG}_p${P} $P 2>&1 | grep -v Warning | grep "합계" | sed "s/^/$TAG obstacle=$P forzaeth x1.2 /" >> $L
  done
done
echo done > $D/x12_${NAME}.done
