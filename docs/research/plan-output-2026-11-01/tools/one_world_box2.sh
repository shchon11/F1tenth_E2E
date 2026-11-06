#!/bin/bash
# 상자 크기 obstacle world에서 이어서 (11-06 10:10, 사용자: "이어서 하고 장애물 거의 뒤에 스폰되도록도 하면 좋을듯").
# one_world_scratch.sh와 같고, 다른 점: --init(이어 받을 checkpoint, 인자 10), obstacle = 상자 크기만(--prop-shapes boxes, 묶음
# prop_banks_box2), 학습 차 출발의 일부를 상자 0.65-1.6 m 뒤에서(--spawn-behind-prop), 충돌 값 80 고정(처음부터 run은 u260에 이미 80),
# 학습률은 처음부터 run의 그 시점 값에서 이어서 1e-4까지. 평가(콘솔 도구)도 F1SIM_PROP_BANK로 상자 묶음을 씀.
# 인자: 이름 학습GPU 평가GPU 환경수 시드 업데이트 지도(25|26) EXTRA 평가간격 init체크포인트
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2
export F1SIM_PROP_COUNT=uniform   # 11-06: 평가 배치도 1개~전부 무작위 (사용자: 최대한 다양한 상황)
NAME=$1; GPU=$2; EGPU=$3; ENVS=$4; S=$5; U=$6; MAPK=$7; PAR=0; EXTRA=$8; EVERY=${9:-50}; INIT=${10}
if [ "$MAPK" = 26 ]; then M=real:korea_2026_competition; MC=real/korea26; else M=real:korea_2025_iccas; MC=real/iccas25; fi
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/one_${NAME}.log
echo "== ${NAME}: 네트워크 하나 + 속도 신호 고침(hit-shaping 0, 경우별 advantage 정규화, level replay 0), init $(basename $(dirname $INIT))/$(basename $INIT), 상자 크기 obstacle" > $L
evalc() {
  local UU=$1; local C=$R/${NAME}/state_u${UU}.pt
  CUDA_VISIBLE_DEVICES=$EGPU timeout 7200 python3 $T/console_layouts.py $C $MC 40 30 777 $D/layouts/o_${NAME}_u${UU} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_o_${NAME}_u${UU}.log
  echo "  u${UU} obstacle 30 layout: $(tail -n 1 $D/layouts_o_${NAME}_u${UU}.log)" >> $L
  for x in "forzaeth 0.6" "forzaeth 1.0" "forzaeth 1.2"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$EGPU timeout 5400 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/${NAME}_u${UU}_$1_x$2 2>&1 | grep -v Warning | grep "합계" | sed "s/^/  u$UU $1 x$2 /" >> $L
  done
  for x in "forzaeth 0.6" "forzaeth 1.0"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$EGPU timeout 7200 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/obs_${NAME}_u${UU}_$1_x$2 1 2>&1 | grep -v Warning | grep "합계" | sed "s/^/  u$UU obstacle+$1 x$2 /" >> $L
  done
  CUDA_VISIBLE_DEVICES=$EGPU timeout 2400 python3 -u $T/follow_probe.py $C $MC 30 12 forzaeth 0.8 777 > $D/follow_probe_${NAME}_u${UU}_x0.8.log 2>&1
  echo "  u${UU} 따라가기 0.8배: $(grep '+3 ~   +6\|-1 ~   +1' $D/follow_probe_${NAME}_u${UU}_x0.8.log | tr -s ' ' | tr '\n' ';')" >> $L
  echo done > $D/one_${NAME}_u${UU}.done
}
( for UU in $(seq $EVERY $EVERY $U); do
    until [ -e $R/${NAME}/state_u${UU}.pt ] || [ -e $D/one_${NAME}.done ]; do sleep 20; done
    [ -e $R/${NAME}/state_u${UU}.pt ] || break
    echo "  u${UU} solo: $(CUDA_VISIBLE_DEVICES=$EGPU timeout 900 python3 $T/console_laps.py $R/${NAME}/state_u${UU}.pt $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $L
    echo "  u${UU} obstacle 앞 8 layout: $(CUDA_VISIBLE_DEVICES=$EGPU timeout 1800 python3 $T/console_layouts.py $R/${NAME}/state_u${UU}.pt $MC 40 8 777 $D/layouts/q_${NAME}_u${UU} 2>&1 | grep '합계' | cut -c1-200)" >> $L
    if [ "$PAR" = 1 ] && [ $UU = $((U / 2)) ]; then evalc $UU; fi
  done ) &
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --noise-corr 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 0 --level-replay 0 --adv-norm-case 1 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.5 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 --severity-min 1.0 \
  --prop-touch-price 40 --prop-bank $R/prop_banks_box2"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage race $PL $COMMON --updates $U \
  --seed $S \
  --opp-present 0.5 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.2]}' \
  --init $INIT --prop-shapes boxes --price-start 80 --price-hold 0 --price-ramp 1 $EXTRA > $D/${NAME}.log 2>&1 || { echo fail > $D/one_${NAME}.done; exit 1; }
wait
evalc $U
echo done > $D/one_${NAME}.done
