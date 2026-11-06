#!/bin/bash
# 시간별 waypoint plan(--plan-format tw: 0.1 s x 10구간의 곡률 + 부호 있는 속도, 전진은 MAP / 후진은 후진 추종, 11-06 22:40, 사용자:
# "t=n 까지의 waypoint 모음으로 만드는게 정석", "좀더 촘촘하게", MPC 없이)로 처음부터. 상자만 있는 묶음(prop_banks_box2), 상자 뒤 출발 30 %,
# 멈춤 벌점(최근 2 s 진행), 느릴 때 큰 탐색 흔들림 + 0.5 s 이어지는 흔들림. 평가도 상자만 있는 묶음, 배치마다 상자 1개~전부 무작위.
# (아래는 원래 처음부터 스크립트의 설명) 처음부터(scratch) 네트워크 하나 + world 하나 PPO (11-06 03:05, 사용자: "애초에 그냥 scratch에서 하면 안 돼? Bias가 너무 껴 있을 것 같음",
# "틈만 나면 꼼수 쓰려고"). 이전 checkpoint 없이(--init 없음) 처음부터: 입력 = LiDAR 1081빔(전체 / obstacle에 맞은 빔 / 차에 맞은 빔, 가까운
# obstacle 위치 입력 없음), policy 후진, 고정 obstacle, 네 경우 world 그대로. 학습률·충돌 값은 state_rl 기본값(lr 3e-4 -> 1e-4, 탐색 std 0.61,
# 충돌 값 0에서 60 update 뒤 200 update 동안 80까지) = 10-29 처음부터 학습이 쓴 값. 평가는 EVAL_GPU에서(5060은 1081빔 학습으로 꽉 참).
# 인자: 이름 학습GPU 평가GPU 환경수 시드 업데이트 지도(25|26) [EXTRA] [평가 간격, 기본 50: EXTRA에 --save-every를 같게]
NAME=$1; GPU=$2; EGPU=$3; ENVS=$4; S=$5; U=$6; MAPK=$7; PAR=0; EXTRA=$8; EVERY=${9:-50}; INIT=scratch
if [ "$MAPK" = 26 ]; then M=real:korea_2026_competition; MC=real/korea26; else M=real:korea_2025_iccas; MC=real/iccas25; fi
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/one_${NAME}.log
echo "== ${NAME}: 시간별 waypoint plan(0.1 s x 10), 처음부터, 상자만 있는 묶음, 멈춤 벌점, 느릴 때 큰 흔들림" > $L
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
PL="--action plan --plan-format tw --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --noise-corr 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 0 --level-replay 0 --adv-norm-case 1 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.5 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 --severity-min 1.0 \
  --prop-touch-price 40 --prop-bank $R/prop_banks_box2 --prop-shapes boxes"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage race $PL $COMMON --updates $U \
  --seed $S \
  --opp-present 0.5 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.2]}' \
  $EXTRA > $D/${NAME}.log 2>&1 || { echo fail > $D/one_${NAME}.done; exit 1; }
wait
evalc $U
echo done > $D/one_${NAME}.done
