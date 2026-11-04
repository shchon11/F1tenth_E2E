#!/bin/bash
# 긴 학습 (11-05 02:25). 네트워크 하나, 네 경우 world, 빠른 solo policy에서 시작 + 탐색 std 절반: 빈 트랙 solo lap 6.58~6.60 s로 u50~u200 내내 유지
# (보통 noise는 6.71 -> 6.90 s). 하지만 obstacle·차는 처음부터 배워서 u200에 30 layout hard 288, race 1.0배 hard 차 16(추월 20/20). 실력을 갖고 시작
# (supervised 512칸)하면 PPO에서 속도가 무너짐(8.36 s, critic 예열해도). 그래서 속도가 지켜지는 설정으로 오래 학습.
# 기각(미리 정함, 시드 777): u600까지 obstacle 30 layout hard가 100 아래로 안 내려가거나 solo lap > 6.65 s.
# 평가: 50 update마다 solo lap, 200 update마다 obstacle 30 layout + race forzaeth 0.6 / 1.0배, 끝에 전체(evalc).
INIT=$1; NAME=$2; GPU=$3; ENVS=$4; S=$5; U=$6; MAPK=$7; PAR=0; EXTRA=$8
if [ "$MAPK" = 26 ]; then M=real:korea_2026_competition; MC=real/korea26; else M=real:korea_2025_iccas; MC=real/iccas25; fi
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/one_${NAME}.log
echo "== ${NAME}: 긴 학습, 네트워크 하나 + 탐색 std 절반, init $(basename $(dirname $INIT))/$(basename $INIT), 추가 인자 $EXTRA" > $L
evalc() {
  local UU=$1; local C=$R/${NAME}/state_u${UU}.pt
  CUDA_VISIBLE_DEVICES=$GPU timeout 7200 python3 $T/console_layouts.py $C $MC 40 30 777 $D/layouts/o_${NAME}_u${UU} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_o_${NAME}_u${UU}.log
  echo "  u${UU} obstacle 30 layout: $(tail -n 1 $D/layouts_o_${NAME}_u${UU}.log)" >> $L
  for x in "forzaeth 0.6" "forzaeth 1.0" "forzaeth 1.2"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$GPU timeout 5400 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/${NAME}_u${UU}_$1_x$2 2>&1 | grep -v Warning | grep "합계" | sed "s/^/  u$UU $1 x$2 /" >> $L
  done
  for x in "forzaeth 0.6" "forzaeth 1.0"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$GPU timeout 7200 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/obs_${NAME}_u${UU}_$1_x$2 1 2>&1 | grep -v Warning | grep "합계" | sed "s/^/  u$UU obstacle+$1 x$2 /" >> $L
  done
  CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 -u $T/follow_probe.py $C $MC 30 12 forzaeth 0.8 777 > $D/follow_probe_${NAME}_u${UU}_x0.8.log 2>&1
  echo "  u${UU} 따라가기 0.8배: $(grep '+3 ~   +6\|-1 ~   +1' $D/follow_probe_${NAME}_u${UU}_x0.8.log | tr -s ' ' | tr '\n' ';')" >> $L
  echo done > $D/one_${NAME}_u${UU}.done
}
( for UU in $(seq 50 50 $U); do
    until [ -e $R/${NAME}/state_u${UU}.pt ] || [ -e $D/one_${NAME}.done ]; do sleep 20; done
    [ -e $R/${NAME}/state_u${UU}.pt ] || break
    echo "  u${UU} solo: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${NAME}/state_u${UU}.pt $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $L
    if [ $((UU % 200)) = 0 ] && [ $UU != $U ]; then
      C=$R/${NAME}/state_u${UU}.pt
      echo "  u${UU} obstacle 30 layout: $(CUDA_VISIBLE_DEVICES=$GPU timeout 7200 python3 $T/console_layouts.py $C $MC 40 30 777 $D/layouts/o_${NAME}_u${UU} 2>&1 | grep '합계' | cut -c1-200)" >> $L
      for x in "forzaeth 0.6" "forzaeth 1.0"; do
        set -- $x
        CUDA_VISIBLE_DEVICES=$GPU timeout 5400 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/${NAME}_u${UU}_$1_x$2 2>&1 | grep -v Warning | grep "합계" | sed "s/^/  u$UU $1 x$2 /" >> $L
      done
    fi
    if [ "$PAR" = 1 ] && [ $UU = $((U / 2)) ]; then evalc $UU; fi
  done ) &
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --noise-corr 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 0 --level-replay 0 --adv-norm-case 1 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.5 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 --severity-min 1.0 \
  --prop-touch-price 40 --prop-bank $R/prop_banks"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage race $PL $COMMON --updates $U \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed $S \
  --opp-present 0.5 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.2]}' \
  --init $INIT $EXTRA > $D/${NAME}.log 2>&1 || { echo fail > $D/one_${NAME}.done; exit 1; }
wait
evalc $U
echo done > $D/one_${NAME}.done
