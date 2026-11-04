#!/bin/bash
# base를 섞은 채로 world 하나 (11-04 15:40). 네트워크 하나는 빈 트랙 속도를 잃음: one-world 2025 6.50 -> 6.90 s(u100), 2026 7.40 -> 7.65 s(u50),
# 10-31 race 단계도 같음(사용자 11-04: "cascade forgetting 때문인가?" -> 그렇다). 고정 solo policy + 보정(ResidualPi)은 빈 트랙은 지키지만 obstacle을
# 못 피함(0에서 시작 u100 30 layout hard 272, one-world actor에서 시작 u0 -> u50 앞 8 layout 38 -> 39, 라인 위 상자에 3~4 m/s로 그대로).
# 속도와 obstacle 기준을 둘 다 통과한 건 base 섞기뿐(2025 47 / 6.50 s, 2026 26 / 7.40 s). 이제 base는 그대로 두고 props 단계와 race 단계를 합침:
# --base(보이는 obstacle이 통로 10 m 안 / 차 10 m 안이면 race policy) + one_world_x.sh의 world(obstacle 50 % x 상대 차 50 %, 4종 0.4~1.2배, 새 layout 세트).
# 기각(미리 정함, 시드 777): obstacle 30 layout hard(벽+obstacle) > 2025 47 / 2026 26, 또는 2025 race 20판 forzaeth 0.6 / 1.0배 hard 차 충돌 > 8 / 3
# 또는 0.6배 추월 < 15/20. solo lap은 base가 운전(확인만). 느린 차 따라가기(follow_probe 0.8배, race u50: 앞 3~6 m 33 %, 그때 속도 0.79) 함께 봄.
# 인자: init체크포인트 base체크포인트 이름 GPU 환경수 시드 업데이트 지도(25|26) 중간평가를학습과동시에(1/0)
INIT=$1; BASE=$2; NAME=$3; GPU=$4; ENVS=$5; S=$6; U=$7; MAPK=$8; PAR=$9
if [ "$MAPK" = 26 ]; then M=real:korea_2026_competition; MC=real/korea26; else M=real:korea_2025_iccas; MC=real/iccas25; fi
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/one_${NAME}.log
echo "== ${NAME}: base 섞은 채 world 하나, init $(basename $(dirname $INIT))/$(basename $INIT), base $(basename $(dirname $BASE))/$(basename $BASE)" > $L
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
    echo "  u${UU} obstacle 앞 8 layout: $(CUDA_VISIBLE_DEVICES=$GPU timeout 1800 python3 $T/console_layouts.py $R/${NAME}/state_u${UU}.pt $MC 40 8 777 $D/layouts/q_${NAME}_u${UU} 2>&1 | grep '합계' | cut -c1-200)" >> $L
    if [ "$PAR" = 1 ] && [ $UU = $((U / 2)) ]; then evalc $UU; fi
  done ) &
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --noise-corr 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.5 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 --severity-min 1.0 \
  --prop-touch-price 40 --prop-bank $R/prop_banks"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage race $PL $COMMON --updates $U \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed $S \
  --opp-present 0.5 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.2]}' \
  --init $INIT --base $BASE > $D/${NAME}.log 2>&1 || { echo fail > $D/one_${NAME}.done; exit 1; }
wait
[ "$PAR" = 1 ] || evalc $((U / 2))
evalc $U
echo done > $D/one_${NAME}.done
