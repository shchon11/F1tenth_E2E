#!/bin/bash
# one-world를 빠른 solo policy에서 시작 (11-04 11:50). one-world 첫 시도(srp25ow*/srp26ow*, props/race checkpoint 시작)는 solo lap이
# 2025 7.7~7.8 s, 2026 9.4~9.6 s로 base policy(6.50 / 7.40 s)보다 느려 기각. 시작점을 base 없이 달려 보니 2025 race u50 8.30 s, props u100
# 8.25 s, 2026 props u100 12.6 s: one-world는 오히려 빨라졌고(8.3 -> 7.7, 12.6 -> 9.4), 느린 이유는 시작점. base를 섞은 props/race 단계는
# 빈 구간을 base가 달려서 그 policy 자체는 빈 트랙 속도를 배운 적이 없음. 이제 빈 트랙이 빠른 solo policy(srp25h2_solo2 / srp26h2_solo2 u300)에서
# 시작: obstacle·상대 차 입력은 가중치 0(첫 update는 solo policy와 같은 운전), world는 one_world_x.sh 그대로(obstacle 50 % x 상대 차 50 %,
# 0.4~1.2배, base 없음), white noise(correlated 0.9 단계는 두 지도 모두 나빠짐).
# 기각(미리 정함, 시드 777): u100/u200 solo lap > base + 0.15 s(2025 6.65, 2026 7.55), 또는 obstacle 30 layout hard(벽+obstacle) > base 섞은
# props 최고(2025 47, 2026 26), 또는 2025 race 20판 forzaeth 0.6 / 1.0배 hard 차 충돌 > 8 / 3 또는 0.6배 추월 < 15/20.
# 평가: obstacle 30 layout, solo lap, race forzaeth 0.6 / 1.0 / 1.2배(obstacle 없음), forzaeth 0.6 / 1.0배 + obstacle. u50마다 solo lap 먼저(속도 조기 확인).
# 인자: 시작체크포인트 이름 GPU 환경수 시드 업데이트 지도(25|26)
INIT=$1; NAME=$2; GPU=$3; ENVS=$4; S=$5; U=$6; MAPK=$7
if [ "$MAPK" = 26 ]; then M=real:korea_2026_competition; MC=real/korea26; else M=real:korea_2025_iccas; MC=real/iccas25; fi
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/one_${NAME}.log
echo "== ${NAME}: one-world, 시작 $(basename $(dirname $INIT))/$(basename $INIT)" > $L
# u50마다 solo lap (학습과 같은 GPU, 30 s 정도)
( for UU in $(seq 50 50 $U); do
    until [ -e $R/${NAME}/state_u${UU}.pt ] || [ -e $D/one_${NAME}.done ]; do sleep 20; done
    [ -e $R/${NAME}/state_u${UU}.pt ] || break
    echo "  u${UU} solo: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${NAME}/state_u${UU}.pt $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $L
  done ) &
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --noise-corr 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.5 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 --severity-min 1.0 \
  --prop-touch-price 40 --prop-bank $R/prop_banks"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage race $PL $COMMON --updates $U \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed $S \
  --opp-present 0.5 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.2]}' \
  --init $INIT > $D/${NAME}.log 2>&1 || { echo fail > $D/one_${NAME}.done; exit 1; }
wait
for UU in $((U / 2)) $U; do
  C=$R/${NAME}/state_u${UU}.pt
  CUDA_VISIBLE_DEVICES=$GPU timeout 5400 python3 $T/console_layouts.py $C $MC 40 30 777 $D/layouts/o_${NAME}_u${UU} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_o_${NAME}_u${UU}.log
  echo "  u${UU} obstacle 30 layout: $(tail -n 1 $D/layouts_o_${NAME}_u${UU}.log)" >> $L
  for x in "forzaeth 0.6" "forzaeth 1.0" "forzaeth 1.2"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/${NAME}_u${UU}_$1_x$2 2>&1 | grep -v Warning | grep "합계" | sed "s/^/  u$UU $1 x$2 /" >> $L
  done
  for x in "forzaeth 0.6" "forzaeth 1.0"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$GPU timeout 5400 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/obs_${NAME}_u${UU}_$1_x$2 1 2>&1 | grep -v Warning | grep "합계" | sed "s/^/  u$UU obstacle+$1 x$2 /" >> $L
  done
done
echo done > $D/one_${NAME}.done
