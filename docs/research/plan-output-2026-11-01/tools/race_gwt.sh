#!/bin/bash
# 상대 단계를 상자 단계가 풀린 선생에서 (11-03 15:00, 사용자: "상대방 대응까지도 되면 계속 파봐"): srp25h3_race(h3 상자 u200에서)는
# 느린 차 옆을 첫 머리핀에서 비집다 부딪히고 1.0배는 따라가다 뒤에서 받음 -- 상자 앞에서 선을 안 바꾸던 것과 같은 꼴. 상자 단계는 이제
# 풀림: gwt u100(라인 위 상자 + 독립 잡음 + 상자 미는 시간 값 40) -- 상자 30배치 7.75 s, 세게 4, 멈춤 없음, 상자 하나 0/12.
# 같은 상대 단계(chain_race3.sh와 같은 인자: 순위 보상 ±80, 상자 30 %, 상대 4종 0.4~1.05배, 독립 잡음, 100회) + 상자 미는 시간 값 40(상자 단계와 같게).
# 시작점 그대로(u0)의 경주도 같이 잼.
# 기각(전과 같음, 같은 20판 시드 777): 0.6배속 추월 < 15/20, 추월당 세게 > 0.2, 1.0배속 세게 > 2; 혼자 랩이 gwt보다 느려짐;
# 상자 30배치에서 멈춤(1 s 넘게) 하나라도 또는 세게 > 15.
# 인자: 시작체크포인트 이름 GPU 환경수 시드 지킴이(1/0) 혼자체크포인트
INIT=$1; NAME=$2; GPU=$3; ENVS=$4; S=$5; GUARD=$6; SOLO=$7
M=real:korea_2025_iccas; MC=real/iccas25
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 25 \
  --noise-corr 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.3 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 --severity-min 1.0 \
  --prop-touch-price 40"
if [ "$GUARD" = 1 ]; then
  bash $T/gpu_guard3.sh "state_[r]l --name ${NAME}|console_[r]aces.py $R/${NAME}|console_[l]a.*${NAME}" $D/race_${NAME}.done > $D/gpu_guard_${NAME}.log 2>&1 &
fi
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage race $PL $COMMON --updates 100 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed $S \
  --opp-present 1.0 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.05]}' \
  --base $SOLO --init $INIT > $D/${NAME}.log 2>&1 || { echo fail > $D/race_${NAME}.done; exit 1; }
L=$D/races_${NAME}.log
: > $L
for U in 50 100; do
  C=$R/${NAME}/state_u$U.pt
  for x in "forzaeth 0.6" "forzaeth 1.0" "forzaeth_pred 0.8" "lane_switch 0.8"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$GPU timeout 1800 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/${NAME}_u${U}_$1_x$2 2>&1 | grep -v Warning | grep "합계" | sed "s/^/u$U $1 x$2 /" >> $L
  done
  echo "u$U 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $C $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $L
done
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $R/${NAME}/state_u100.pt $MC 40 30 777 $D/layouts/r_${NAME} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_r_${NAME}.log
echo "u100 상자 30배치 $(tail -n 1 $D/layouts_r_${NAME}.log)" >> $L
echo done > $D/race_${NAME}.done
