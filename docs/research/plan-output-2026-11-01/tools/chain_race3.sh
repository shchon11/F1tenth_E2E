#!/bin/bash
# 상대 단계 다시 (11-03, 사용자: "상대방 대응까지도 되면 계속 파봐"): 고친 상자 단계(chain_props3.sh)에서 이어서, 같은 원칙으로 --
# 탐색 잡음은 혼자 단계와 같게(상관 0, 하한 없음; racec~raceh는 모두 상관 0.9 + 하한), 배우는 학생이 볼 수 있는 상대 차만(학생 격자 안,
# 무잡음 라이다 시선 -- 정답 상대 칸·가까운 차 기하는 비평가만), 상자 수 1..7 고르게. 나머지는 racep(순위 보상 ±80, 상자 30 %,
# 상대 4종 0.4~1.05배, 100회)와 같음. 기각(전과 같음, 같은 20판 시드 777): 0.6배속 추월 < 15/20, 추월당 세게 > 0.2, 1.0배속 세게 > 2;
# 혼자 랩이 상자 단계 선생보다 느려짐.
# 인자: 이름머리 맵 콘솔맵 혼자체크포인트 GPU 환경수 시드머리 지킴이(1/0)
TAG=$1; M=$2; MC=$3; SOLO=$4; GPU=$5; ENVS=$6; S=$7; GUARD=$8
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 25 \
  --noise-corr 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.3 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1"
if [ "$GUARD" = 1 ]; then
  bash $T/gpu_guard3.sh "state_[r]l --name ${TAG}_race|console_[r]aces.py $R/${TAG}_race" $D/chain_${TAG}_race.done > $D/gpu_guard_${TAG}_race.log 2>&1 &
fi
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_race --stage race $PL $COMMON --updates 100 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed ${S}7 \
  --opp-present 1.0 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.05]}' \
  --base $SOLO --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_race.log 2>&1 || exit 1
L=$D/races_${TAG}.log
: > $L
for U in 50 100; do
  C=$R/${TAG}_race/state_u$U.pt
  for x in "forzaeth 0.6" "forzaeth 1.0" "forzaeth_pred 0.8" "lane_switch 0.8"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$GPU timeout 1800 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/${TAG}_u${U}_$1_x$2 2>&1 | grep -v Warning | grep "합계" | sed "s/^/u$U $1 x$2 /" >> $L
  done
  echo "u$U 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $C $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $L
done
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $R/${TAG}_race/state_u100.pt $MC 40 30 777 $D/layouts/r_${TAG} 2>&1 | grep "합계\|Error" | sed "s/^/u100 상자 30배치 /" >> $L
echo done > $D/chain_${TAG}_race.done
