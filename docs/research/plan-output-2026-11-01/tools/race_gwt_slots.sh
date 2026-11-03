#!/bin/bash
# 상대 단계 + 보이는 상대 차 칸 (11-03 20:05): srp25h3gwt_race u50(세게 104 -> 36)의 남은 세게 8번(0.6배)은 출발 2, 나란히 3, 막 앞선 직후 3
# (`car_cf.py`). 선생의 구조화된 차 정보는 통로에 겹친 시공간 칸뿐인데 통로는 0.4 m 앞부터 -- 막 앞서 뒤-옆에 있는 차(간격 -0.5~-1 m)는
# 라이다 108빔에만 있음. 학생 격자는 뒤 2 m·옆 3.5 m까지 봄. 고침: --car-slots 1 (`visible_car_slots`: 격자 안·라이다 시선이 닿는 차의
# 위치·상대 속도, 가까운 3대, 관측 끝에 붙여 --init의 가중치는 0에서 시작). 라이다 뒤 90° 사각에 든 차는 학생처럼 안 보임.
# 나머지는 race_gwt.sh와 같음(gwt u100에서, 세게 값 80, 순위 ±80, 100회). 세게 값 400(race_gwt_cp.sh)은 u50에서 기각.
# 기각(미리 정함): u50·u100 중 나은 쪽이 srp25h3gwt_race u50(세게 8/3/9/16)보다 '나란히'·'막 앞선 직후' 충돌을 절반 넘게 못 줄이거나,
# 추월당 세게 > 0.2, 1.0배 세게 > 2, 0.6배 추월 < 15/20, 혼자 랩 느려짐.
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
  --prop-touch-price 40 --car-slots 1"
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
