#!/bin/bash
# policy 하나, world 하나 (11-04 02:25). 사용자: "base policy와 props 를 별개로 학습하는 이유가 있나? ... 학습 환경에 중간중간 뭐 반반
# 비율로 장애물 아예 없는 상황도 넣어놓던가" + "그 상대차 있는 경우도 별개로 나누는 이유를 모르겠네".
# 지금까지: solo policy -> props 단계 -> race 단계를 따로, 그리고 base policy(장애물을 못 보는 frozen solo policy)가 빈 구간을 운전(gating).
# 대가: props policy는 빈 구간을 학습한 적 없음, base policy는 obstacle을 못 봄, 넘겨받는 순간 상태 불일치(2026 벽 긁기), race 단계에서
# props 실력이 흐려짐. 이제: --stage race 하나에 obstacle 50 %(새 layout 세트, 1..전부, removal 10~30 s 50 %) x 상대 차 50 %
# (4종 0.4~1.2배, 11-04 사용자: "최대를 1배가 아니라 한 1.2배까지 해도 될듯. 지금 너무 다들 안정적임") = 빈 트랙 / obstacle만 / 차만 / 둘 다 각 25 %, --base 없음. reward는 race 단계 그대로(순위 ±80, 미는 시간 40).
# 기각(미리 정함, 시드 777): obstacle 30 layout에서 hard(벽+obstacle) > 2025 재학습 props checkpoint u100(base 섞기) 값, 또는
# solo lap > base policy 6.50 + 0.15 s, 또는 race 20판(forzaeth 0.6배 / 1.0배)에서 hard 차 충돌 > race checkpoint u50(8 / 3) 또는 0.6배 추월 < 15/20.
# one_world_x26.sh = one_world_x.sh의 한국 2026판 (11-04 03:20). one_world_x.sh (11-04 03:00): one_world.sh + 평가 여부(1/0) + --noise-corr 값. 2025 연장(white) 200 update에 hard 78 -> 47, 2026 correlated 0.9는 50 update에 45 -> 33.
# 인자: 시작체크포인트 이름 GPU 환경수 시드 업데이트 평가(1/0) noise-corr [기다릴파일들...]
INIT=$1; NAME=$2; GPU=$3; ENVS=$4; S=$5; U=$6; EVAL=$7; NOISE=$8; shift 8
for f in "$@"; do until [ -e "$f" ]; do sleep 30; done; done
M=real:korea_2026_competition; MC=real/korea26
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --noise-corr $NOISE --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.5 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 --severity-min 1.0 \
  --prop-touch-price 40 --prop-bank $R/prop_banks"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage race $PL $COMMON --updates $U \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed $S \
  --opp-present 0.5 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.2]}' \
  --init $INIT > $D/${NAME}.log 2>&1 || { echo fail > $D/one_${NAME}.done; exit 1; }
if [ "$EVAL" != 1 ]; then echo done > $D/one_${NAME}.done; exit 0; fi
L=$D/one_${NAME}.log
echo "== ${NAME}: policy 하나 world 하나 (시작 $(basename $(dirname $INIT))/$(basename $INIT))" > $L
for UU in $((U / 2)) $U; do
  C=$R/${NAME}/state_u${UU}.pt
  CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $C $MC 40 30 777 $D/layouts/o_${NAME}_u${UU} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_o_${NAME}_u${UU}.log
  echo "  u${UU} obstacle 30 layout: $(tail -n 1 $D/layouts_o_${NAME}_u${UU}.log)" >> $L
  echo "  u${UU} solo: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $C $MC 60 2>&1 | grep '주행' | cut -c1-200)" >> $L
  for x in "forzaeth 0.6" "forzaeth 1.0"; do
    set -- $x
    CUDA_VISIBLE_DEVICES=$GPU timeout 1800 python3 $T/console_races.py $C $MC 30 20 $1 $2 777 $D/races/${NAME}_u${UU}_$1_x$2 2>&1 | grep -v Warning | grep "합계" | sed "s/^/  u$UU $1 x$2 /" >> $L
  done
done
echo done > $D/one_${NAME}.done
