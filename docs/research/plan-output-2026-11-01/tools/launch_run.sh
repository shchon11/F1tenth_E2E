#!/bin/bash
# 어느 지도든 같은 방식으로 학습을 띄우는 실행기 (11-15, 사용자: "korea 2025 맵에 대해서만 뭘 자꾸 만들어지고 있어서.. 효율화").
# 지금까지는 run마다 launch_plan26?.sh를 베껴 Korea 2025 경로를 박아 넣었음 -> 지도는 세계 설정 하나로만 정함:
#   - 학습: plan_rl --world <세계 json>  (트랙·상자 배치·접촉 무게 contact_scale이 그 안에 있음; map_setup.sh / new_map.sh가 만듦)
#   - 시험·평가·영상: checkpoint에 적힌 세계를 따라감(world_of.py) -> core_loop.sh가 지도와 상관없이 그대로 돎
# 학습 방식은 지금까지 가장 좋은 policy(plan26s -> avg26s u900)를 만든 인자 그대로. 처음부터면 조향 잡음 잇기를 u200부터 켬
# (처음부터 켜면 서 버리는 쪽으로 무너짐: plan26m 11-14; plan26r은 u200 fork에서 켬).
# 두 GPU(4070 rank0 608대 + 5060 rank1 416대), GPU 지킴이, 핵심 평가 반복(core_loop), 콘솔 watch 폴더까지.
#
# 인자: 이름 세계(지도 이름 또는 json 경로) [끝 update 1000] [이어 받을 checkpoint] [그 checkpoint의 update 번호]
#       그 뒤 인자는 plan_rl에 그대로 덧붙임(같은 인자는 뒤에 온 것이 이김)
# 예: bash tools/launch_run.sh k26a korea_2026_competition                       # 처음부터
#     bash tools/launch_run.sh k26b korea_2026_competition 1300 ~/f1sim_runs/avg26s/plan_u900.pt 900   # Korea 2025 최고에서 이어
set -e
N=$1; W=$2; U1=${3:-1000}; INIT=${4:-}; OFF=${5:-0}
shift $(( $# < 5 ? $# : 5 ))
[ -n "$N" ] && [ -n "$W" ] || { sed -n 2,14p "$0"; exit 1; }
RT=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
D=$RT/out; R=/home/shchon11/f1sim_runs; CFG=/home/shchon11/F1tenth_E2E/f1sim/f1sim/learn/configs
case "$W" in */*|*.json) WJ=$(readlink -f "$W") ;; korea25|korea_2025_iccas) WJ=$CFG/plan_world_korea25.json ;; *) WJ=$CFG/plan_world_$W.json ;; esac
[ -e "$WJ" ] || { echo "세계 설정 없음: $WJ (map_setup.sh / new_map.sh 먼저)"; exit 1; }
[ -e $R/$N ] && { echo "$N: run 이름이 이미 있음 (11-05 이름 겹침 사고)"; exit 1; }
MAPN=$(python3 -c "import json; print(json.load(open('$WJ'))['tracks'])")
CS=$(python3 -c "import json; print(json.load(open('$WJ')).get('contact_scale', 1.0))")
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])")
BASE="--format v2 --seed 1 --save-every 25 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --noise-probe-every 4 --accum-ratchet --compile-leaves --align-episodes --noise-corr 0.9 --noise-corr-steer --noise-corr-box 1.5,4.0 --noise-corr-car 2.0,4.0 --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef 1.0 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555"
if [ -n "$INIT" ]; then
  START="--init $INIT --init-critic --update-offset $OFF"; U0=$(( OFF + 25 ))
else
  START="--noise-corr-ramp 200,250"; U0=25
fi
ARGS="$BASE --world $WJ --updates $U1 $START $*"
echo "$(date +%T) $N | 세계 $WJ ($MAPN, 접촉 무게 $CS) | ${INIT:-처음부터} | 끝 u$U1 | 덧붙임: $*" | tee -a $D/switch_$N.log
cd $RT
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
F=/home/shchon11/F1tenth_E2E/f1sim
# LAUNCH_GPU=both (기본: 4070 608대 + 5060 416대) | 0 (4070 혼자) | 1 (5060 혼자); 혼자일 때 차 수 LAUNCH_ENVS (기본 512)
# -- 두 run을 나란히 비교할 때(11-15 Korea 2026 이어 받기 vs 처음부터) 한 GPU씩, 같은 차 수로
G=${LAUNCH_GPU:-both}
if [ "$G" = both ]; then
  setsid nohup bash -c "cd $F && CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.plan_rl --name $N $ARGS --envs 416 --dist-world 2 --dist-init tcp://127.0.0.1:$PORT --dist-rank 1 > $D/${N}_rank1.log 2>&1" > /dev/null 2>&1 &
  setsid nohup bash -c "cd $F && CUDA_VISIBLE_DEVICES=0 python3 -u -m f1sim.learn.plan_rl --name $N $ARGS --envs 608 --dist-world 2 --dist-init tcp://127.0.0.1:$PORT --dist-rank 0 > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
else
  setsid nohup bash -c "cd $F && CUDA_VISIBLE_DEVICES=$G python3 -u -m f1sim.learn.plan_rl --name $N $ARGS --envs ${LAUNCH_ENVS:-512} > $D/$N.log 2>&1; echo done > $D/one_$N.done" > /dev/null 2>&1 &
fi
sleep 5
# the guard pauses a 4070 run while anything else uses the 4070 (`gpu_guard3.sh`)
if [ "$G" != 1 ]; then      # (a background `[ ] || cmd &` list kept a subshell holding this script's stdout open)
  setsid nohup bash tools/gpu_guard3.sh "plan_[r]l --name $N " out/guard4070_$N.done > out/guard4070_$N.log 2>&1 < /dev/null &
fi
CORE_VAL_ENVS=512 setsid nohup bash tools/core_loop.sh $N $U0 25 $U1 > out/core_loop_$N.out 2>&1 < /dev/null &
mkdir -p $R/watch_$N
setsid nohup bash tools/watch_latest_plan.sh watch_$N $N $D/one_$N.done > out/watch_$N.log 2>&1 < /dev/null &
sleep 2
ps -eo pid,cmd | grep -E "plan_rl --name $N |core_loop.sh $N|gpu_guard3.sh .*$N" | grep -v grep | cut -c1-120
