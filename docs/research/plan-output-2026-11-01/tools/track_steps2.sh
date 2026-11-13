#!/bin/bash
# 실제 update가 측정한 방향으로 얼마나 가나 (11-14, 구조 점검; 학습 결과를 쓰려는 run이 아님).
# 질문: u150 actor를 학습 없이 rollout 7개의 평균 gradient 방향으로 0.1 옮기면 frozen exam에서 빈 트랙 +20 m / 90 s (무작위 방향 0.1은
#       변화 없음). 그런데 저장된 checkpoint들은 그 방향으로 25 update에 약 0.02만 감(같은 동안 가중치 전체는 5.9 움직임).
#       update 안의 어디에서 그 방향이 사라지나: optimizer 걸음(Adam, update마다 4 epoch x 16 minibatch)마다 actor가 그 방향으로 간 거리,
#       자른 gradient가 가리킨 거리, 첫 epoch에서 손실 항(policy / 속도 짐작 / 기억 값 / plan 뒷부분 일관성)마다 그 방향을 민 거리.
# 읽기: gradient는 그 방향을 가리키는데 걸음이 안 가면 -> optimizer(Adam의 가중치별 크기 맞춤, 걸음 수, 잘림)가 신호를 지움.
#       policy 항은 가리키는데 다른 항이 반대로 밀면 -> 그 항이 학습을 붙잡음. 둘 다 아니면 -> update마다 신호 자체가 작음.
# 쓰는 법: bash track_steps2.sh GPU 이름 [잡음 이음 비율, 기본 0] [update 수, 기본 8] [한 걸음에 합칠 minibatch 수, 기본 1] [속도 짐작 손실 무게, 기본 1.0]
# (track_steps.sh + 속도 짐작 손실 무게: 11-14 00:50, 몸통 가중치에서 그 gradient가 policy의 ~18배 -> Adam 걸음 크기를 그것이 정함)
#   설정은 plan25y 그대로(학습률은 u150 시점 값 2.7e-4 고정), u150에서 출발, critic도 이어 받음. Adam 상태는 새로 시작(checkpoint에 없음).
gpu=$1; name=$2; corr=${3:-0}; n=${4:-8}; acc=${5:-1}; vest=${6:-1.0}
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
[ -e /home/shchon11/f1sim_runs/$name ] && { echo "$name: run 이름이 이미 있음"; exit 1; }
OUT=$R/out/track/$name; mkdir -p $OUT
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=$gpu python3 -u -m f1sim.learn.plan_rl --name $name --format v2 --updates $n --seed 1 --save-every $n \
  --lr 2.7e-4 --lr-end 2.7e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 \
  --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 \
  --lidar-mount-x 0.12555 --track-scales korea25 \
  --init /home/shchon11/f1sim_runs/plan25y/plan_u150.pt --init-critic --noise-corr $corr --accum $acc --vest-coef $vest \
  --track-steps $R/out/netaudit/track_dirs_plan25y_u150.npz > $OUT/run.log 2>&1
echo "rc $?" >> $OUT/run.log
touch $OUT/done
