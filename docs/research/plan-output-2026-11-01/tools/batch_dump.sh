#!/bin/bash
# 학습 batch 측정 (11-13 밤, 구조 점검; 학습 아님: update를 한 번도 안 함).
# 질문: plan25y u150이 학습 때 실제로 받는 batch 안에, "상자 앞에서 첫 0.1 s 조향을 상자 반대쪽으로 더 튼 결정이 advantage가 높다"는
#       신호가 들어 있는가. 들어 있다면 한 update에서 얼마나 또렷한가(기울기 / 표준오차).
# 근거: 복제 실행(nudge_value.py)에서는 상자 1.5-6 m 앞에서 2도 x 0.3 s 틀면 +0.24~0.36 s, 0.1 s 한 번만 틀어도(blip) 방향이 같음.
#       그런데 반응 거리는 update를 거듭해도 2.5 m 근처에서 느리게만 늘어남.
# 읽기: batch에도 또렷하면(여러 update에 걸쳐 같은 부호) -> 신호는 오는데 update가 못 따라감(update 쪽을 봄).
#       batch에는 없으면 -> advantage 계산(critic + GAE)이나 학습 세계가 그 신호를 지움(그쪽을 봄).
# 방식: plan_rl --init CHECKPOINT --init-critic --dump-rollout: plan25y와 같은 세계·인자·잡음으로 rollout만 하고 저장. 앞의 3 update(출발 직후)는 안 씀.
# (--mem-keep-s는 뺌: 기억의 처음 값일 뿐이고 checkpoint의 기억을 그대로 읽어 옴)
# 21:50 결과: 신호가 batch에 없음(2도당 0 +-0.02 s). 결정 하나의 잡음은 차를 옆으로 3 cm, 0.5 s 옮길 뿐(닿는 건 10-20 cm 모자라서).
# 둘째 질문(다섯째 인자): 잡음의 크기는 그대로 두고 여러 결정 동안 이어지게 뽑으면(--noise-corr, 결정마다 남는 비율) 그 신호가 batch에 나타나는가.
#       나타나면(1.5-6 m에서 +, 여러 update 같은 부호) -> 탐색 잡음을 시간으로 잇는 것이 구조 변경 후보. 안 나타나면 이 후보도 버림.
# 쓰는 법: bash batch_dump.sh GPU번호 RUN UPDATE [update 수] [잡음이 결정마다 남는 비율, 기본 0]
gpu=$1; run=$2; upd=$3; n=${4:-10}; corr=${5:-0}
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
N=${run}_dump$upd
[ "$corr" != 0 ] && N=${N}_c${corr#0.}
OUT=$R/out/batchdump/$N
[ -e /home/shchon11/f1sim_runs/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
mkdir -p $OUT
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
export F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=$gpu python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates $n --seed 1 --save-every 100000 \
  --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 \
  --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 \
  --lidar-mount-x 0.12555 --track-scales korea25 \
  --init /home/shchon11/f1sim_runs/$run/plan_u$upd.pt --init-critic --noise-corr $corr --dump-rollout $OUT > $OUT/run.log 2>&1
echo "rc $?" >> $OUT/run.log
touch $OUT/done
