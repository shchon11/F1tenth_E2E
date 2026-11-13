#!/bin/bash
# 학습 batch + 그 batch의 policy gradient 측정 (11-13 밤, 구조 점검; 학습 아님: update를 한 번도 안 함).
# 질문: rollout 하나가 actor를 미는 방향(policy gradient)이 rollout마다 얼마나 같은가. 서로 다른 rollout 둘의 gradient 사이
#       cosine = 한 rollout의 gradient 가운데 신호인 몫. 탐색 잡음을 결정마다 새로 뽑을 때와 여러 결정 동안 이을 때(--noise-corr)를 견줌.
# 근거: batch_signal.py / batch_cells.py (22:15): 잡음을 결정마다 새로 뽑으면 상자 앞 결정들에 '틀 방향'이 없음(묶음들이 0과 다른 정도 0.93 = 잡음뿐),
#       0.9로 이으면 생김(6.2; 상자 없는 결정은 1.4 -> 111). 이것이 actor의 가중치를 미는 방향에서도 그런지 직접 잼.
# 읽기: 이은 쪽의 cosine이 뚜렷이 높으면 -> 탐색 잡음을 시간으로 잇는 것이 학습 신호를 키움(처음부터 학습으로 확인할 근거).
#       같거나 낮으면 -> batch의 방향 신호가 가중치까지는 안 옴(이 후보 버림).
# 방식: batch_dump.sh와 같음 + rollout마다 g<N>.npz (조향 평균 / 속도 평균 / 잡음 크기 쪽 gradient를 따로). 앞의 3 update는 안 씀.
# 쓰는 법: bash grad_dump.sh GPU번호 RUN UPDATE [update 수] [잡음이 결정마다 남는 비율, 기본 0] [minibatch 수, 기본 16]
#   (update를 안 하므로 minibatch 수는 결과와 무관: gradient는 rollout 전체의 합. 5060은 16이면 메모리가 모자라 64로)
gpu=$1; run=$2; upd=$3; n=${4:-10}; corr=${5:-0}; mb=${6:-16}
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
N=${run}_gdump$upd
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
  --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches $mb \
  --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 \
  --lidar-mount-x 0.12555 --track-scales korea25 \
  --init /home/shchon11/f1sim_runs/$run/plan_u$upd.pt --init-critic --noise-corr $corr --dump-rollout $OUT > $OUT/run.log 2>&1
echo "rc $?" >> $OUT/run.log
touch $OUT/done
