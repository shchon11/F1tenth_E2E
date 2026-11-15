#!/bin/bash
# (11-16) 앞이 막혀 서는 것(벽·상자 0.1-0.3 m 앞에서 섬, 후진을 거의 안 씀)이 학습 세계에서 얼마나 자주 생기고, 생겼을 때 보상이 후진을
# 가르치는지 재기 전용. kr26a(Korea 2026, 처음부터) checkpoint 그대로 같은 세계·인자·잡음으로 rollout만 저장(update 없음).
# 읽기: nose_block_scan.py (차 앞 여유 < 0.35 m에서 2 s 넘게 섬 -> 빈도, 어떻게 풀렸나, 그때 후진 쪽 잡음의 advantage 기울기).
# 인자: checkpoint update 번호 (기본 750) [이름 꼬리] [plan_rl에 덧붙일 인자...]
#   (11-16) 예: bash dump_kr26a.sh 800 s --noise-corr-stall 0.5,0.9   -> 서 있을 때 속도 잡음 잇기를 켠 rollout (반증용)
U=${1:-750}; TAIL=${2:-}; shift $(( $# < 2 ? $# : 2 ))
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
N=kr26a_dump$U$TAIL; OUT=$R/out/batchdump/$N
[ -e /home/shchon11/f1sim_runs/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
mkdir -p $OUT
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
# launch_run.sh의 BASE 그대로(두 GPU 나누기만 뺌) + 같은 세계 + 그 checkpoint에서 (잡음 잇기 ramp는 u250에 끝났으니 0.9 그대로)
CUDA_VISIBLE_DEVICES=1 bash $R/tools/slot5060.sh timeout 3600 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --seed 3 --save-every 100000 \
  --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 \
  --compile-leaves --align-episodes --noise-corr 0.9 --noise-corr-steer --noise-corr-box 1.5,4.0 --noise-corr-car 2.0,4.0 \
  --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef 1.0 --speed-std-floor 0.4 --tail-consistency 1.0 \
  --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 \
  --world /home/shchon11/F1tenth_E2E/f1sim/f1sim/learn/configs/plan_world_korea_2026_competition.json \
  --envs 384 --updates $((U + 16)) --init /home/shchon11/f1sim_runs/kr26a/plan_u$U.pt --init-critic --update-offset $U \
  --dump-rollout $OUT "$@" > $OUT/run.log 2>&1
echo "rc $?" >> $OUT/run.log
touch $OUT/done
