#!/bin/bash
# (11-15 01:00) plan26o u200 (조향 잡음 0.9로 이어짐) batch: 이어진 잡음이 통과 틈을 바꾸나, 비키기 신호가 batch에 생기나.
# 두 GPU 나누기만 뺌), update 없이 rollout만 저장. 읽기: batch_expcost.py가 지금 보상과 (가)를 더한 보상으로 advantage를 다시 계산해
# 상자 1.5-6 m 앞 결정의 "상자 반대쪽 조향 잡음" 기울기를 비교. (가)로도 0이면 (가) 버림.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
N=plan26o_dump200; OUT=$R/out/batchdump/$N
[ -e /home/shchon11/f1sim_runs/$N ] && { echo "$N: run 이름이 이미 있음"; exit 1; }
mkdir -p $OUT
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=0 timeout 3000 python3 -u -m f1sim.learn.plan_rl --name $N --format v2 --updates 12 --seed 3 --save-every 100000 \
  --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 \
  --compile-leaves --opp-every 4 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --vest-coef 1.0 \
  --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 \
  --lidar-mount-x 0.12555 --envs 384 --noise-corr 0.9 --noise-corr-steer \
  --init /home/shchon11/f1sim_runs/plan26o/plan_u200.pt --init-critic --dump-rollout $OUT > $OUT/run.log 2>&1
echo "rc $?" >> $OUT/run.log
touch $OUT/done
