#!/bin/bash
# 두 GPU 실제 크기 속도 (11-14 08:50, 4070이 빈 틈): 다음 학습 설정 그대로(처음부터), 3 update씩
#   한 GPU: 4070 차 1024대 / 두 GPU: 4070 1024대 + 5060 640대
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/dist_test
L=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/.slot5060
exec 8>$L.0; flock 8; exec 9>$L.1; flock 9
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
A="--format v2 --updates 3 --seed 1 --save-every 100 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --horizon 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5 --speed-std-floor 0.4 --tail-consistency 1.0 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25"
PY="python3 -u -W ignore -m f1sim.learn.plan_rl"
echo "== 한 GPU $(date +%T)"
CUDA_VISIBLE_DEVICES=0 $PY --name dspeed_1 $A --envs 1024 > $O/F_single.log 2>&1; echo "single rc $?"
echo "== 두 GPU $(date +%T)"
D="--dist-world 2 --dist-init tcp://127.0.0.1:29551"
CUDA_VISIBLE_DEVICES=1 $PY --name dspeed_2 $A --envs 640 $D --dist-rank 1 > $O/F_r1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY --name dspeed_2 $A --envs 1024 $D --dist-rank 0 > $O/F_r0.log 2>&1; echo "dual r0 rc $?"; wait
echo "== 끝 $(date +%T)"
touch $O/done_f
