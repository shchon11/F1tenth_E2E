#!/bin/bash
# 두 GPU 속도 시험 (11-14): plan26f가 4070 메모리 8 GB를 잡고 있어 4070 쪽 차 256대로 줄임. 한 GPU(4070 256대) vs 두 GPU(4070 256 + 5060 160),
# rollout 128 결정, 3 update씩. 5060 분석 자리 둘을 잡고 돎.
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/dist_test
L=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/.slot5060
exec 8>$L.0; flock 8; exec 9>$L.1; flock 9
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
COMMON="--format v2 --seed 1 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25"
MID="--updates 3 --save-every 100 --horizon 128 --mem-chunk 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5"
PY="python3 -u -W ignore -m f1sim.learn.plan_rl"
echo "== C $(date +%T)"
CUDA_VISIBLE_DEVICES=0 $PY --name dtest_c1 $COMMON $MID --envs 256 > $O/C_single.log 2>&1; echo "C single rc $?"
DC="--dist-world 2 --dist-init tcp://127.0.0.1:29544"
CUDA_VISIBLE_DEVICES=1 $PY --name dtest_c2 $COMMON $MID --envs 160 $DC --dist-rank 1 > $O/C_r1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY --name dtest_c2 $COMMON $MID --envs 256 $DC --dist-rank 0 > $O/C_r0.log 2>&1; echo "C r0 rc $?"; wait; echo "C r1 done"
echo "== 끝 $(date +%T)"
touch $O/done_c
