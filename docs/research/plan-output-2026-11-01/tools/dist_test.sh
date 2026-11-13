#!/bin/bash
# 두 GPU 학습 시험 (11-14 새벽, 사용자 "ㄱㄱㄱ"): (A) 같은 데이터면 한 GPU와 같은 update가 나오는가 (B) 다른 GPU 둘이 걸음마다 같은 가중치를
# 갖는가 (C) 실제로 얼마나 빨라지나 (지금 학습 plan26f가 4070을 쓰고 있어 크기를 줄여 잼; 4070 차례에는 gpu_guard가 plan26f를 잠시 멈춤).
# 5060의 분석 자리 둘을 잡고 돎(그동안 core_loop는 기다림). 결과 out/dist_test/
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/dist_test
L=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/.slot5060
exec 8>$L.0; flock 8; exec 9>$L.1; flock 9
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
COMMON="--format v2 --seed 1 --lr 3e-4 --lr-end 1e-4 --memory 128 --mem-front --mem-value 0.15 --mem-keep-s 30 --speed-std-floor 0.4 --tail-consistency 1.0 --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25"
SMALL="--updates 2 --save-every 2 --envs 64 --horizon 32 --mem-chunk 32 --minibatches 4 --accum-actor 2"
PY="python3 -u -W ignore -m f1sim.learn.plan_rl"
echo "== A $(date +%T)"
CUDA_VISIBLE_DEVICES=1 $PY --name dtest_single $COMMON $SMALL > $O/A_single.log 2>&1; echo "single rc $?"
CUDA_VISIBLE_DEVICES=1 $PY --name dtest_single2 $COMMON $SMALL > $O/A_single2.log 2>&1; echo "single2 rc $?"
DA="--dist-world 2 --dist-same-seed --dist-init tcp://127.0.0.1:29541"
CUDA_VISIBLE_DEVICES=1 $PY --name dtest_same $COMMON $SMALL $DA --dist-rank 1 > $O/A_same_r1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY --name dtest_same $COMMON $SMALL $DA --dist-rank 0 > $O/A_same_r0.log 2>&1; echo "same r0 rc $?"; wait; echo "same r1 done"
echo "== B $(date +%T)"
SMALLB="--updates 3 --save-every 3 --envs 128 --horizon 32 --mem-chunk 32 --minibatches 4 --accum-actor-auto --accum-auto-frac 0.5"
DB="--dist-world 2 --dist-init tcp://127.0.0.1:29542"
CUDA_VISIBLE_DEVICES=1 $PY --name dtest_x $COMMON $SMALLB $DB --dist-rank 1 > $O/B_r1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY --name dtest_x $COMMON $SMALLB $DB --dist-rank 0 > $O/B_r0.log 2>&1; echo "B r0 rc $?"; wait; echo "B r1 done"
echo "== C $(date +%T)"
MID="--updates 3 --save-every 100 --horizon 128 --mem-chunk 128 --minibatches 16 --accum-actor-auto --accum-auto-frac 0.5"
CUDA_VISIBLE_DEVICES=0 $PY --name dtest_c1 $COMMON $MID --envs 512 > $O/C_single.log 2>&1; echo "C single rc $?"
DC="--dist-world 2 --dist-init tcp://127.0.0.1:29543"
CUDA_VISIBLE_DEVICES=1 $PY --name dtest_c2 $COMMON $MID --envs 320 $DC --dist-rank 1 > $O/C_r1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY --name dtest_c2 $COMMON $MID --envs 512 $DC --dist-rank 0 > $O/C_r0.log 2>&1; echo "C r0 rc $?"; wait; echo "C r1 done"
echo "== 끝 $(date +%T)"
touch $O/done
