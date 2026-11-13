#!/bin/bash
# plan 뒷부분을 실제로 간 길로 학습(사용자 11-14: 2) -- 실제 크기 확인 (측정용, 이 run을 쓰려는 것 아님): plan26f u75에서 8 update,
# 차 512대, 5060. 볼 것: 실제로 간 길과 0.5 s / 1.0 s 차이가 update마다 줄어드는가, 몸통 gradient 비율이 0.5 근처로 맞춰지는가,
# 바퀴·속도·세게 닿음이 무너지지 않는가. 5060 분석 자리 둘을 잡고 돎.
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/tail_check
L=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/.slot5060
exec 8>$L.0; flock 8; exec 9>$L.1; flock 9
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=1 python3 -u -W ignore -m f1sim.learn.plan_rl --name plan26f_tailchk75 --format v2 --updates 8 --seed 1 --save-every 8 --envs 512 \
  --lr 2.7e-4 --lr-end 2.7e-4 --memory 128 --mem-chunk 128 --mem-front --mem-value 0.15 --horizon 128 --minibatches 16 \
  --accum-actor-auto --accum-auto-frac 0.5 --speed-std-floor 0.4 --tail-consistency 1.0 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5 \
  --plan-hit-share 0 --box-range 12 --plan-lat-margin 0.95 --opp-restart-s 2 --lidar-mount-x 0.12555 --track-scales korea25 \
  --init /home/shchon11/f1sim_runs/plan26f/plan_u75.pt --init-critic > $O/run.log 2>&1
echo "rc $?" >> $O/run.log
touch $O/done
