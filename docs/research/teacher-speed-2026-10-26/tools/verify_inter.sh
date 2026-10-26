#!/bin/bash
# 10-26: the INTERACTIVE teacher (offsets -0.6..0.6 x speeds, walls_within_plan for its line candidates) as the ego in Korea
# races, the same protocol as verify_guard.sh race (32 envs, 2-car race, props on, 4 kinds) -- can it pass where the raceline
# teacher's guard only follows? usage: SEEDS="81 82" verify_inter.sh GPU
D=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research/teacher_speed
G=$1
MAPJ=$(cat $D/entry_map.json); CK=/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar_slow.pt
cd /home/shchon11/F1tenth_E2E/f1sim; mkdir -p $D/out/verify
COMMON="--teacher --teacher-kind interactive --action-mode plan --tracks real:korea_2026_competition --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --device cuda:0 --collision-mode soft --speed-mode wp4 --plan-follower map --speed-cap 10.0 --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92"
for S in ${SEEDS:-81}; do for K in forzaeth forzaeth_pred lane_switch interactive; do
  N=race_inter_s${S}_$K; F=$D/out/verify/$N.json; [ -f $F ] && continue
  CUDA_VISIBLE_DEVICES=$G OUT=$D/out/verify/$N.bins.json WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 \
    /home/shchon11/F1tenth_E2E/.venv/bin/python -u $D/teacher_speed_chain.py $CK $COMMON --plan-follower-map "$MAPJ" --output $F \
    --envs 32 --steps 2400 --seed $S --race-size 2 --graphs --movable-obstacles --spawn-runway 3.0 \
    --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 \
    --opp-slots "[{\"kind\": \"$K\", \"speed_scale\": [0.7, 1.0]}]" > $F.log 2>&1 || { echo "[$(date +%H:%M)] $N FAILED"; exit 1; }
  echo "[$(date +%H:%M)] $N done"
done; done
echo "[$(date +%H:%M)] verify inter all done"
