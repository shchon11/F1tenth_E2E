#!/bin/bash
# 10-26: the raceline teacher (guard walls-on-plan, cap 10, MAP) with a higher lateral limit in its speed profile, same line.
# usage: verify_alat.sh GPU A_LAT...
D=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research/teacher_speed
G=$1; shift
MAPJ=$(cat $D/entry_map.json); CK=/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar_slow.pt
cd /home/shchon11/F1tenth_E2E/f1sim; mkdir -p $D/out/verify
COMMON="--teacher --action-mode plan --tracks real:korea_2026_competition --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --device cuda:0 --collision-mode soft --speed-mode wp4 --plan-follower map --speed-cap 10.0 --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92"
for A in "$@"; do for S in 81 82; do
  N=solo_alat${A}_s$S; F=$D/out/verify/$N.json; [ -f $F ] && continue
  TEACHER_ALAT=$A CUDA_VISIBLE_DEVICES=$G OUT=$D/out/verify/$N.bins.json WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 \
    /home/shchon11/F1tenth_E2E/.venv/bin/python -u $D/teacher_speed_chain.py $CK $COMMON --plan-follower-map "$MAPJ" --output $F \
    --envs 16 --steps 2400 --seed $S --race-size 1 --procedural-obstacles 0.0 > $F.log 2>&1 || { echo "[$(date +%H:%M)] $N FAILED"; exit 1; }
  echo "[$(date +%H:%M)] $N done"
done; done
echo "[$(date +%H:%M)] verify alat all done"
