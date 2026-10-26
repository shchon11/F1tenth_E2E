#!/bin/bash
# 5060 (10-26 14:50): THE REAL CAR'S SPEED LOOP (competition recordings: drive-side cap 2.5 m/s^2, tau 0.7 s, +0.1 s delay;
# realspeed_eval.py). Everything so far used the fast default. Props on, races 81-82 x 4 kinds + solo 81-82: the entry without
# limits vs with both limits. Questions: how much slower is the car, how many more hard hits, do the limits (which assume 0.2 s of
# lag) still help? Falsifier of 'the limits carry over to the real speed loop': hard contacts with the limits not below without.
R=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research
CK=/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar_slow.pt
cd /home/shchon11/F1tenth_E2E/f1sim
COMMON="--action-mode plan --tracks real:korea_2026_competition --envs 32 --steps 2400 --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --speed-cap 9.0 --device cuda:0 --graphs --collision-mode soft --movable-obstacles --spawn-runway 3.0 --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 --procedural-raceline-corridor off --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92"
for ROW in nolim both; do
  O=$R/out/realspeed_$ROW; mkdir -p $O
  EXTRA=""; [ $ROW = both ] && EXTRA="--opponent-speed-limit --obstacle-speed-limit"
  for S in 81 82; do for K in forzaeth forzaeth_pred lane_switch interactive; do
    F=$O/race_s${S}_${K}.json; [ -f $F ] && continue
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u $R/realspeed_eval.py $CK $COMMON \
      --seed $S --race-size 2 --opp-slots "[{\"kind\": \"$K\", \"speed_scale\": [0.7, 1.0]}]" --output $F $EXTRA > $F.log 2>&1 || { echo "[$(date +%H:%M)] $ROW $S $K FAILED"; exit 1; }
  done; done
  for S in 81 82; do
    F=$O/solo_s${S}.json; [ -f $F ] && continue
    CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u $R/realspeed_eval.py $CK $COMMON \
      --seed $S --race-size 1 --output $F $EXTRA > $F.log 2>&1 || { echo "[$(date +%H:%M)] $ROW solo $S FAILED"; exit 1; }
  done
  echo "[$(date +%H:%M)] real speed loop $ROW done"
done
