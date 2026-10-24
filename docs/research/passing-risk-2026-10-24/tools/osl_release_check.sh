#!/bin/bash
# GPU CORRIDOR TAG (10-25): the SHIPPED path -- `evaluate --opponent-speed-limit` (EnvConfig.opponent_speed_limit, fed by
# common.student_policy) on the archived checkpoint, fresh seeds 91-92, the finals race cells -- against the research hook's
# numbers for the same cells (props on: hard car 0.49, total 1.63; props off: 0.54 / 1.51)
GPU=$1; CORR=$2; TAG=$3
OUT=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research/out/release_$TAG
mkdir -p $OUT
cd /home/shchon11/F1tenth_E2E/f1sim
for S in 91 92; do for K in forzaeth forzaeth_pred lane_switch interactive; do
  F=$OUT/race_s${S}_${K}.json
  [ -f $F ] && continue
  CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u -m f1sim.learn.evaluate \
    /home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar.pt --action-mode plan --tracks real:korea_2026_competition \
    --envs 32 --steps 2400 --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time \
    --dial-offset 0.0 --speed-cap 9.0 --device cuda:0 --seed $S --graphs --collision-mode soft --movable-obstacles --spawn-runway 3.0 \
    --race-size 2 --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 --procedural-raceline-corridor $CORR \
    --opp-slots "[{\"kind\": \"$K\", \"speed_scale\": [0.7, 1.0]}]" --output $F --dial-value 1.0489 \
    --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 --opponent-speed-limit > $F.log 2>&1 || { echo "[$(date +%H:%M)] $S $K FAILED"; exit 1; }
done; done
echo "[$(date +%H:%M)] release check $TAG done"
