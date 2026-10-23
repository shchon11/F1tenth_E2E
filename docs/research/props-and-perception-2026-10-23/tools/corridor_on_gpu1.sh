#!/bin/bash
# 5060 (10-23, R3d): the finals race cells with the props kept OFF the racing line (--procedural-raceline-corridor on, the
# layout's design default) -- how much of the race hard-hit metric is the on-line props that the opponents knock about.
R=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research
for spec in "korea26_v2_ppo_s2421_u512 corr_v2entry" "korea26_rep_s2001_u256 corr_v1entry"; do
  set -- $spec
  mkdir -p $R/out/$2
  for seed in 81 82; do for kind in forzaeth forzaeth_pred lane_switch interactive; do
    cd /data/wt/v2k/f1sim && PYTHONPATH=/data/wt/v2k/f1sim:/data/wt/v2k/f1sim_ros CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 \
      /home/shchon11/F1tenth_E2E/.venv/bin/python -u -m f1sim.learn.evaluate /home/shchon11/F1tenth_E2E/checkpoints/$1.pt --action-mode plan \
      --tracks real:korea_2026_competition --envs 32 --steps 2400 --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 \
      --raceline-objective min_time --dial-offset 0.0 --speed-cap 9.0 --device cuda:0 --seed $seed --graphs --collision-mode soft \
      --movable-obstacles --spawn-runway 3.0 --race-size 2 --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 \
      --procedural-raceline-corridor on --opp-slots "[{\"kind\": \"$kind\", \"speed_scale\": [0.7, 1.0]}]" --dial-value 1.0489 \
      --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 --output $R/out/$2/race_s${seed}_${kind}.json > $R/out/$2/race_s${seed}_${kind}.log 2>&1 \
      || { echo "FAILED $2 $seed $kind"; exit 1; }
  done; done
  echo "[$(date +%H:%M)] $2 done"
done
echo "[$(date +%H:%M)] CORRIDOR_ON_DONE"
