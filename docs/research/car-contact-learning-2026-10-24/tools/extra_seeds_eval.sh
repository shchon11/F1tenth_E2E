#!/bin/bash
# extra selection-protocol race seeds (83-84 x 4 kinds, props on the line as the finals) for a checkpoint: GPU CKPT TAG
GPU=$1; CK=$2; TAG=$3
R=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research
mkdir -p $R/out/xseeds/$TAG
for seed in 83 84; do for kind in forzaeth forzaeth_pred lane_switch interactive; do
  f=$R/out/xseeds/$TAG/race_s${seed}_${kind}.json; [ -f $f ] && continue
  cd /data/wt/v2r/f1sim && PYTHONPATH=/data/wt/v2r/f1sim:/data/wt/v2r/f1sim_ros CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 \
    /home/shchon11/F1tenth_E2E/.venv/bin/python -u -m f1sim.learn.evaluate $CK --action-mode plan \
    --tracks real:korea_2026_competition --envs 32 --steps 2400 --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 \
    --raceline-objective min_time --dial-offset 0.0 --speed-cap 9.0 --device cuda:0 --seed $seed --graphs --collision-mode soft \
    --movable-obstacles --spawn-runway 3.0 --race-size 2 --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 \
    --procedural-raceline-corridor off --opp-slots "[{\"kind\": \"$kind\", \"speed_scale\": [0.7, 1.0]}]" --dial-value 1.0489 \
    --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 --output $f > $f.log 2>&1 || { echo "FAILED $TAG $seed $kind"; exit 1; }
done; done
echo "[$(date +%H:%M)] $TAG extra seeds done"
