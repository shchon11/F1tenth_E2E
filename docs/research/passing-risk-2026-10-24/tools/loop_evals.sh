#!/bin/bash
# 10-26: evaluations for the real-loop fine-tune. usage: loop_evals.sh GPU MODE CKPT OUTNAME
# MODE real = the competition car's loop (drive cap 2.5 + 0.1 s speed delay, braking default; realspeed_eval.py TAU="") ; default = plain
R=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research
GPU=$1; MODE=$2; CK=$3; O=$R/out/$4; mkdir -p $O
cd /home/shchon11/F1tenth_E2E/f1sim
COMMON="--action-mode plan --tracks real:korea_2026_competition --envs 32 --steps 2400 --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --speed-cap 9.0 --device cuda:0 --graphs --collision-mode soft --movable-obstacles --spawn-runway 3.0 --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 --procedural-raceline-corridor off --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92"
if [ $MODE = real ]; then ENTRY="$R/realspeed_eval.py"; export TAU=""; else ENTRY="-m f1sim.learn.evaluate"; fi
for S in 81 82; do for K in forzaeth forzaeth_pred lane_switch interactive; do
  F=$O/race_s${S}_${K}.json; [ -f $F ] && continue
  CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u $ENTRY $CK $COMMON \
    --seed $S --race-size 2 --opp-slots "[{\"kind\": \"$K\", \"speed_scale\": [0.7, 1.0]}]" --output $F > $F.log 2>&1 || { echo "[$(date +%H:%M)] $4 $S $K FAILED"; exit 1; }
done; done
for S in 81 82; do
  F=$O/solo_s${S}.json; [ -f $F ] && continue
  CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u $ENTRY $CK $COMMON \
    --seed $S --race-size 1 --output $F > $F.log 2>&1 || { echo "[$(date +%H:%M)] $4 solo $S FAILED"; exit 1; }
done
echo "[$(date +%H:%M)] $4 done"
