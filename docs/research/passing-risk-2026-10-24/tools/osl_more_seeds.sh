#!/bin/bash
# GPU CORRIDOR TAG (10-25): more fresh seeds (93, 94) for the shipped speed limit, limit off then on, on ONE card (a comparison
# across cards diverges in the prop-contact chaos: same code, 5060 vs 4070, props on, total 1.63 vs 2.22 on seeds 91-92)
GPU=$1; CORR=$2; TAG=$3
ROOT=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research/out
cd /home/shchon11/F1tenth_E2E/f1sim
for MODE in off on; do
  OUT=$ROOT/more_${TAG}_${MODE}; mkdir -p $OUT
  FLAG=""; [ $MODE = on ] && FLAG="--opponent-speed-limit"
  for S in 93 94; do for K in forzaeth forzaeth_pred lane_switch interactive; do
    F=$OUT/race_s${S}_${K}.json
    [ -f $F ] && continue
    CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u -m f1sim.learn.evaluate \
      /home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar.pt --action-mode plan --tracks real:korea_2026_competition \
      --envs 32 --steps 2400 --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time \
      --dial-offset 0.0 --speed-cap 9.0 --device cuda:0 --seed $S --graphs --collision-mode soft --movable-obstacles --spawn-runway 3.0 \
      --race-size 2 --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 --procedural-raceline-corridor $CORR \
      --opp-slots "[{\"kind\": \"$K\", \"speed_scale\": [0.7, 1.0]}]" --output $F --dial-value 1.0489 \
      --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 $FLAG > $F.log 2>&1 || { echo "[$(date +%H:%M)] $MODE $S $K FAILED"; exit 1; }
  done; done
  echo "[$(date +%H:%M)] $TAG $MODE done"
done
