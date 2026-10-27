#!/bin/bash
# One checkpoint on the rl-speed yardstick (10-27), on the MAIN checkout's code: Korea solo (no props, rep floor, the checkpoint's
# own MAP, cap 10, 16 envs x 2400 steps, seed 81: lap, per-raceline-speed-bin command/speed via teacher_speed_chain.py) and races
# (race of 2 vs each of the 4 kinds, props on the line, 32 envs x 2400 steps, seeds 81-82).
# usage: eval_ckpt.sh GPU NAME CKPT [solo|race|both]
D=$(cd $(dirname $0) && pwd); G=$1; N=$2; CK=$3; WHAT=${4:-both}; MAPJ=$(cat $D/entry_map.json)
O=${OUT_DIR:-$D/../out}/$N; mkdir -p $O
cd /home/shchon11/F1tenth_E2E/f1sim && export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
COMMON="--action-mode plan --tracks real:korea_2026_competition --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --device cuda:0 --collision-mode soft --speed-cap ${CAP:-10.0} --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92"
if [ "$WHAT" != race ]; then
  F=$O/solo.json
  [ -f $F ] || CUDA_VISIBLE_DEVICES=$G OUT=$O/solo.bins.json WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 $PY -u $D/teacher_speed_chain.py $CK $COMMON \
    --output $F --envs 16 --steps 2400 --seed 81 --race-size 1 --procedural-obstacles 0.0 > $F.log 2>&1 || { echo "$N solo FAILED"; exit 1; }
  echo "[$(date +%H:%M)] $N solo done"
fi
if [ "$WHAT" != solo ]; then
  for S in 81 82; do for K in forzaeth forzaeth_pred lane_switch interactive; do
    F=$O/race_s${S}_$K.json; [ -f $F ] && continue
    CUDA_VISIBLE_DEVICES=$G WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 $PY -u -m f1sim.learn.evaluate $CK $COMMON \
      --plan-follower map --plan-follower-map "$MAPJ" --output $F --envs 32 --steps 2400 --seed $S --race-size 2 --graphs \
      --movable-obstacles --spawn-runway 3.0 --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 \
      --opp-slots "[{\"kind\": \"$K\", \"speed_scale\": [0.7, 1.0]}]" > $F.log 2>&1 || { echo "$N s$S $K FAILED"; exit 1; }
  done; done
  echo "[$(date +%H:%M)] $N races done"
fi
