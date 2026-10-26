#!/bin/bash
# Evaluate each dg_ft DAgger iteration on Korea as it lands (10-26): solo, no props (lap, top speed, the command on the
# straights -- the thing the run is for) and one race per kind pair (speed, hard contacts, passes), MAP with the Korea
# parameters, cap 10. usage: GPU=1 watch_dgft.sh TAG ITERS
TAG=$1; ITERS=$2; G=${GPU:-1}
R=/home/shchon11/f1sim_runs/$TAG; D=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research/teacher_speed
O=$D/out/watch_$TAG; mkdir -p $O; MAPJ=$(cat $D/entry_map.json)
WT=${WT:-/data/wt/ft}; cd $WT/f1sim && export PYTHONPATH=$WT/f1sim:$WT/f1sim_ros:${PYTHONPATH:-}
COMMON="--action-mode plan --tracks real:korea_2026_competition --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --device cuda:0 --collision-mode soft --plan-follower map --speed-cap 10.0 --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92"
for it in $(seq 0 $((ITERS - 1))); do
  CK=$R/student_it$it.pt
  while [ ! -f $CK ]; do grep -q "${TAG} exit" /home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/$TAG.log 2>/dev/null && [ ! -f $CK ] && { echo "[$(date +%H:%M)] run ended before it$it"; exit 0; }; sleep 60; done
  sleep 30
  F=$O/solo_it$it.json
  [ -f $F ] || CUDA_VISIBLE_DEVICES=$G OUT=$O/solo_it$it.bins.json WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u \
    $D/teacher_speed_chain.py $CK $COMMON --plan-follower-map "$MAPJ" --output $F --envs 16 --steps 2400 --seed 81 --race-size 1 \
    --procedural-obstacles 0.0 > $F.log 2>&1 || echo "[$(date +%H:%M)] it$it solo FAILED"
  for K in forzaeth_pred interactive; do
    F=$O/race_it${it}_$K.json
    [ -f $F ] || CUDA_VISIBLE_DEVICES=$G WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u -m f1sim.learn.evaluate \
      $CK $COMMON --plan-follower-map "$MAPJ" --output $F --envs 32 --steps 2400 --seed 81 --race-size 2 --graphs --movable-obstacles \
      --spawn-runway 3.0 --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 \
      --opp-slots "[{\"kind\": \"$K\", \"speed_scale\": [0.7, 1.0]}]" > $F.log 2>&1 || echo "[$(date +%H:%M)] it$it race $K FAILED"
  done
  echo "[$(date +%H:%M)] it$it evaluated"
done
echo "[$(date +%H:%M)] watch $TAG all done"
