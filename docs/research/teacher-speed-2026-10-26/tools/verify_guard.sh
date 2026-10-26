#!/bin/bash
# 10-26: the raceline teacher's env guard, walls on the plan only (new default) vs the whole 1 s preview (old) vs no guard.
# usage: verify_guard.sh GPU solo|race
D=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research/teacher_speed
G=$1; MODE=$2
MAPJ=$(cat $D/entry_map.json); CK=/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar_slow.pt
cd /home/shchon11/F1tenth_E2E/f1sim; mkdir -p $D/out/verify
COMMON="--teacher --action-mode plan --tracks real:korea_2026_competition --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --device cuda:0 --collision-mode soft --speed-mode wp4 --plan-follower map --speed-cap 10.0 --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92"
run() {  # NAME ENVVAR=VAL extra args...
  N=$1; EV=$2; shift 2
  F=$D/out/verify/$N.json; [ -f $F ] && return 0
  env $EV CUDA_VISIBLE_DEVICES=$G OUT=$D/out/verify/$N.bins.json WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 \
    /home/shchon11/F1tenth_E2E/.venv/bin/python -u $D/teacher_speed_chain.py $CK $COMMON --plan-follower-map "$MAPJ" --output $F "$@" > $F.log 2>&1 \
    || { echo "[$(date +%H:%M)] $N FAILED"; exit 1; }
  echo "[$(date +%H:%M)] $N done"
}
if [ $MODE = solo ]; then
  for S in 81 82; do
    run solo_new_s$S GUARD_PLAN_ONLY=1 --envs 16 --steps 2400 --seed $S --race-size 1 --procedural-obstacles 0.0
    run solo_old_s$S GUARD_PLAN_ONLY=0 --envs 16 --steps 2400 --seed $S --race-size 1 --procedural-obstacles 0.0
    run solo_none_s$S NOGUARD=1 --envs 16 --steps 2400 --seed $S --race-size 1 --procedural-obstacles 0.0
  done
else
  for S in ${SEEDS:-81}; do for K in forzaeth forzaeth_pred lane_switch interactive; do for V in new old; do
    E=GUARD_PLAN_ONLY=1; [ $V = old ] && E=GUARD_PLAN_ONLY=0
    run race_${V}_s${S}_$K $E --envs 32 --steps 2400 --seed $S --race-size 2 --graphs --movable-obstacles --spawn-runway 3.0 \
      --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 \
      --opp-slots "[{\"kind\": \"$K\", \"speed_scale\": [0.7, 1.0]}]"
  done; done; done
fi
echo "[$(date +%H:%M)] verify $MODE all done"
