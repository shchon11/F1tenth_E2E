#!/bin/bash
# 10-26: (a) the teacher's profile a_lat 7 / 8 / 9 on the same line under the FULL randomisation (mu 0.7-1.1, mu_f 0.85-1.0
# and the rest), MAP; (b) the guard fix under the iLQR tracker DAgger collects with (preview "legacy_mpc"). Korea solo.
# usage: verify_dr.sh GPU
D=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research/teacher_speed
G=$1
MAPJ=$(cat $D/entry_map.json); CK=/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar_slow.pt
cd /home/shchon11/F1tenth_E2E/f1sim; mkdir -p $D/out/verify
BASE="--teacher --action-mode plan --tracks real:korea_2026_competition --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --device cuda:0 --collision-mode soft --speed-mode wp4 --speed-cap 10.0 --dial-value 1.0489 --envs 16 --steps 2400 --race-size 1 --procedural-obstacles 0.0"
run() {  # NAME ENVVAR=VAL extra args...
  N=$1; EV=$2; shift 2
  F=$D/out/verify/$N.json; [ -f $F ] && return 0
  env $EV CUDA_VISIBLE_DEVICES=$G OUT=$D/out/verify/$N.bins.json WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 \
    /home/shchon11/F1tenth_E2E/.venv/bin/python -u $D/teacher_speed_chain.py $CK $BASE --output $F "$@" > $F.log 2>&1 \
    || { echo "[$(date +%H:%M)] $N FAILED"; exit 1; }
  echo "[$(date +%H:%M)] $N done"
}
for S in 81 82; do
  for A in 7.0 8.0 9.0; do
    run dr_alat${A}_s$S TEACHER_ALAT=$A --seed $S --plan-follower map --plan-follower-map "$MAPJ"
  done
  run mpc_new_s$S GUARD_PLAN_ONLY=1 --seed $S --plan-follower mpc --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92
  run mpc_old_s$S GUARD_PLAN_ONLY=0 --seed $S --plan-follower mpc --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92
done
echo "[$(date +%H:%M)] verify dr all done"
