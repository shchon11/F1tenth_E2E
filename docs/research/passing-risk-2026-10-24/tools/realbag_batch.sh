#!/bin/bash
# CPU (10-25): every competition-venue bag through the policy node (nearest-car readings) and the map check (objects the map
# cannot explain), then the head check. Skips bags already done.
R=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research
for B in /home/shchon11/F1tenth_E2E/real_data/01_competition_0826-0827_map08x23/*/; do
  N=$(basename $B | cut -c1-15)
  [ -f $R/out/realbag/check_$N.txt ] && continue
  [ -f $R/out/realbag/read_$N.npz ] || bash $R/replay_bag_policy.sh $B $R/out/realbag/read_$N.npz > /dev/null 2>&1
  [ -f $R/out/realbag/read_$N.npz ] || { echo "[$(date +%H:%M)] $N: no readings"; continue; }
  (cd ~ && source ~/f1tenth_ws/activate.sh >/dev/null 2>&1 && GOOD=0.95 OBJS=$R/out/realbag/objs_$N.npz timeout 900 python3 $R/real_bag_unexplained.py $B > $R/out/realbag/map_$N.txt 2>&1)
  /home/shchon11/F1tenth_E2E/.venv/bin/python $R/real_bag_head_check.py $R/out/realbag/objs_$N.npz $R/out/realbag/read_$N.npz > $R/out/realbag/check_$N.txt 2>&1
  echo "[$(date +%H:%M)] $N done"
done
