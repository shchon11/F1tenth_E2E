#!/bin/bash
# 10-27 RL research, "price the consequence, not the proxy": the clean reward (no grip budget, no lap bonus, cap 10) with the hard-hit
# multiplier at the entry's own 80 instead of 20. The clean run at lambda 20 got fast (solo 8.32 s, straights 7.31) by sliding past
# the grip (p90 10.6) and hitting (races 3.61 hard/km): at 20 m a hit was cheap next to the lap-time bonus. Any grip charge instead
# (0.95 or 1.0) held corners ~0.5 under the limit and the straights at ~6.9. Falsifier: at u256 straights < 7.0 or solo lap > 8.45 s
# (the price cuts speed like the proxy did), or race hard contacts > ~1.6/km (the price does not restrain the sliding).
# usage: clean80_rl.sh GPU
P=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline; G=$1
echo "[$(date +%H:%M)] clean80 korea start"
cd $P && GPU=$G TRAIN_WT=/data/wt/ft UPDATES=512 \
  RACE_EXTRA_ADD="--lagrange-init 80 --cap0 10 --cap1 10 --grip-budget-penalty 0 --lap-bonus 0" \
  bash $P/v2PH_run.sh v2PHc80_s2438 /home/shchon11/f1sim_runs/v2PH_s2421/ppo_u512.pt 2438 || { echo "[$(date +%H:%M)] clean80 korea FAILED"; exit 1; }
echo "[$(date +%H:%M)] clean80 korea done"
