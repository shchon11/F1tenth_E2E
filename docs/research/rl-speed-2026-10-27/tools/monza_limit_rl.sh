#!/bin/bash
# 10-27: Monza with the grip budget at the physical limit (limit_rl.sh's reward on rt:Monza): the old Monza RL (budget at 0.95)
# cut corner |v r| p90 10.4 -> 6.1 and slowed everything (straights 8.08 -> 5.63); without the budget (monzaClean) straights 8.65
# but p90 13.7 and hard wall 4.3/km. From dgFT it9, lambda 20, cap 10, min-curvature lines, POLICY_MAP defaults, 512 updates.
# usage: monza_limit_rl.sh GPU
P=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline; G=$1
echo "[$(date +%H:%M)] limit monza start"
cd $P && GPU=$G TRAIN_WT=/data/wt/ft TRACK_MAP=rt:Monza MAPJ_FILE=$P/tools/out/map_policy_default.json UPDATES=512 \
  RACE_EXTRA_ADD="--lagrange-init 20 --cap0 10 --cap1 10 --lap-bonus 0 --grip-budget-penalty 2.0 --grip-budget-rho 1.0 --raceline-objective min_curvature" \
  bash $P/v2PH_run.sh monzaLim_s2535 /home/shchon11/f1sim_runs/dgFT_s2501/student_it9.pt 2535 || { echo "[$(date +%H:%M)] limit monza FAILED"; exit 1; }
echo "[$(date +%H:%M)] limit monza done"
