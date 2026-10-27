#!/bin/bash
# 10-27 RL research, the ratchet: Monza with the grip budget at the physical limit (rho 1.0) RAMPED from 0 over 256 updates
# (--grip-budget-ramp 256), else monza_limit_rl.sh. H: the early shock of the full quadratic charge on a start far over the limit
# made PPO cut speed everywhere, and a 2 cm speed signal never brought it back (limit run: straights 8.08 -> 5.88, corner p90 6.3;
# noisy p90 7.25, 1.8 under the budget). Falsifier: at u512 the straights still fall below ~7.5 (the cut happens anyway), or the
# corner p90 stays above ~10 (the charge never removes the sliding).
# usage: monza_ramp_rl.sh GPU
P=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline; G=$1
echo "[$(date +%H:%M)] ramp monza start"
cd $P && GPU=$G TRAIN_WT=/data/wt/ft TRACK_MAP=rt:Monza MAPJ_FILE=$P/tools/out/map_policy_default.json UPDATES=512 \
  RACE_EXTRA_ADD="--lagrange-init 20 --cap0 10 --cap1 10 --lap-bonus 0 --grip-budget-penalty 2.0 --grip-budget-rho 1.0 --grip-budget-ramp 256 --raceline-objective min_curvature" \
  bash $P/v2PH_run.sh monzaRamp_s2537 /home/shchon11/f1sim_runs/dgFT_s2501/student_it9.pt 2537 || { echo "[$(date +%H:%M)] ramp monza FAILED"; exit 1; }
echo "[$(date +%H:%M)] ramp monza done"
for U in 128 256 384 512; do MZOUT=monza_ramp bash $P/tools/research/teacher_speed/monza_eval2.sh $G u$U /home/shchon11/f1sim_runs/monzaRamp_s2537/ppo_u$U.pt; done
echo "[$(date +%H:%M)] ramp monza evaluated"
