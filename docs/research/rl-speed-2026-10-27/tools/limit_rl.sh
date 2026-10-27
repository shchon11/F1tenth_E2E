#!/bin/bash
# 10-27 RL research, "the grip at the limit": the clean reward (lap bonus 0, lambda 20, cap 10) with the grip budget BACK but at the
# physical limit (--grip-budget-rho 1.0: |v r| up to the front axle's capacity mu x 0.92 x g = 9.03 m/s^2 is free, sliding beyond
# it pays 2 per (m/s^2)^2 per second). Why: at rho 0.95 the term held the straights back (the negative speed signal); without it
# (v2PHclean_s2432) the policy got faster everywhere (solo 8.36 s, straights 7.30 at u256) but slid past the limit (corner p90
# 10.6) and hit far more (races 3.61 hard/km). Falsifier: corner |v r| p90 stays above ~9.5 or races stay above ~2 hard/km
# (lambda-20 arm 1.26-1.53), or the straights fall back to the lambda-20 arm's (6.87 at u256).
# usage: limit_rl.sh GPU
P=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline; G=$1
echo "[$(date +%H:%M)] limit korea start"
cd $P && GPU=$G TRAIN_WT=/data/wt/ft UPDATES=512 \
  RACE_EXTRA_ADD="--lagrange-init 20 --cap0 10 --cap1 10 --lap-bonus 0 --grip-budget-penalty 2.0 --grip-budget-rho 1.0" \
  bash $P/v2PH_run.sh v2PHlim_s2434 /home/shchon11/f1sim_runs/v2PH_s2421/ppo_u512.pt 2434 || { echo "[$(date +%H:%M)] limit korea FAILED"; exit 1; }
echo "[$(date +%H:%M)] limit korea done"
