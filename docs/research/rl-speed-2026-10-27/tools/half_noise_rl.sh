#!/bin/bash
# 10-27 RL research, H3 (the noise margin): limit_rl.sh's run with the exploration noise HALVED and held (--log-std-shift -0.693
# --freeze-log-std). Measured: the limit-budget checkpoint u256 drove with its training noise at corner |v r| p90 8.96 (the budget
# 9.03) and without it at 8.52; noise cost 0.5 m/s on the straights. Falsifier: at u256 the deterministic corner p90 is not above
# ~8.8 and the straights not above ~7.0 (limit run: 8.52 / 6.87), or hard contacts rise above the limit run's.
# usage: half_noise_rl.sh GPU
P=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline; G=$1
echo "[$(date +%H:%M)] half korea start"
cd $P && GPU=$G TRAIN_WT=/data/wt/ft UPDATES=512 \
  RACE_EXTRA_ADD="--lagrange-init 20 --cap0 10 --cap1 10 --lap-bonus 0 --grip-budget-penalty 2.0 --grip-budget-rho 1.0 --log-std-shift -0.693 --freeze-log-std" \
  bash $P/v2PH_run.sh v2PHhalf_s2436 /home/shchon11/f1sim_runs/v2PH_s2421/ppo_u512.pt 2436 || { echo "[$(date +%H:%M)] half korea FAILED"; exit 1; }
echo "[$(date +%H:%M)] half korea done"
