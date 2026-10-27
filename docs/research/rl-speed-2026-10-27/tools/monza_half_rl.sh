#!/bin/bash
# 10-27 RL research, the noise tails at speed: Monza with the grip budget at the limit (monza_limit_rl.sh) and the exploration noise
# HALVED and held (--log-std-shift -0.693 --freeze-log-std). Why: a_lat = v^2 kappa, so the same curvature-knot noise throws lateral
# spikes ~v^2 larger at Monza's speeds -- the limit run's noisy drive had p90 7.25 but p99 11.5 (4 % over 9.03), and with the charge set
# by those tails the policy settled at p90 ~6.1-6.3, straights ~5.9, whether the charge came at once or ramped over 256 updates.
# Falsifier: at u512 straights < 7.0 and corner p90 < 7.5 (the noise tails are not what holds Monza back).
# usage: monza_half_rl.sh GPU
P=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline; G=$1
echo "[$(date +%H:%M)] half monza start"
cd $P && GPU=$G TRAIN_WT=/data/wt/ft TRACK_MAP=rt:Monza MAPJ_FILE=$P/tools/out/map_policy_default.json UPDATES=512 \
  RACE_EXTRA_ADD="--lagrange-init 20 --cap0 10 --cap1 10 --lap-bonus 0 --grip-budget-penalty 2.0 --grip-budget-rho 1.0 --log-std-shift -0.693 --freeze-log-std --raceline-objective min_curvature" \
  bash $P/v2PH_run.sh monzaHalf_s2539 /home/shchon11/f1sim_runs/dgFT_s2501/student_it9.pt 2539 || { echo "[$(date +%H:%M)] half monza FAILED"; exit 1; }
echo "[$(date +%H:%M)] half monza done"
for U in 128 256 384 512; do MZOUT=monza_half bash $P/tools/research/teacher_speed/monza_eval2.sh $G u$U /home/shchon11/f1sim_runs/monzaHalf_s2539/ppo_u$U.pt; done
echo "[$(date +%H:%M)] half monza evaluated"
