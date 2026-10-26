#!/bin/bash
# 10-27 RL research: the "clean reward" fine-tunes. Measured (tools/research/rl_speed, snr_components.py): the grip-budget penalty
# (lateral |v r| above 0.95 x the front axle at the floor's mu, quadratic, x2) is the reward term that pushes the speed DOWN (its
# share of the v1 signal -0.0045 of -0.0061; faster straights carry more lateral g into the next corner), and the critic learnt
# it (after 40 adaptation updates without it the critic still cancels the progress signal: +0.0088 vs -0.0067). Its purpose -- a
# dial that is a command -- is moot with the dial held. The lap bonus (5 a lap) is proportional to progress but spiky: removing
# both cuts the advantage sd 2.33 -> 1.48 and the critic loss 5x. So: --grip-budget-penalty 0 --lap-bonus 0, else the v2PH
# recipe with lambda 20 and cap 10 (no unfounded cap). Falsifier: the straights climb no faster than the lambda-20 race arm
# (cmd@line9+ 6.56 / 6.74 / 6.87 at u64 / 128 / 256), or Monza still sheds its straight speed (8.08 -> 5.63 at u512).
# usage: clean_rl.sh korea|monza GPU
P=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline; W=$1; G=$2
CLEAN="--lagrange-init 20 --cap0 10 --cap1 10 --grip-budget-penalty 0 --lap-bonus 0"
echo "[$(date +%H:%M)] clean $W start"
if [ "$W" = korea ]; then
  cd $P && GPU=$G TRAIN_WT=/data/wt/ft UPDATES=512 RACE_EXTRA_ADD="$CLEAN" \
    bash $P/v2PH_run.sh v2PHclean_s2432 /home/shchon11/f1sim_runs/v2PH_s2421/ppo_u512.pt 2432 || { echo "[$(date +%H:%M)] clean $W FAILED"; exit 1; }
else
  cd $P && GPU=$G TRAIN_WT=/data/wt/ft TRACK_MAP=rt:Monza MAPJ_FILE=$P/tools/out/map_policy_default.json UPDATES=512 \
    RACE_EXTRA_ADD="$CLEAN --raceline-objective min_curvature" \
    bash $P/v2PH_run.sh monzaClean_s2533 /home/shchon11/f1sim_runs/dgFT_s2501/student_it9.pt 2533 || { echo "[$(date +%H:%M)] clean $W FAILED"; exit 1; }
fi
echo "[$(date +%H:%M)] clean $W done"
