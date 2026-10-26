#!/bin/bash
# 10-26 THE FIXED-TEACHER PIPELINE (the user: "teacher부터 잘못됐네"). One GPU job at a time on the 4070, stop on failure.
#  1. dg_ft.sh: the general DAgger (149 train tracks, races of 4, props off the line) from the fixed teacher, init = the entry,
#     plans followed by MAP, no 8 m/s cap, 10 iterations. TEACHER / ALAT from the env (decided by the 10-26 verification).
#     The 5060 evaluates every iteration on Korea meanwhile (watch_dgft.sh).
#  2. v2PH_run.sh: the Korea fine PPO that made the entry (races of 4 + league, MAP with the Korea parameters, rep floor, dial
#     held, lambda held 80, 512 updates) from the last DAgger iteration, with the cap at 10 instead of 9.
# usage: TEACHER=.. ALAT=.. ft_chain.sh DG_TAG DG_SEED PPO_TAG PPO_SEED
set -u
P=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline; T=$P/tools
DG=$1; DGS=$2; PT=$3; PTS=$4; ITERS=${ITERS:-10}
echo "[$(date +%H:%M)] ft chain: DAgger $DG (teacher ${TEACHER:-raceline}, profile a_lat ${ALAT:-8.0}, $ITERS iterations)"
GPU=0 WT=/data/wt/ft bash $T/dg_ft.sh $DG $DGS $ITERS || { echo "[$(date +%H:%M)] DAgger FAILED"; exit 1; }
LAST=/home/shchon11/f1sim_runs/$DG/student_it$((ITERS - 1)).pt
[ -f $LAST ] || { echo "[$(date +%H:%M)] no $LAST"; exit 1; }
grep -q "${DG} exit 0" $P/$DG.log || { echo "[$(date +%H:%M)] DAgger exit not 0"; exit 1; }
echo "[$(date +%H:%M)] DAgger done -> PPO $PT from $LAST"
GPU=0 TRAIN_WT=/data/wt/ft RACE_EXTRA_ADD="--cap0 10 --cap1 10" bash $P/v2PH_run.sh $PT $LAST $PTS || { echo "[$(date +%H:%M)] PPO FAILED"; exit 1; }
[ -f /home/shchon11/f1sim_runs/$PT/ppo_u512.pt ] || { echo "[$(date +%H:%M)] no u512"; exit 1; }
echo "[$(date +%H:%M)] ft chain done"
