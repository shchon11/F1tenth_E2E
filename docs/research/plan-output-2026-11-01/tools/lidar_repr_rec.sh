#!/bin/bash
# usage: rec.sh NAME COUNT NENV SECS RMT SEED "EXTRA ENV..."  -> ~/F1tenth/reprtest/out/NAME.npz (+ .log, peak RSS in .rss)
N=$1; CNT=$2; NE=$3; SECS=$4; RMT=$5; SEED=$6; XE=$7
O=$HOME/F1tenth/reprtest/out; mkdir -p $O
cd $HOME/F1tenth/reprtest/f1sim
env PYTHONPATH=$HOME/F1tenth/reprtest/f1sim WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 \
  F1SIM_LIDAR_MOUNT_X=0.12555 CUDA_VISIBLE_DEVICES=0 F1SIM_EXAM_SEED=$SEED $XE \
  systemd-run --user --scope -q -p MemoryMax=10G /usr/bin/time -v -o $O/$N.rss timeout 5400 \
  $HOME/F1tenth/.venv/bin/python3 -u -W ignore $HOME/F1tenth/reprtest/tools/exam_record.py \
  $HOME/f1sim_runs/race33b/plan_u925.pt $O/$N.npz ideal $NE $CNT $SECS $RMT > $O/$N.log 2>&1
echo "rc $?" >> $O/$N.log
