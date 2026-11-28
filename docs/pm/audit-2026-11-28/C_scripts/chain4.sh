#!/bin/bash
# Audit C chain 4 (after chain 3): the same qualifier exam on a hose variant never used in training (+hose23)
S=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/C
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
RUNS=/home/shchon11/f1sim_runs
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim
until [ -e $S/rec/chain3.done ]; do sleep 15; done
for X in solo_scan31b475_mapHO:pick_scan31b_u475:map solo_hs43b900_mapHO:pick_hs43b_u900:map solo_scan31b475_idealHO:pick_scan31b_u475:ideal; do
  NAME=${X%%:*}; R=${X#*:}; P=${R%%:*}; MODE=${R#*:}
  [ -s $S/rec/$NAME.npz ] && continue
  echo "$(date +%T) start $NAME"
  env C_HOSE=23 C_BANK=$S/bank_ho23 F1SIM_EXAM_ALL_BOXES=1 timeout 3600 $PY -u -W ignore $S/run_heldout.py $RUNS/$P/state_latest.pt $S/rec/$NAME.npz $MODE 256 line3 120 60 > $S/rec/$NAME.log 2>&1
  echo "$(date +%T) end $NAME rc $?"
done
echo "$(date +%T) chain4 done"; touch $S/rec/chain4.done
