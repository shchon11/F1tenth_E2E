#!/bin/bash
# Audit C chain 3 (after chain 2): the modular ForzaETH stack (min-time line + spline avoidance + particle filter + MAP) as ego
S=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/C
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
RUNS=/home/shchon11/f1sim_runs
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim
until [ -e $S/rec/chain2.done ]; do sleep 20; done
Q1=$RUNS/pick_scan31b_u475/state_latest.pt
for NAME_K_S in ego_forzaeth_map:forzaeth:0; do
  NAME=${NAME_K_S%%:*}; R=${NAME_K_S#*:}; K=${R%%:*}; ST=${R#*:}
  [ -s $S/rec/$NAME.npz ] && continue
  echo "$(date +%T) start $NAME"
  env C_EGO=$K C_STRESS=$ST F1SIM_EXAM_ALL_BOXES=1 timeout 3600 $PY -u -W ignore $S/run_ego.py $Q1 $S/rec/$NAME.npz 256 line3 120 60 > $S/rec/$NAME.log 2>&1
  echo "$(date +%T) end $NAME rc $?"
done
echo "$(date +%T) chain3 done"; touch $S/rec/chain3.done
