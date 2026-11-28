#!/bin/bash
# Audit C chain 1: qualifier-format solo exams + 1v1 vs self-copy, sequential on the 4070 (CUDA_VISIBLE_DEVICES=0)
S=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/C
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
RUNS=/home/shchon11/f1sim_runs
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim
run() {  # name ckpt mode count stress [extra env...]
  local NAME=$1 CK=$2 MODE=$3 CNT=$4 ST=$5; shift 5
  [ -e $S/rec/$NAME.npz ] && { echo "$(date +%T) skip $NAME"; return; }
  echo "$(date +%T) start $NAME"
  env C_STRESS=$ST "$@" timeout 3600 $PY -u -W ignore $S/run_exam.py $CK $S/rec/$NAME.npz $MODE 256 $CNT 120 60 > $S/rec/$NAME.log 2>&1
  echo "$(date +%T) end $NAME rc $?"
}
Q1=$RUNS/pick_scan31b_u475/state_latest.pt
Q2=$RUNS/pick_hs43b_u900/state_latest.pt
R1=$RUNS/pick_race33b_u975/state_latest.pt
R2=$RUNS/pick_rc42a_u950/state_latest.pt
run solo_scan31b475_ideal $Q1 ideal line3 0 F1SIM_EXAM_ALL_BOXES=1
run solo_scan31b475_map   $Q1 map   line3 0 F1SIM_EXAM_ALL_BOXES=1
run solo_hs43b900_ideal   $Q2 ideal line3 0 F1SIM_EXAM_ALL_BOXES=1
run solo_hs43b900_map     $Q2 map   line3 0 F1SIM_EXAM_ALL_BOXES=1
run solo_scan31b475_mapS  $Q1 map   line3 1 F1SIM_EXAM_ALL_BOXES=1
run solo_hs43b900_mapS    $Q2 map   line3 1 F1SIM_EXAM_ALL_BOXES=1
run race_race33b975_map   $R1 map   race4 0 F1SIM_EXAM4_PACES=0.7,0.8,0.9,0.95
run race_race33b975_ideal $R1 ideal race4 0 F1SIM_EXAM4_PACES=0.7,0.8,0.9,0.95
run race_rc42a950_map     $R2 map   race4 0 F1SIM_EXAM4_PACES=0.7,0.8,0.9,0.95
echo "$(date +%T) chain1 done"; touch $S/rec/chain1.done
