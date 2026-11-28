#!/bin/bash
# chain3.sh 대신 (23:30): R5c가 끝나면 R2s, N4만. N0(명목 차 이상 실행)은 이 PC 메모리 압박을 줄이려고 뺌.
S=/home/shchon11/F1tenth_E2E/docs/pm/execgap-2026-11-28/scripts
O=/data/pm-execgap-2026-11-28/rec
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
Q=/home/shchon11/f1sim_runs/pick_scan31b_u475/state_latest.pt
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim
export F1SIM_EXAM_ALL_BOXES=1 F1SIM_EXAM_SAVE_PLAN=1
until [ -s $O/R5c_lut_xg.npz ] || ! ps -p 2425168 > /dev/null; do sleep 10; done
echo "$(date +%T) end R5c_lut (chain4 saw it)"
wait_res() {
  while true; do
    AV=$(free -g | awk '/^Mem:/{print $7}')
    GU=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '/4070/{print $2}')
    if [ "$AV" -ge 5 ] && [ "$GU" -lt 9000 ]; then return; fi
    echo "$(date +%T) wait: RAM avail ${AV} GB, 4070 used ${GU} MiB"; sleep 60
  done
}
run() {
  local NAME=$1 ST=$2; shift 2
  [ -s $O/${NAME}_xg.npz ] && { echo "$(date +%T) skip $NAME"; return; }
  wait_res
  echo "$(date +%T) start $NAME"
  env XG_STAGE=$ST "$@" timeout 5400 nice $PY -u -W ignore $S/stage_run.py $Q $O/$NAME.npz 128 line3 120 60 > $O/$NAME.log 2>&1
  echo "$(date +%T) end $NAME rc $?"
}
run R2s_servoonly servo_nodelay
run N4_map        map   XG_NOMINAL=1
echo "$(date +%T) chain4 done"; touch $O/chain4.done
