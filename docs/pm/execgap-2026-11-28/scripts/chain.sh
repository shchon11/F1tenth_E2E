#!/bin/bash
# 실행 격차 분해: 같은 정책(scan31b u475)·같은 세션 64개(exam3 배치 64개, 120 s, 60 s에 상자 치움)를 단계별로 차례로.
# 4070만(CUDA_VISIBLE_DEVICES=0), 한 번에 하나. 시작 전에 RAM 여유 5 GB, 4070 메모리 여유를 확인하고 모자라면 기다림.
S=/home/shchon11/F1tenth_E2E/docs/pm/execgap-2026-11-28/scripts
O=/data/pm-execgap-2026-11-28/rec
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
Q=/home/shchon11/f1sim_runs/pick_scan31b_u475/state_latest.pt
mkdir -p $O
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim
export F1SIM_EXAM_ALL_BOXES=1 F1SIM_EXAM_SAVE_PLAN=1
wait_res() {
  while true; do
    AV=$(free -g | awk '/^Mem:/{print $7}')
    GU=$(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '/4070/{print $2}')
    if [ "$AV" -ge 5 ] && [ "$GU" -lt 8000 ]; then return; fi
    echo "$(date +%T) wait: RAM avail ${AV} GB, 4070 used ${GU} MiB"; sleep 120
  done
}
run() {  # name stage [extra env]
  local NAME=$1 ST=$2; shift 2
  [ -s $O/${NAME}_xg.npz ] && { echo "$(date +%T) skip $NAME"; return; }
  wait_res
  echo "$(date +%T) start $NAME"
  env XG_STAGE=$ST "$@" timeout 5400 $PY -u -W ignore $S/stage_run.py $Q $O/$NAME.npz 128 line3 120 60 > $O/$NAME.log 2>&1
  echo "$(date +%T) end $NAME rc $?"
}
run R4_map        map
run R0_ideal      ideal
run R1_delay      delay
run R2_servo      servo
run R2b_motor     servo_motor
run R3_mpc        mpc
run R6_pf         pf
run R3b_pforacle  pf_oracle
run R5a_la        map_la
run R5b_gain      map_gain
run R5c_lut       map_lut
run N0_ideal      ideal XG_NOMINAL=1
run N4_map        map   XG_NOMINAL=1
run N3_mpc        mpc   XG_NOMINAL=1
echo "$(date +%T) chain done"; touch $O/chain.done
