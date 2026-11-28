#!/bin/bash
# chain.sh의 나머지를 순서를 바꿔 이어 감 (23:15, 다른 작업 둘이 4070을 같이 써서 시간이 모자람): R5a/b/c -> R2s -> N4 -> N0.
# N3(명목 차 + PlanMPC)은 R3에서 PlanMPC가 무너져서(무접촉 2 %) 뺌.
S=/home/shchon11/F1tenth_E2E/docs/pm/execgap-2026-11-28/scripts
O=/data/pm-execgap-2026-11-28/rec
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
Q=/home/shchon11/f1sim_runs/pick_scan31b_u475/state_latest.pt
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim
export F1SIM_EXAM_ALL_BOXES=1 F1SIM_EXAM_SAVE_PLAN=1
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
  env XG_STAGE=$ST "$@" timeout 5400 $PY -u -W ignore $S/stage_run.py $Q $O/$NAME.npz 128 line3 120 60 > $O/$NAME.log 2>&1
  echo "$(date +%T) end $NAME rc $?"
}
run R5a_la        map_la
run R5b_gain      map_gain
run R5c_lut       map_lut
run R2s_servoonly servo_nodelay
run N4_map        map   XG_NOMINAL=1
run N0_ideal      ideal XG_NOMINAL=1
echo "$(date +%T) chain3 done"; touch $O/chain3.done
