#!/bin/bash
# chain.sh 다음: R2s(이상 실행 + 서보만, 지연 없음). 같은 세션·같은 설정.
S=/home/shchon11/F1tenth_E2E/docs/pm/execgap-2026-11-28/scripts
O=/data/pm-execgap-2026-11-28/rec
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
Q=/home/shchon11/f1sim_runs/pick_scan31b_u475/state_latest.pt
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim
export F1SIM_EXAM_ALL_BOXES=1 F1SIM_EXAM_SAVE_PLAN=1
until [ -e $O/chain.done ]; do sleep 20; done
for X in R2s_servoonly:servo_nodelay; do
  NAME=${X%%:*}; ST=${X#*:}
  [ -s $O/${NAME}_xg.npz ] && continue
  while [ "$(free -g | awk '/^Mem:/{print $7}')" -lt 5 ]; do echo "$(date +%T) wait RAM"; sleep 120; done
  echo "$(date +%T) start $NAME"
  env XG_STAGE=$ST timeout 5400 $PY -u -W ignore $S/stage_run.py $Q $O/$NAME.npz 128 line3 120 60 > $O/$NAME.log 2>&1
  echo "$(date +%T) end $NAME rc $?"
done
touch $O/chain2.done
