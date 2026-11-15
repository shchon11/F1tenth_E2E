#!/bin/bash
# (11-16 03:50, 사용자: "지금 550 1200 인데 뭔 250영상이야") 평가가 학습을 못 따라감 -> 평가는 늘 가장 최신 checkpoint로:
# 1분마다 run별로, 지금 평가 중인 것(아직 .done 없는 가장 작은 update)과 가장 최신 checkpoint 사이에 밀린 것을 건너뜀(.done = skip).
# 인자: run 이름들. core_loop.sh가 끝나면(one_<run>.done) 스스로 끝남
A=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/analysis; D=$(dirname $A); RUNS=/home/shchon11/f1sim_runs
while true; do
  live=0
  for r in "$@"; do
    [ -e $D/one_$r.done ] || live=1
    L=$(ls $RUNS/$r/ 2>/dev/null | sed -n 's/^plan_u\([0-9]*\)\.pt$/\1/p' | sort -n | tail -1); [ -n "$L" ] || continue
    cur=""
    for u in $(ls $RUNS/$r/ | sed -n 's/^plan_u\([0-9]*\)\.pt$/\1/p' | sort -n); do
      [ -e $A/${r}_u$u.done ] && continue
      if [ -z "$cur" ]; then cur=$u; continue; fi           # the one being evaluated now
      [ "$u" -lt "$L" ] && echo "skip (최신만 평가, $(date +%H:%M))" > $A/${r}_u$u.done
    done
  done
  [ $live = 1 ] || break
  sleep 60
done
