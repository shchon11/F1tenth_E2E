#!/bin/bash
# (11-16) 학습 중인 run의 최신 checkpoint(watch_<run>/state_latest.pt)를 lab PC에 15분마다 보냄 -- lab 콘솔의 '학습 중' 항목이 옛 사본에
# 멈춰 있던 것(kr26a u200을 보는데 학습은 u250). 인자: run 이름들. 학습이 다 끝나면(one_<run>.done) 스스로 끝남
H=shchon11@10.100.0.2; SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"; RUNS=/home/shchon11/f1sim_runs
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
while true; do
  live=0
  for r in "$@"; do
    [ -e $D/one_$r.done ] || live=1
    for t in 1 2 3; do (tar -chf - -C $RUNS watch_$r | $SSH $H 'tar -x -C ~/f1sim_runs') && break; sleep 5; done
  done
  echo "$(date +%T) 보냄: $*"
  [ $live = 1 ] || break
  sleep 900
done
