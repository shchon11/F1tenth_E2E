#!/bin/bash
# 콘솔에서 학습 중 checkpoint 보기 (11-05 02:35, 사용자: "지금 학습돌리고있는것들 체크포인트 나온것들 콘솔에서 볼 수 있나?"). 콘솔은 state_latest.pt가
# 있는 폴더만 나열하고 state_rl은 state_u*.pt만 저장 -> watch_<run>/state_latest.pt가 그 run의 가장 최근 state_u*.pt를 가리키게 2분마다 갱신.
# 인자: (watch폴더 run폴더 끝표시파일) 묶음 반복
R=/home/shchon11/f1sim_runs
while true; do
  left=0
  set -- "$@"
  i=1; args=("$@")
  while [ $i -le ${#args[@]} ]; do
    W=${args[$((i-1))]}; RUN=${args[$i]}; DONE=${args[$((i+1))]}
    last=$(ls -1 $R/$RUN/state_u*.pt 2>/dev/null | sed 's/.*state_u\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    [ -n "$last" ] && [ "$(readlink $R/$W/state_latest.pt)" != "$last" ] && ln -sfn "$last" $R/$W/state_latest.pt
    [ -e "$DONE" ] || left=1
    i=$((i+3))
  done
  [ $left = 0 ] && break
  sleep 120
done
