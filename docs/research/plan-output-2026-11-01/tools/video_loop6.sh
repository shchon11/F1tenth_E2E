#!/bin/bash
# 정기 영상 (11-10, 사용자: "그 영상은 학습환경하고 제어기 붙인걸로 각각 뽑아주고, 한 영상에 네 경우(빈, 장애물 있다없어진, +상대차)다
# 들어가게 만들어"): checkpoint(STEP update)마다 four_case_video.sh 두 개 -- 학습 조건(ideal) / 물리 차 + 새 제어기(track).
# 결과 out/videos/NAME_uUU_{ideal,track}_4cases.mp4(+ .log). 인자: 이름 시작 간격 끝
NAME=$1; U0=$2; STEP=$3; U1=$4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
for UU in $(seq $U0 $STEP $U1); do
  C=$R/${NAME}/plan_u${UU}.pt
  until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 10
  for M in ideal track; do
    V=$D/videos/${NAME}_u${UU}_${M}_4cases
    [ -e ${V}.done ] && continue
    bash $T/four_case_video.sh $C $M ${V}.mp4
    echo done > ${V}.done
  done
done
