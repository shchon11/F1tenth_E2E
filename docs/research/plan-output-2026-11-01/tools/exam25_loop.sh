#!/bin/bash
# (11-17, 사용자: "저 충돌위치 분포그림 100부터 25단위로 쭈르륵 보고싶은데 ... 100단위는 너무 sparse해서") frozen exam을 100 사이의
# checkpoint(25 · 50 · 75)에도: 5060에서(학습 GPU를 멈추지 않게), 100의 배수는 exam_loop.sh가 맡으니 건너뜀. 두 개를 나란히
# (첫U 125와 150, 간격 50)로 띄우면 밀린 것도 둘씩 돎(slot5060.sh가 5060 자리 둘을 나눔). 인자: RUN 첫U 끝U [간격 50]
RUN=$1; U0=$2; U1=$3; STEP=${4:-50}
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools; D=$T/../out
for UU in $(seq $U0 $STEP $U1); do
  [ $(( UU % 100 )) -eq 0 ] && continue
  C=/home/shchon11/f1sim_runs/$RUN/plan_u$UU.pt
  until [ -e $C ] || [ -e $D/one_$RUN.done ]; do sleep 30; done
  [ -e $C ] || break
  O=$D/exam/exam1_${RUN}_u${UU}_ideal_uniform_n256.npz
  for try in 1 2 3; do
    [ -e $O ] && break
    EVAL_GPU=1 bash $T/exam_judge.sh $RUN $UU
    [ -e $O ] || { echo "$(date +%T) $RUN u$UU 시험 기록 없음 (시도 $try)"; sleep 120; }
  done
  echo "$(date +%T) $RUN u$UU 시험 $([ -e $O ] && echo 끝 || echo 실패)"
done
