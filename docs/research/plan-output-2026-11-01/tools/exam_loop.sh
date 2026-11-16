#!/bin/bash
# (11-16, 사용자: "평가는 그 충돌맵으로 100 iter마다 해", "각자 gpu에서 하게 하자") 평가 = frozen exam(같은 128세션)과 그 충돌 지도만,
# 100 update마다, 그 run이 학습하는 GPU에서(0 = 4070 / slot4070.sh, 1 = 5060 / slot5060.sh). 학습 조건 평가·기억 비교·영상은 안 함.
# 시험이 GPU 메모리 부족 등으로 기록 없이 끝나면 2분 뒤 다시(세 번까지). 인자: RUN 첫U 끝U GPU
RUN=$1; U0=$2; U1=$3; G=$4
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; D=$R/out; A=$D/analysis
RUNS=/home/shchon11/f1sim_runs
for UU in $(seq $U0 100 $U1); do
  C=$RUNS/$RUN/plan_u$UU.pt
  until [ -e $C ] || [ -e $D/one_$RUN.done ]; do sleep 30; done
  [ -e $C ] || break
  O=$D/exam/exam1_${RUN}_u${UU}_ideal_uniform_n256.npz
  for try in 1 2 3; do
    [ -e $O ] && break
    python3 -c "import os; p='$A/${RUN}_u${UU}_exam.done'; os.path.exists(p) and os.remove(p)"
    EVAL_GPU=$G bash $T/exam_judge.sh $RUN $UU
    [ -e $O ] || { echo "$(date +%T) $RUN u$UU 시험 기록 없음 (시도 $try)"; sleep 120; }
  done
  echo "$(date +%T) $RUN u$UU 시험 $([ -e $O ] && echo 끝 || echo 실패)"
done
