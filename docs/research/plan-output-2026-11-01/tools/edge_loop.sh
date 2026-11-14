#!/bin/bash
# edge case 표 (11-14, 사용자: "그 edge case 관리는 어떻게된거지"): frozen exam이 나올 때마다 상자 통과를 상황별로 나눈 표(exam_boxes.py, CPU만).
# 늦게 보인 상자, 같은 상자 몇 번째 통과, 지난번 세게 닿은 상자, 급하게 줄인 통과, 혼자/상대 -> analysis/RUN_uU_boxes.txt (사이트에 나옴)
# 인자: RUN
RUN=$1; R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; A=$R/out/analysis
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
cd $R/tools
while true; do
  for d in $A/${RUN}_u*_exam.done; do
    [ -e "$d" ] || continue
    U=$(basename $d | sed -E "s/${RUN}_u([0-9]+)_exam.done/\1/")
    O=$A/${RUN}_u${U}_boxes.txt
    N=$R/out/exam/exam1_${RUN}_u${U}_ideal_uniform_n256.npz
    [ -e $O ] || [ ! -e $N ] || nice -n 10 timeout 900 python3 -W ignore exam_boxes.py $N > $O 2>&1
  done
  [ -e $R/out/one_${RUN}.done ] && ! ls $A/${RUN}_u*_exam.done 2>/dev/null | while read d; do [ -e ${d%_exam.done}_boxes.txt ] || echo x; done | grep -q x && break
  sleep 120
done
