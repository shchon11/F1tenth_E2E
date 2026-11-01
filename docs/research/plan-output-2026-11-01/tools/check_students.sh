#!/bin/bash
# 학생(라이다 격자, 혼자 + 상자 선생) 콘솔 확인 (5060): 혼자 60 s, 학습 장애물 60 s, 궤적 일관성, 주행 그림
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
for X in "srp25h2 real/iccas25" "srp26xh2 real/korea26x12"; do
  set -- $X
  until [ -f $D/sst_$1_props.done ]; do sleep 60; done; sleep 10
  C=$(ls -t ~/f1sim_runs/sst_$1_props/student_i*.pt | head -1)
  echo "== 학생 $1 ($(basename $C)) @ $2"
  echo "혼자: $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py $C $2 60 2>&1 | grep '주행\|Error')"
  echo "학습 장애물: $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py $C $2 60 - 1 2>&1 | grep '주행\|Error')"
  CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/plan_consistency.py $C $2 30 2>&1 | grep -v Warning | tail -3
  CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/drive_pic.py $C $2 20 $D/../drive_sst_$1_props.png 2>&1 | grep "그림\|Error"
done
echo done > $D/check_students.done
