#!/bin/bash
# Korea 2026 원본 연쇄(srp26h2) 콘솔 확인 (11-02 21:15): 5060 = 라이다 학생(혼자/학습 장애물/궤적 일관성/그림 -> 30배치),
# 4070 = 참 상태 선생 propsmix(혼자/학습 장애물 -> 30배치). ×1.2 평가와 같은 조건(40 s x 30배치, 시드 777).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
M=real/korea26
(
  bash $T/check_student.sh srp26h2 $M > $D/check_student_srp26h2.log 2>&1
  bash $T/layouts_run.sh sst26 $R/sst_srp26h2_props/student_i80.pt $M 40 30 777 > $D/layouts_sst26.log 2>&1
) &
(
  source /home/shchon11/F1tenth_E2E/.venv/bin/activate
  export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
  cd /home/shchon11/F1tenth_E2E/f1sim
  C=$R/srp26h2_propsmix/state_u200.pt
  echo "혼자: $(CUDA_VISIBLE_DEVICES=0 timeout 900 python3 $T/console_laps.py $C $M 60 2>&1 | grep '주행\|Error')" > $D/solo_tch26.log
  echo "학습 장애물: $(CUDA_VISIBLE_DEVICES=0 timeout 900 python3 $T/console_laps.py $C $M 60 - 1 2>&1 | grep '주행\|Error')" >> $D/solo_tch26.log
  CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 $T/console_layouts.py $C $M 40 30 777 $D/layouts/tch26 2>&1 | grep "배치\|합계\|Error\|error" > $D/layouts_tch26.log
) &
wait
echo done > $D/check_map26.done
