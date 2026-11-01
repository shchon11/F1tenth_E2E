#!/bin/bash
# 학생 하나 콘솔 확인 (5060): 혼자 60 s, 학습 장애물 60 s, 궤적 일관성, 주행 그림. 인자: 머리 맵 [시작 시각 이후의 완료 파일만 인정]
# (11-02 04:54 교훈: 죽인 실행이 남긴 완료 파일 때문에 학습 중인 학생을 확인하려 했음 -> 완료 파일이 학습 시작보다 새것인지 확인)
TAG=$1; MAP=$2
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
R=~/f1sim_runs/sst_${TAG}_props
until [ -f $D/sst_${TAG}_props.done ] && [ $D/sst_${TAG}_props.done -nt $R/args.json ]; do sleep 60; done; sleep 10
C=$(ls -t $R/student_i*.pt | head -1)
echo "== 학생 $TAG ($(basename $C)) @ $MAP"
echo "혼자: $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py $C $MAP 60 2>&1 | grep '주행\|Error')"
echo "학습 장애물: $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py $C $MAP 60 - 1 2>&1 | grep '주행\|Error')"
CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/plan_consistency.py $C $MAP 30 2>&1 | grep -v Warning | tail -3
CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/drive_pic.py $C $MAP 20 $D/../drive_sst_${TAG}_props.png 2>&1 | grep "그림\|Error"
echo done > $D/check_student_${TAG}.done
