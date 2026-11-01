#!/bin/bash
# 실제 콘솔 GUI(창 + SessionController + 워커 프로세스)로 추천 하나를 학습 장애물에서 몰고(세션 시드 기록),
# 같은 시드를 console_laps(워커를 직접 호출)와 props_pic으로 다시 몰아 비교 (11-02, 5060).
# 인자: 추천 이름 체크포인트 맵 초 출력머리
PICK=$1; C=$2; MAP=$3; SECS=$4; OUT=$5
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
L=$(CUDA_VISIBLE_DEVICES=1 timeout 600 python3 /home/shchon11/F1tenth_E2E/docs/research/console-redesign-2026-11-02/console_shot.py $OUT $SECS "" $PICK 2>&1 | grep "세션 설정\|찍음\|Error")
echo "GUI: $L"
S=$(echo "$L" | grep "세션 설정" | awk '{print $4}')
echo "도구(같은 시드 $S): $(CUDA_VISIBLE_DEVICES=1 timeout 600 python3 $T/console_laps.py $C $MAP $SECS - 1 $S 2>&1 | grep '주행\|Error')"
CUDA_VISIBLE_DEVICES=1 timeout 600 python3 $T/props_pic.py $C $MAP $SECS $S ${OUT}_props.png 2>&1 | grep "그림\|Error"
