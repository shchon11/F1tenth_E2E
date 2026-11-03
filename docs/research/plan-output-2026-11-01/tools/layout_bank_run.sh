#!/bin/bash
# 경로 생성기로 판정한 균일 상자 배치 은행 (11-03 밤, layout_bank.py). 인자: 체크포인트 출력 GPU 이름
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
CUDA_VISIBLE_DEVICES=$3 python3 -u tools/layout_bank.py $1 $2 400 32 300 > out/bank_$4.log 2>&1
echo done > out/bank_$4.done
