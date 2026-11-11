#!/bin/bash
# 크기 바꾼 트랙(+scaleNNN)의 상자 배치 은행 (11-12). 인자: GPU, 출력 이름(korea25_scale_a), 트랙 이름들(쉼표)
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled BANK_SHAPES=boxes BANK_TRACKS="$3"
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
CUDA_VISIBLE_DEVICES=$1 python3 -u tools/layout_bank.py /home/shchon11/f1sim_runs/srp25sc4/state_u225.pt /home/shchon11/f1sim_runs/prop_banks_scale_tmp/$2.pt 400 32 300 > out/bank_$2.log 2>&1
echo done > out/bank_$2.done
