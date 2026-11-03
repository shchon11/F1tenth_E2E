#!/bin/bash
# 은행 세계에서 박는 상자에 정책이 반응하나 (11-04 02:10, crate_cf.py): 2025 재학습 u50, 콘솔 30배치 중 나쁜 판 1,19,26,28 (시드 777)
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
CUDA_VISIBLE_DEVICES=1 timeout 2400 python3 -u tools/crate_cf.py /home/shchon11/f1sim_runs/srp25bk_props/state_u50.pt real/iccas25 1,19,26,28 777 > out/crate_cf_bank25.log 2>&1
echo done > out/crate_cf_bank25.done
