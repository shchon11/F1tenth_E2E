#!/bin/bash
# 상자 있을 때 선생 진단 (11-03): Korea 2025 선생(4070), Korea 2026 원본 선생(5060), 30배치 x 40 s, 시드 777(콘솔 평가와 같은 배치)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 $T/teacher_props_probe.py ~/f1sim_runs/pick_srp25h2/state_latest.pt real/iccas25 40 30 777 $D/tprobe25 2>&1 | grep -v "Warn\|warn\|ret = \|print(" > $D/tprobe25.log &
CUDA_VISIBLE_DEVICES=1 timeout 3600 python3 $T/teacher_props_probe.py ~/f1sim_runs/pick_srp26h2/state_latest.pt real/korea26 40 30 777 $D/tprobe26 2>&1 | grep -v "Warn\|warn\|ret = \|print(" > $D/tprobe26.log &
wait
echo done > $D/tprobe.done
