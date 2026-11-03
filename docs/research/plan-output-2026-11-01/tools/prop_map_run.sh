#!/bin/bash
# 학습 때 뽑히는 상자 위치 분포 두 지도 (11-03, prop_map.py), 5060
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
python3 tools/prop_map.py /home/shchon11/f1sim_runs/srp25h3gwt_props/state_u100.pt 256 40 out/prop_map_korea25.png "한국 2025" > out/prop_map.log 2>&1
python3 tools/prop_map.py /home/shchon11/f1sim_runs/srp26gwt_props/state_u100.pt 256 40 out/prop_map_korea26.png "한국 2026" >> out/prop_map.log 2>&1
touch out/prop_map.done
