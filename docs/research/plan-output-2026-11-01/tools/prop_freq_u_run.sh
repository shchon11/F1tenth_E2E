#!/bin/bash
# 균일 배치 빈도 지도 + 통과 시험, 두 지도 (11-03 밤, prop_freq.py ... uniform), 5060
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
python3 tools/prop_freq.py /home/shchon11/f1sim_runs/srp25h3gwt_props/state_u100.pt 256 60 out/prop_freq_korea25_uniform.png "한국 2025" uniform > out/prop_freq_u.log 2>&1
python3 tools/prop_freq.py /home/shchon11/f1sim_runs/srp26gwt_props/state_u100.pt 256 60 out/prop_freq_korea26_uniform.png "한국 2026" uniform >> out/prop_freq_u.log 2>&1
touch out/prop_freq_u.done
