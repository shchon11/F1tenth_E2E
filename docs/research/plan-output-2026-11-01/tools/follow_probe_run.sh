#!/bin/bash
# follow_probe.py: 기존 race checkpoint u50 vs forzaeth 0.6배, obstacle 없음 (11-04 04:00), 5060
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
CUDA_VISIBLE_DEVICES=1 timeout 2400 python3 -u tools/follow_probe.py /home/shchon11/f1sim_runs/srp25h3gwt_race/state_u50.pt real/iccas25 30 12 forzaeth 0.6 777 > out/follow_probe_race_u50.log 2>&1
echo done > out/follow_probe_race_u50.done
