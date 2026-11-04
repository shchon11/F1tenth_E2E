#!/bin/bash
# follow_probe.py 2차 (11-04 04:15): 0.6배는 금방 추월해 '멀리서 따라가기'가 0.1 %뿐 -> forzaeth 0.8 / 1.0배, gating별로 나눔. 5060
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
for X in 0.8 1.0; do
  CUDA_VISIBLE_DEVICES=1 timeout 2400 python3 -u tools/follow_probe.py /home/shchon11/f1sim_runs/srp25h3gwt_race/state_u50.pt real/iccas25 30 12 forzaeth $X 777 > out/follow_probe_race_u50_x$X.log 2>&1
done
echo done > out/follow_probe2.done
