#!/bin/bash
# follow_probe.py를 one-world 최종(2025 2단계 u100)에 (11-04 04:45): 기각(미리 정함, 0.8배) -- 앞 3~6 m 시간 > 15 % 또는 그때 속도/raceline < 0.9
# 또는 나란히(±1 m) 속도 < 0.6. 기존 race checkpoint u50은 33 %, 0.79, 0.33. 4070, one_srp25oww.done 뒤.
until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/one_srp25oww.done ]; do sleep 60; done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
for X in 0.8 1.0; do
  CUDA_VISIBLE_DEVICES=0 timeout 2400 python3 -u tools/follow_probe.py /home/shchon11/f1sim_runs/srp25oww/state_u100.pt real/iccas25 30 12 forzaeth $X 777 > out/follow_probe_oww_u100_x$X.log 2>&1
done
echo done > out/follow_probe_ow.done
