#!/bin/bash
# 11-09 시뮬 물리 실측 반영(서보 0.04 s, 명령 지연 0.03 s, 가속 한계 5.0 / 7 m/s) 뒤, 같은 조건 비교용으로 plan25b2 u100을 새 물리로 다시
# 평가(학습 조건 / 새 제어기 / MAP). 결과는 out/one_plan25b2_newphys.log
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
C=/home/shchon11/f1sim_runs/plan25b2/plan_u100.pt
for M in ideal track map; do
  CUDA_VISIBLE_DEVICES=1 timeout 3000 python3 $T/plan_validate.py $C $M 256 40 "  u100 새물리 " 2>&1 | grep "^  u100" >> $D/one_plan25b2_newphys.log
done
echo done > $D/val_plan25b2_u100_newphys.done
