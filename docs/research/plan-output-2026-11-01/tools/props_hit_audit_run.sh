#!/bin/bash
# 상자 충돌을 "처음 보인 뒤 몇 초 만인가"로 나눔 (11-03): srp25h3 u200 대 이어 학습 srp25h3b u200, 학습 세계, 평균 행동
cd /home/shchon11/F1tenth_E2E/f1sim
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
for r in srp25h3_props srp25h3b_props; do
  CUDA_VISIBLE_DEVICES=0 python3 -u ../docs/research/plan-output-2026-11-01/tools/props_hit_audit.py /home/shchon11/f1sim_runs/$r/state_u200.pt 512 120 5
done > $O/props_hit_audit_h3b.log 2>&1
touch $O/props_hit_audit_h3b.done
