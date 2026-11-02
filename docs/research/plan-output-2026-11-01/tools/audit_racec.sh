#!/bin/bash
# 상대 단계 학습 세계 진단 (11-02 13:20): racec u25(J 높음) / u100(J 낮음) / 출발점과 같은 u0 대용 pmcst 없이 -- 같은 세계, 같은 시드, 탐색 잡음
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E
L=$D/out/audit_racec.log
: > $L
for U in 25 100; do
  CUDA_VISIBLE_DEVICES=1 python3 $D/tools/race_world_audit.py $R/srp25h2_racec/state_u$U.pt 512 40 4242 $D/out/audit_racec_u$U 1 2>&1 | grep -v Warning >> $L
done
echo done > $D/out/audit_racec.done
