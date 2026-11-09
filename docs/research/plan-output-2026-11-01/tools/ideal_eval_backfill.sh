#!/bin/bash
# 11-10 01:25 (사용자: "학습 환경 평가나 잘해. 제어기까지 돌린건 서브로 두고"): 표의 본 칸 = 학습 조건(plan대로 정확히 실행) 평가를 넉넉히
# (환경 1024 = 학습 차 512대 x 40 s; 256 환경은 빈 트랙 차가 12대뿐). 이미 지난 시작점·u25·u50을 같은 크기로 다시 잼.
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools; D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
cd /home/shchon11/F1tenth_E2E/f1sim
for job in "plan25n2 100" "plan25n3 25" "plan25n3 50"; do
  set -- $job
  CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 $T/plan_validate.py $R/$1/plan_u$2.pt ideal 1024 40 "  u$2 " 2>&1 | grep "^  u$2" >> $D/one_$1_i.log
done
echo done > $D/ideal_backfill.done
