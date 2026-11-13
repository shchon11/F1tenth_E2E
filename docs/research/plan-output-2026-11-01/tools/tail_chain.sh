#!/bin/bash
# (11-14 06:45) plan26f u100 분석 뒤: 대조군 8 update -> 시험 셋(plan26f u75, 뒷부분=실제 길 u8, 대조군 u8) -> plan과 실제로 간 길 거리
until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/analysis/plan26f_u100.done ]; do sleep 30; done
bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/tail_ctl.sh
for x in "plan26f 75" "plan26f_tailchk75 8" "plan26f_tailctl75 8"; do bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/exam_judge.sh $x; done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
E=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam
python3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/plan_vs_driven.py "plan26f u75 (출발점)=$E/exam1_plan26f_u75_ideal_uniform_n256.npz" "+8 뒷부분 원래대로=$E/exam1_plan26f_tailctl75_u8_ideal_uniform_n256.npz" "+8 뒷부분 실제로 간 길=$E/exam1_plan26f_tailchk75_u8_ideal_uniform_n256.npz" > /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/tail_check/result.txt 2>&1
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools && python3 -W ignore grad_step_table.py $E/exam1_plan26f_u75_ideal_uniform_n256.npz "+8 뒷부분 원래대로=$E/exam1_plan26f_tailctl75_u8_ideal_uniform_n256.npz" "+8 뒷부분 실제로 간 길=$E/exam1_plan26f_tailchk75_u8_ideal_uniform_n256.npz" >> /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/tail_check/result.txt 2>&1
touch /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/tail_check/chain.done
