#!/bin/bash
# 5060에서 둘씩: 0.9 평균 방향 -3/-1/+1/+3, 지금 학습 방식 평균 +3, rollout 하나의 방향(지금 학습 방식 4·7·9번, 0.9 4·9번) +3
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
run() { bash $T/grad_step_exam.sh 1 "$@"; }
( run c9_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 3; run c9_m3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 -3; run w_r4_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150 3 4; run w_r9_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150 3 9; run c9_r4_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 3 4 ) &
( run c9_p1 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 1; run c9_m1 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 -1; run w_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150 3; run w_r7_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150 3 7; run c9_r9_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 3 9 ) &
wait
touch /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/gstep150_chain.done
