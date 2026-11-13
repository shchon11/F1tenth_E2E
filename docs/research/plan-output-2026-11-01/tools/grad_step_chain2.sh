#!/bin/bash
# (11-13 23:40) 거리 1은 차가 서 버림 -> 훨씬 작은 폭으로: 0.9 평균 방향 +0.003..+0.3, 반대로 -0.03 / -0.1
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
( bash $T/grad_step_exam.sh 1 c9_p0.03 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 0.03; bash $T/grad_step_exam.sh 1 c9_p0.01 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 0.01 ) &
( bash $T/grad_step_exam.sh 1 c9_m0.03 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 -0.03; bash $T/grad_step_exam.sh 1 c9_p0.003 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 0.003 ) &
( until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump125/done ]; do sleep 10; done; sleep 5
  bash $T/grad_step_exam.sh 0 c9_p0.1 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 0.1; bash $T/grad_step_exam.sh 0 c9_m0.1 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 -0.1; bash $T/grad_step_exam.sh 0 c9_p0.3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 0.3 ) &
wait
touch /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/gstep150_chain2.done
