#!/bin/bash
# 4070이 비면(u125 gradient 측정 끝) 5060 줄의 뒤쪽 넷을 먼저 처리 (5060 줄은 npz가 있으면 건너뜀)
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump125/done ]; do sleep 10; done
sleep 5
bash $T/grad_step_exam.sh 0 c9_r4_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 3 4
bash $T/grad_step_exam.sh 0 c9_r9_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 3 9
bash $T/grad_step_exam.sh 0 w_r9_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150 3 9
bash $T/grad_step_exam.sh 0 w_r7_p3 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150 3 7
touch /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/gstep150_chain4070.done
