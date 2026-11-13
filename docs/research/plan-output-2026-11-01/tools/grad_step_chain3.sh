#!/bin/bash
# (11-13 23:45) 0.9 평균 방향 +0.1이 나아짐 -> 대조군: 지금 학습 방식(결정마다 새 잡음)의 평균 방향 / rollout 하나의 방향, 무작위 방향,
# 0.9의 조향 부분만 / 속도 부분만, 0.7의 평균 방향. 모두 같은 폭 0.1
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
( until grep -q "^rc" /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/exam1_gstep150_c9_p0.3.log 2>/dev/null; do sleep 10; done
  bash $T/grad_step_exam.sh 0 w_p0.1 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150 0.1
  bash $T/grad_step_exam.sh 0 rand1_p0.1 random:1:/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150 0.1
  bash $T/grad_step_exam.sh 0 c9steer_p0.1 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 0.1 4-10 steer ) &
( until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/gstep150_chain2.done ]; do sleep 10; done
  ( bash $T/grad_step_exam.sh 1 w_r4_p0.1 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150 0.1 4; bash $T/grad_step_exam.sh 1 c9speed_p0.1 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c9 0.1 4-10 speed ) &
  ( bash $T/grad_step_exam.sh 1 c7_p0.1 /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150_c7 0.1; bash $T/grad_step_exam.sh 1 rand2_p0.1 random:2:/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/batchdump/plan25y_gdump150 0.1 ) &
  wait ) &
wait
touch /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/gstep150_chain3.done
