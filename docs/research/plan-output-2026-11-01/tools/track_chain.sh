#!/bin/bash
# 4070: 지금 학습 방식 그대로 8 update, 이어서 잡음 이음 0.7로 8 update (둘 다 u150에서, 측정용)
until grep -q "^rc" /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/exam1_gstep150_c9steer_p0.1.log 2>/dev/null; do sleep 10; done
sleep 5
bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/track_steps.sh 0 plan25y_trk150 0 8
bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/track_steps.sh 0 plan25y_trk150_c7 0.7 8
touch /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/track/chain.done
