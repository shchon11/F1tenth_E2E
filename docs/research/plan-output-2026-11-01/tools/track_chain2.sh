#!/bin/bash
# 4070 (00:17): 지금 설정 8 update(돌고 있음)가 끝나면 -> 같은 것을 한 epoch에 한 걸음(16 minibatch 합침)으로 8 update -> 잡음 이음 0.7로 8 update
until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/track/plan25y_trk150/done ]; do sleep 10; done
sleep 5
python3 -c "import os; os.replace('/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/track_steps.sh.new', '/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/track_steps.sh')"
bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/track_steps.sh 0 plan25y_trk150_acc16 0 8 16
bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/track_steps.sh 0 plan25y_trk150_c7 0.7 8 1
touch /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/track/chain2.done
