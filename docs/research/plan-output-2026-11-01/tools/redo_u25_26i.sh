#!/bin/bash
# (11-14 11:40) plan26i u25 학습 조건 평가와 시험이 5060 메모리 부족으로 빠짐(제 측정이 같은 때 5060을 써서) -> u50 분석이 끝나면 다시
until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/analysis/plan26i_u50.done ]; do sleep 30; done
L=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/.slot5060; exec 8>$L.0; flock 8; exec 9>$L.1; flock 9
python3 -c "
import os
for f in ('/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/analysis/plan26i_u25.done', '/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/analysis/plan26i_u25_exam.done', '/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/analysis/plan26i_u25_exam.txt', '/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/exam/exam1_plan26i_u25_ideal_uniform_n256.log'):
    if os.path.exists(f): os.remove(f)
"
exec 8>&- 9>&-
CORE_VAL_ENVS=512 bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/core_loop.sh plan26i 25 25 25
