#!/bin/bash
# (11-14 04:05) 지금 돌고 있는 u25 시험·영상이 끝나면 core_loop.sh plan26f 시작 (u25 학습 조건 평가부터)
while kill -0 3405562 2>/dev/null; do sleep 10; done
[ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/videos/plan26f_u25_train_4cases.mp4 ] && echo done > /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/videos/plan26f_u25_train_4cases.done
until [ -e /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/analysis/plan26f_u25_exam.done ] || ! kill -0 3372166 2>/dev/null; do sleep 10; done
exec bash /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools/core_loop.sh plan26f 25 25 1000
