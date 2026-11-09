#!/bin/bash
# 11-09 22:50: 상대 차 영상 반복을 5060 자리 2개 버전(video_loop4)으로 바꿈 -- 지금 만드는 u50 MAP 영상이 끝나면 옛 반복(video_loop3)을 멈추고
# u75부터 새 반복, 그 다음 지금 학습의 시작점(plan25n u100) 분석(start_battery2: 추이 표의 시작 행).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
OLD=3587739
until [ -e $D/videos/plan25n2_u50_map_opp.done ]; do sleep 2; done
kill -STOP $OLD; CH=$(ps -o pid= --ppid $OLD); kill -9 $OLD; for c in $CH; do kill $c 2>/dev/null; done
setsid nohup bash $T/video_loop4.sh plan25n2 75 25 500 > $D/video_loop4_plan25n2.out 2>&1 &
setsid nohup bash $T/start_battery2.sh plan25n2 /home/shchon11/f1sim_runs/plan25n/plan_u100.pt "plan25n u100" > $D/start_battery2_plan25n2.out 2>&1 &
echo "switched $(date +%T)"
