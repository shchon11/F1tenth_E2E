#!/bin/bash
# (11-16) 사용자: "아예 처음부터 scratch에서 지금 e f 돌리고 있는 세팅으로 갈까?" -> kr26g = 처음부터 + 장면 재생 0.25를 처음부터(사용자 안),
# 4070. 짝: kr26h = 처음부터 + 장면 재생 0.1을 u150부터(5060, 13:52). 대조 = kr26a(처음부터, 장면 재생 없음; 같은 인자·seed·차 수).
# kr26e(4070)의 u1300 frozen exam이 끝나면 kr26e를 멈추고 시작.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
until [ -e $R/out/analysis/kr26e_u1300_exam.done ]; do sleep 30; done
P=$(pgrep -f "plan_rl --name kr26e " | head -1); [ -n "$P" ] && kill $P
while pgrep -f "plan_rl --name kr26e " > /dev/null; do sleep 5; done
cd $R
EVAL_GPU=0 LAUNCH_GPU=0 LAUNCH_ENVS=512 bash tools/launch_run.sh kr26g korea_2026_competition 1000 "" 0 --scene-replay 0.25
setsid nohup bash tools/eval_latest_only.sh kr26g > out/eval_latest_only_kr26g.out 2>&1 < /dev/null &
setsid nohup bash tools/lab_watch_sync.sh kr26g > out/lab_watch_sync_g.log 2>&1 < /dev/null &
echo "$(date +%T) kr26g 시작"
