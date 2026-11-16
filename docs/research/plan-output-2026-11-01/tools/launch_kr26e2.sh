#!/bin/bash
# (11-16 14:05, 사용자: "더 보고싶은데 1400보다") kr26e(장면 재생만 0.25, 4070)는 --updates 1400으로 시작해 그 숫자를 돌고 있는 학습에서 바꿀 수
# 없음 -> u1400에서 스스로 끝나면 같은 인자로 그 checkpoint에서 이어 kr26e2를 u1800까지(update 번호 이어 셈; optimizer만 새로 시작).
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
until [ -e $R/out/one_kr26e.done ]; do sleep 30; done
while pgrep -f "plan_rl --name kr26e " > /dev/null; do sleep 5; done
cd $R
EVAL_GPU=0 LAUNCH_GPU=0 LAUNCH_ENVS=512 bash tools/launch_run.sh kr26e2 korea_2026_competition 1800 /home/shchon11/f1sim_runs/kr26e/plan_u1400.pt 1400 --scene-replay 0.25
setsid nohup bash tools/eval_latest_only.sh kr26e2 > out/eval_latest_only_kr26e2.out 2>&1 < /dev/null &
setsid nohup bash tools/lab_watch_sync.sh kr26e2 > out/lab_watch_sync_e2.log 2>&1 < /dev/null &
echo "$(date +%T) kr26e2 시작"
