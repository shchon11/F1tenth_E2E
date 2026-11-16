#!/bin/bash
# (11-16 15:58, 사용자: "흠 e 정체된것 같아. update당 결정 수를 유동적으로 늘릴 수 있게 만들어야할 것 같아") kr26e2(장면 재생 0.25, 4070)를
# 다음 checkpoint(u1475)에서 같은 인자 + --horizon-auto 3 으로 이어 kr26e3 (u1800까지). 근거: 잡음 크기 ~58k 결정 > rollout 32.8k 결정
# (actor가 rollout 하나에 한 걸음으로 묶여 있음). K는 3까지(4070 12 GB: 학습 + 시험 3.4 GB).
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; C=/home/shchon11/f1sim_runs/kr26e2/plan_u1475.pt
until [ -e $C ]; do sleep 15; done
sleep 20
P=$(pgrep -f "plan_rl --name kr26e2 " | head -1); [ -n "$P" ] && kill $P
while pgrep -f "plan_rl --name kr26e2 " > /dev/null; do sleep 5; done
cd $R
EVAL_GPU=0 LAUNCH_GPU=0 LAUNCH_ENVS=512 bash tools/launch_run.sh kr26e3 korea_2026_competition 1800 $C 1475 --scene-replay 0.25 --horizon-auto 3
setsid nohup bash tools/lab_watch_sync.sh kr26e3 > out/lab_watch_sync_e3.log 2>&1 < /dev/null &
echo "$(date +%T) kr26e3 시작"
