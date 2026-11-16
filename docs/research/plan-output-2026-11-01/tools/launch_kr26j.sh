#!/bin/bash
# (11-16) kr26j = kr26e3의 끝(u1800)에서 '상대 차 환경 바로잡기' 하나만 바꿔 이어감 (4070, kr26e3과 같은 크기: 512대, update당 결정 수 자동 3배):
#   - 상대 차가 있는 경주: 상자 1-3개, 상대 차 레이싱 라인 띠 밖에만 (--opp-box-keep 3 --opp-box-corridor)
#     근거: 상대 차가 상자 하나당 분당 ~0.55번 박고 다시 출발 2.66/분 (상자 없음 0.07, 띠 비움 0.06)
#   - 상대 차 속도 배율 0.9-1.1 (--opp-scale; 우리 차의 0.7-0.82배 = 보통 팀; 지금 0.4-1.2는 64 %가 0.7배 미만, 1.1 넘으면 벽에 박음)
#   - 상대 차의 30 %는 우리 policy 복제본, 최고 속도 6.5-9.0 m/s에서 뽑음 (--self-opp 0.3 --self-opp-vcap 6.5 9.0; 우리 차 0.85-1.0배
#     = 최상위 팀; 혼자 빈 트랙 90 % 속도 7.35, 최대 8.9 m/s)
#   - 갈래 첫 10 update 학습률 k/10 (--lr-warmup 10; 첫 update KL 0.179 -> 0.0074)
# 시험(frozen exam)은 세계 설정 그대로(상대 차 0.4-1.2, 상자 제한 없음)라 kr26e3과 같은 세션끼리 비교됨.
# kr26e3이 끝날 때(plan_rl 프로세스가 없어지고 plan_u1800.pt가 있을 때)까지 기다렸다가 시작.
R=/home/shchon11/f1sim_runs; T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
while pgrep -f "^python3 -u -m f1sim.learn.plan_rl --name kr26e3 " > /dev/null; do sleep 60; done
[ -e $R/kr26e3/plan_u1800.pt ] || { echo "$(date +%T) kr26e3 u1800 없음 -- 시작 안 함"; exit 1; }
sleep 30
LAUNCH_GPU=0 LAUNCH_ENVS=512 bash $T/launch_run.sh kr26j korea_2026_competition 2300 $R/kr26e3/plan_u1800.pt 1800 \
  --scene-replay 0.25 --horizon-auto 3 --lr-warmup 10 --opp-box-corridor --opp-box-keep 3 --opp-scale 0.9,1.1 \
  --self-opp 0.3 --self-opp-from 1801 --self-opp-vcap 6.5 9.0
