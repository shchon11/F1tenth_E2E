#!/bin/bash
# (11-17) kr26k = 처음부터 (사용자: "자꾸 했던거에서 재시작할래? Bias가 존나 남아있을거아냐", "가 나 다를 다 해야지", "구조적으로 깔끔하게.
# 덕지덕지 무언가를 안놓고 rl로 깔끔하게 해결"). 이어 받지 않고, 덧붙인 보상·안전장치 없이 RL로:
#   (가) --scene-keep-unsolved: 장면 보관함이 '깨끗하게 통과한 횟수가 가장 많은 장면'부터 비움 -> 계속 박는 장면은 해결될 때까지 남음
#        (지금까지: 오래된 것부터 지워 한 장면이 ~26 update, ~12번 해 보고 해결 못 한 채 사라짐; 재생의 42-46 %가 다시 박음)
#   (나) --scene-episode-s 13: 장면 에피소드가 같은 자리를 한 바퀴 뒤에 한 번 더 지나감(박기 1-3 s 전 + 한 바퀴 7.6-9 s + 여유)
#        -> '앞 바퀴에 박은 것을 기억해 다음 바퀴에 안 박음'이 학습에서 보상받음 (지금: 앞 바퀴에 박은 상자를 다음 바퀴에 또 박는 비율 33-73 %)
#   (다) --mem-keep-s 30 --mem-keep-log: 기억(GRU)이 오래 남도록 시작(값마다 0.2 s .. 30 s, 반은 짧게) -- kr26j u2000은 128개 중 1 s 넘게
#        남는 값 1개, 지운 기억과의 차이가 1.6 s에 반으로 줄어 한 바퀴를 못 넘김
#   + kr26j의 상대 차 환경(상대 차 경주 상자 1-3개·상대 차 라인 밖, 배율 0.9-1.1, 복제본 30 %를 u200부터, 최고 속도 6.5-9.0 m/s),
#     update당 결정 수 자동(최대 3배), 장면 재생 25 %(장면은 u100부터 저장)
# 4070 혼자 512대(kr26e3·kr26j와 같은 크기). frozen exam은 100마다 4070, 상대 차 시험은 100마다 5060(oppexam_loop.sh).
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
LAUNCH_GPU=0 LAUNCH_ENVS=512 bash $T/launch_run.sh kr26k korea_2026_competition 1500 "" 0 \
  --mem-keep-s 30 --mem-keep-log \
  --scene-replay 0.25 --scene-keep-unsolved --scene-episode-s 13 --scene-from 100 --horizon-auto 3 \
  --opp-box-corridor --opp-box-keep 3 --opp-scale 0.9,1.1 --self-opp 0.3 --self-opp-from 200 --self-opp-vcap 6.5 9.0
