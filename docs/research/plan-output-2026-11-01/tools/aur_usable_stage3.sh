#!/bin/bash
# 3단계: 고른 MAP 설정으로 포팅본을 상대 차로 쓸 수 있는지 재는 주행. 인자: 고른 MAP 덮어쓰기 JSON (예: '{"t_clip_min": 0.65}')
# 물음: 포팅본이 자기 LiDAR만으로(바닥 점은 자기 자이로로 거름) MAP 제어기로 달릴 때, 빈 트랙과 상자 2개에서 선 속도의 몇 배까지 깨끗하게 도는가.
#       우리 policy(plan25x u200, plan대로 정확히 실행)는 같은 세계에서 빈 트랙 7.40초·접촉 0, 상자 2개 7.50초·상자 세게 0.14 /분.
# 바꾸는 것: 따라가는 제어기만(원본 조향 <-> MAP + 고른 설정). 경로 고르기·속도 규칙·차 보정·실제 물리는 같음. 제어기 설정은 옆 거리 RMSE로만 골랐고
#            여기 숫자(한 바퀴, 접촉)는 설정을 고르는 데 쓰지 않음.
# 재는 것: 실제 지도, 차 128대 x 40초, 같은 seed, 배율 0.6 / 0.8 / 1.0. 빈 트랙과 상자 2개(시험 배치).
# 반증 조건: (1) 0.8배·1.0배에서 MAP의 벽 접촉이 원본 조향보다 줄지 않는다 -> 벽 접촉은 따라가기 탓이 아님.
#            (2) 상자 2개에서 비켜 가는 중의 벽 접촉이 원본과 같거나 많다 -> 원인은 경로 고르기(경로 중심만 벽에서 0.2 m, 차 반폭 0.155 m).
#            (3) 어느 배율에서도 한 바퀴가 우리 policy보다 1.0초 안쪽으로 들어오면서 세게 닿음이 0.2 /분 아래인 것이 없다 -> 제어기를 바꿔도 우리 속도의 상대 차로는 못 씀.
S=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/b681fa17-d53e-4caa-8c76-9ce9b0410ed3/scratchpad
MP="$1"
while [ "$(ps -eo cmd | grep -c '[a]ur_q[234].sh')" -gt 0 ]; do sleep 5; done
bash $S/aur_one.sh fin_map_empty  A '{"follower": "map"}' "$MP" 256 40 gate
bash $S/aur_one.sh fin_map_box2   B '{"follower": "map"}' "$MP" 256 40 gate
bash $S/aur_one.sh fin_orig_empty A '{}' '{}' 256 40 gate
bash $S/aur_one.sh fin_orig_box2  B '{}' '{}' 256 40 gate
touch $S/aur_q5.done
