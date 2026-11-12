#!/bin/bash
# 4단계: 포팅본의 회피 경로가 벽에서 더 떨어지게 하면 벽 접촉이 줄어드는가 (11-12 22:05).
# 물음: 스캔 점을 장애물로 치는 기준(벽에서 0.2 m)은 그대로 두고 후보 경로의 중심만 벽에서 0.30 / 0.35 m 떨어지게 하면,
#       상자 2개에서 비켜 가는 중의 벽 접촉이 줄어드는가.
# 근거(fin_map_box2 기록): 비켜 가다 벽에 닿은 순간 차는 받은 경로 위에 있었고(벗어난 거리 중앙값 0.01~0.06 m), 접촉 지점 옆 경로 중심의
#       벽 거리는 중앙값 0.29~0.32 m(0.35 m 미만이 59~89 %), 경로 전체에서는 중앙값 0.25 m. 차체는 0.31 x 0.58 m(모서리까지 0.33 m).
#       '벽에 안 닿는 경로가 없어 벽 검사를 뺀 경로'는 접촉의 3~12 %뿐 -> 원인은 벽 검사 기준 자체.
# 바꾸는 것: AurorusConfig.path_wall_clearance_m 하나(없음 -> 0.30, 0.35). 제어기(MAP)·속도 규칙·보는 것·seed·세계는 fin_map_box2와 같음.
# 반증 조건: (1) 0.8배에서 벽 접촉(지금 1.78 /분)이 절반 아래로 내려가지 않는다 -> 경로 중심의 벽 거리가 원인이 아님.
#            (2) 상자 접촉 + 멈춰서 다시 출발이 벽 접촉이 줄어든 만큼보다 더 늘어난다 -> 길이 좁아 이 값은 못 씀.
#            (3) '벽·장애물에 안 닿는 경로가 없음' 비율이 지금(0.7~2.3 %)에서 10 % 위로 간다 -> 좁은 곳에서 후보가 모두 걸림.
S=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/b681fa17-d53e-4caa-8c76-9ce9b0410ed3/scratchpad
MP='{"t_clip_min": 0.65}'
while [ ! -e $S/aur_q5.done ]; do sleep 10; done
bash $S/aur_one.sh pw30_map_box2 B '{"follower": "map", "path_wall_clearance_m": 0.30}' "$MP" 256 40 gate
bash $S/aur_one.sh pw35_map_box2 B '{"follower": "map", "path_wall_clearance_m": 0.35}' "$MP" 256 40 gate
touch $S/aur_q6.done
