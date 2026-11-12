#!/bin/bash
# 4단계 뒤 확인: 후보 경로 중심을 벽에서 0.30 m 떼는 값이 빈 트랙 주행을 해치지 않는가 (11-12 22:52).
# 근거: 상자 2개에서 0.30 m는 반증 셋을 다 통과(0.8배 벽 접촉 1.78 -> 0.59 /분, 상자 접촉·멈춤 안 늘고, 안 닿는 경로 없음 0.7~1.2 %).
# 물음: 빈 트랙(fin_map_empty: 0.8배 8.80 s, 벽 0.14 /분, 비켜 가는 중 5 %)에서 한 바퀴 시간과 벽 접촉이 그대로인가.
# 반증: 0.8배에서 한 바퀴가 0.2 s 넘게 느려지거나 벽 접촉이 0.14 /분보다 늘면, 이 값은 상자가 있을 때만 이득 -> 설정에 넣기 전에 다시 봄.
S=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/b681fa17-d53e-4caa-8c76-9ce9b0410ed3/scratchpad
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
MP='{"t_clip_min": 0.65}'
while [ ! -e $R/out/aurorus/pw35_map_box2.done ]; do sleep 10; done
bash $S/aur_one.sh pw30_map_empty A '{"follower": "map", "path_wall_clearance_m": 0.30}' "$MP" 256 40 gate
