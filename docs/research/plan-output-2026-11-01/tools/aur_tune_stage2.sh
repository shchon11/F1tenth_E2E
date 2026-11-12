#!/bin/bash
# 2단계: 앞보기 거리가 속도에 따라 늘어나는 기울기(m_l1)와 지연 보상(delay_comp). 인자: 1단계에서 고른 t_clip_min
# 근거: 1단계에서 t_clip_min을 0.65 m 이하로 줄이면 0.6·0.8배에서는 하한이 더는 작동하지 않고(L1 = 0.352 v - 0.242 > 0.65 m, v > 2.5 m/s),
#       0.8배에 남은 옆 거리(0.053 m)는 빠른 완만한 굽이·곧은 구간(0.04-0.07 m)에 있음. 거기서 L1은 1.3-2 m.
# 바꾸는 것: 조향 쪽 값만(m_l1 0.25 / 0.30, delay_comp 0.03 / 0.07). 속도 쪽 값은 그대로.
# 재는 것: 받은 경로(자기 선)와의 옆 거리 RMSE, 빈 트랙·장애물 점 없음(oracle 바닥), 차 128대 x 25초, 같은 seed.
# 반증 조건: (1) m_l1을 줄여도 0.8배의 곧은·완만한 구간 RMSE가 줄지 않는다 -> 그 오차는 앞보기 거리 탓이 아님.
#            (2) m_l1을 줄이면 1.0배의 99 %가 커진다 -> 빠른 속도에서 흔들림: 줄인 값은 못 씀.
#            (3) delay_comp를 바꿔도 세 배율 RMSE 합이 0.01 m 안에서 같다 -> 지연 보상은 그대로 둠.
S=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/b681fa17-d53e-4caa-8c76-9ce9b0410ed3/scratchpad
TC=$1
while [ "$(ps -eo cmd | grep -c '[a]ur_q[23].sh')" -gt 0 ]; do sleep 5; done
for m in 0.25 0.30; do
  bash $S/aur_one.sh ml_$m A '{"follower": "map"}' "{\"t_clip_min\": $TC, \"m_l1\": $m}" 256 25 oracle
done
for d in 0.03 0.07; do
  bash $S/aur_one.sh dc_$d A '{"follower": "map"}' "{\"t_clip_min\": $TC, \"delay_comp\": $d}" 256 25 oracle
done
touch $S/aur_q4.done
