#!/bin/bash
# (11-16) 사고가 같은 자리에서 반복되고 checkpoint마다 고쳐졌다 도로 나빠지는 것(사용자: "edge case들 학습관리는 도대체 어떻게된거야?
# 왜 그냥 서서히 잊어가는거같지?")에 대한 깨끗한 비교. kr26a(Korea 2026, 처음부터) u1000에서 두 갈래, 같은 seed, 같은 차 수:
#   kr26c (4070) = 그대로 (대조)
#   kr26d (5060) = 고친 장면 재생 --scene-replay 0.25 (벽·상자·차 세게 닿기 1-3 s 전 상태에서 다시 출발; 11-16 고침: 상대 차의 제어기·
#                  운전자·이벤트·LiDAR 기록까지 되살림, scene_selftest.py로 64경주 모두 같은 길 확인) + --noise-corr-stall 0.5,0.9
#                  (0.5 s 넘게 선 차의 속도 잡음 잇기; rollout 반증: 서 있는 결정의 속도 잡음-advantage 기울기 +0.004 -> -0.015 ± 0.005)
# 판단: u1100부터 frozen exam 여러 checkpoint -- 자리별 세게 닿음(왼쪽 아래 코너 벽, 가운데 S자 상자, 오른쪽 직선 차), 2 s 넘게 선 구간과
#   풀린 방식, 혼자·빈 트랙 거리. 차이가 보이면 seed 하나 더.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
CK=/home/shchon11/f1sim_runs/kr26a/plan_u1000.pt
while [ ! -e $R/out/one_kr26a.done ]; do sleep 30; done
while pgrep -f "plan_rl --name kr26a " > /dev/null; do sleep 10; done
[ -e $CK ] || { echo "$(date +%T) $CK 없음"; exit 1; }
cd $R
echo "$(date +%T) kr26a 끝, 두 갈래 시작"
LAUNCH_GPU=0 LAUNCH_ENVS=512 bash tools/launch_run.sh kr26c korea_2026_competition 1400 $CK 1000
sleep 60
LAUNCH_GPU=1 LAUNCH_ENVS=512 bash tools/launch_run.sh kr26d korea_2026_competition 1400 $CK 1000 --scene-replay 0.25 --noise-corr-stall 0.5,0.9
echo "$(date +%T) 시작함"
