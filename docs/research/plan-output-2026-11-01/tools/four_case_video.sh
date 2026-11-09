#!/bin/bash
# 한 영상에 네 경우 (11-10, 사용자: "그 영상은 학습환경하고 제어기 붙인걸로 각각 뽑아주고, 한 영상에 네 경우(빈, 장애물 있다없어진,
# +상대차)다 들어가게 만들어"): 1 혼자 빈 트랙 / 2 혼자 장애물 -> 20 s에 치움 / 3 상대 차 + 빈 트랙 / 4 상대 차 + 장애물 -> 20 s에 치움,
# 각 40 s를 한 mp4로. 장애물 = 시드 777의 첫 배치(왼쪽 위 U턴 입구 상자 있음, 지금까지 영상과 같음), 상대 차 = 학습 세계 종류 0.8배 속도.
# 인자: checkpoint 모드(ideal = 학습 조건 | track = 물리 차 + 새 제어기) 출력mp4. 접촉 기록은 출력 이름.log
C=$1; M=$2; OUT=$3
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_MODE=$M CUDA_VISIBLE_DEVICES=1
cd /home/shchon11/F1tenth_E2E/f1sim
B=${OUT%.mp4}
S="bash $T/slot5060.sh"
L=$( [ "$M" = ideal ] && echo "학습 조건" || echo "새 제어기" )
F1SIM_VIDEO_CASE="1/4 혼자 · 빈 트랙 · $L" F1SIM_VIDEO_EMPTY=1 \
  $S timeout 3000 python3 -u $T/layouts_video.py $C real/iccas25 40 1 777 ${B}_c1.mp4 > ${B}_c1.log 2>&1
F1SIM_VIDEO_CASE="2/4 혼자 · 장애물 → 20 s에 치움 · $L" F1SIM_VIDEO_REMOVE_AT=20 \
  $S timeout 3000 python3 -u $T/layouts_video.py $C real/iccas25 40 1 777 ${B}_c2.mp4 > ${B}_c2.log 2>&1
F1SIM_VIDEO_CASE="3/4 상대 차 · 빈 트랙 · $L" F1SIM_VIDEO_EMPTY=1 F1SIM_VIDEO_OPP=0.8 \
  $S timeout 3000 python3 -u $T/layouts_video.py $C real/iccas25 40 1 777 ${B}_c3.mp4 > ${B}_c3.log 2>&1
F1SIM_VIDEO_CASE="4/4 상대 차 · 장애물 → 20 s에 치움 · $L" F1SIM_VIDEO_REMOVE_AT=20 F1SIM_VIDEO_OPP=0.8 \
  $S timeout 3000 python3 -u $T/layouts_video.py $C real/iccas25 40 1 777 ${B}_c4.mp4 > ${B}_c4.log 2>&1
printf "file '%s'\n" ${B}_c1.mp4 ${B}_c2.mp4 ${B}_c3.mp4 ${B}_c4.mp4 > ${B}_list.txt
ffmpeg -v error -y -f concat -safe 0 -i ${B}_list.txt -c copy $OUT
grep -h "배치" ${B}_c1.log ${B}_c2.log ${B}_c3.log ${B}_c4.log > ${B}.log
