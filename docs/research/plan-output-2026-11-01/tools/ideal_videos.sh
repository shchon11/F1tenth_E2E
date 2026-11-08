#!/bin/bash
# 학습 조건(plan대로 정확히 실행) 영상 (11-08 20:10, 사용자: planner는 물리적으로 가능한 라인을 내고, 시뮬에서 제어 지연·차 상태는 예측 가능하니
# 제어기가 못 따라가는 건 지금 단계에서 매달릴 일이 아님 -> planner 자체를 보려면 학습 조건으로 몰아 봐야 함). eval_loop_plan.sh의 실차 조건 영상이
# 끝난 뒤(5060이 비었을 때) 같은 checkpoint를 콘솔 학습 조건(F1SIM_PLAN_MODE=ideal)으로 빈 트랙 40 s + 상자 8배치.
# 인자: 이름 평가GPU 첫U 간격 끝U
NAME=$1; EGPU=$2; U0=$3; STEP=$4; U1=$5
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_MODE=ideal
cd /home/shchon11/F1tenth_E2E/f1sim
for UU in $(seq $U0 $STEP $U1); do
  until [ -e $D/videos/${NAME}_u${UU}.done ] || [ -e $D/one_${NAME}.done ]; do sleep 60; done
  [ -e $D/videos/${NAME}_u${UU}.done ] || break
  C=/home/shchon11/f1sim_runs/${NAME}/plan_u${UU}.pt
  V=$D/videos/${NAME}_u${UU}_ideal
  F1SIM_VIDEO_EMPTY=1 CUDA_VISIBLE_DEVICES=$EGPU timeout 3000 python3 -u $T/layouts_video.py $C real/iccas25 40 1 777 ${V}_empty.mp4 > ${V}_empty.log 2>&1
  CUDA_VISIBLE_DEVICES=$EGPU timeout 6000 python3 -u $T/layouts_video.py $C real/iccas25 40 8 777 ${V}_8layouts.mp4 > ${V}.log 2>&1
  printf "file '%s'\nfile '%s'\n" ${V}_empty.mp4 ${V}_8layouts.mp4 > ${V}_list.txt
  ffmpeg -v error -y -f concat -safe 0 -i ${V}_list.txt -c copy ${V}_empty_8layouts.mp4
  echo done > ${V}.done
done
