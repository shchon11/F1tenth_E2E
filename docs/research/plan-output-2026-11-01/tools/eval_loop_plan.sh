#!/bin/bash
# plan_rl 평가 반복 (11-08, 사용자: "validation도 평가 환경에다가 보상값만 쭈르륵 뽑으면 빠른거 아냐?", "그냥 200도 내리고 ㄱㄱ"):
# STEP update마다 (1) 학습 세계 그대로 고정 시드로 차 수백 대를 한꺼번에(plan_validate.py) 학습 조건 / 실차 조건, (2) 영상(콘솔: 빈 트랙
# 40 s + 상자 8배치). 콘솔로 한 대씩 돌리던 빈 트랙·배치 평가는 뺌. 인자: 이름 평가GPU 시작 간격 끝 [차 수]
NAME=$1; EGPU=$2; U0=$3; STEP=$4; U1=$5; NE=${6:-256}
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/one_${NAME}.log
for UU in $(seq $U0 $STEP $U1); do
  C=$R/${NAME}/plan_u${UU}.pt
  until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 10
  for M in ideal map; do
    CUDA_VISIBLE_DEVICES=$EGPU timeout 3000 python3 $T/plan_validate.py $C $M $NE 40 "  u${UU} " 2>&1 | grep "^  u${UU}" >> $L
  done
  echo done > $D/val_${NAME}_u${UU}.done
  V=$D/videos/${NAME}_u${UU}
  F1SIM_VIDEO_EMPTY=1 CUDA_VISIBLE_DEVICES=$EGPU timeout 3000 python3 -u $T/layouts_video.py $C real/iccas25 40 1 777 ${V}_empty.mp4 > ${V}_empty.log 2>&1
  CUDA_VISIBLE_DEVICES=$EGPU timeout 6000 python3 -u $T/layouts_video.py $C real/iccas25 40 8 777 ${V}_8layouts.mp4 > ${V}.log 2>&1
  printf "file '%s'\nfile '%s'\n" ${V}_empty.mp4 ${V}_8layouts.mp4 > ${V}_list.txt
  ffmpeg -v error -y -f concat -safe 0 -i ${V}_list.txt -c copy ${V}_empty_8layouts.mp4
  echo done > ${V}.done
done
