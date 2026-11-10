#!/bin/bash
# 정기 영상 v7 (11-11, 사용자: "영상도 학습환경 기준으로뽑아"): checkpoint(STEP update)마다
#   1) train_video.py: 표와 같은 학습 세계(원래 + 변형 트랙, 학습 상자 배치, 상대 차, 치우는 경주), 학습 조건, 고정 시드, 네 경우 한 mp4
#   2) four_case_video.sh track: 물리 차 + 새 제어기(콘솔 세계, 원래 트랙, 시드 777 배치) -- 제어기 쪽은 따로 봄
# 결과 out/videos/NAME_uUU_train_4cases.mp4(.log), NAME_uUU_track_4cases.mp4. 인자: 이름 시작 간격 끝
NAME=$1; U0=$2; STEP=$3; U1=$4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
for UU in $(seq $U0 $STEP $U1); do
  C=$R/${NAME}/plan_u${UU}.pt
  until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 10
  V=$D/videos/${NAME}_u${UU}_train_4cases
  if [ ! -e ${V}.done ]; then
    (cd /home/shchon11/F1tenth_E2E/f1sim && CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 -u $T/train_video.py $C ${V}.mp4 40 256 > ${V}.log 2>&1)
    echo done > ${V}.done
  fi
  V=$D/videos/${NAME}_u${UU}_track_4cases
  if [ ! -e ${V}.done ]; then
    bash $T/four_case_video.sh $C track ${V}.mp4
    echo done > ${V}.done
  fi
done
