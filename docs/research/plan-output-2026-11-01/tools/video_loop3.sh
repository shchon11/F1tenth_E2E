#!/bin/bash
# 초반 추이용 영상 (11-09, 사용자: "시각화들 싹다 어디갔어 영상들이랑"): checkpoint(STEP update)마다 5060에서 물리 차 영상 --
# 새 제어기(F1SIM_PLAN_MODE=track)와 MAP 두 가지, 상대 차 1대(레이싱라인을 0.8배 속도로, 사용자 11-09: "영상들에서 상대차는 어디다
# 팔아먹은거야?") + 빈 트랙 40 s + 상자 4배치(시드 777). 결과 out/videos/NAME_uUU_{track,map}_empty_4layouts.mp4
# 인자: 이름 시작 간격 끝
NAME=$1; U0=$2; STEP=$3; U1=$4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
for UU in $(seq $U0 $STEP $U1); do
  C=$R/${NAME}/plan_u${UU}.pt
  until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
  [ -e $C ] || break
  for M in track map; do
    # one heavy job at a time on the 5060 (11-09: three at once ran the 14 GB machine out of memory)
    while pgrep -f "plan_[v]alidate.py|mem_[p]robe.py|accel_[u]sage.py|line_[m]intime.py|line_[c]ompare.py|tail_[b]ias.py" > /dev/null; do sleep 20; done
    V=$D/videos/${NAME}_u${UU}_${M}_opp
    F1SIM_VIDEO_OPP=0.8 F1SIM_PLAN_MODE=$M F1SIM_VIDEO_EMPTY=1 CUDA_VISIBLE_DEVICES=1 timeout 3000 python3 -u $T/layouts_video.py $C real/iccas25 40 1 777 ${V}_empty.mp4 > ${V}_empty.log 2>&1
    F1SIM_VIDEO_OPP=0.8 F1SIM_PLAN_MODE=$M CUDA_VISIBLE_DEVICES=1 timeout 4000 python3 -u $T/layouts_video.py $C real/iccas25 40 4 777 ${V}_4layouts.mp4 > ${V}.log 2>&1
    printf "file '%s'\nfile '%s'\n" ${V}_empty.mp4 ${V}_4layouts.mp4 > ${V}_list.txt
    ffmpeg -v error -y -f concat -safe 0 -i ${V}_list.txt -c copy ${V}_empty_4layouts.mp4
    echo done > ${V}.done
  done
done
