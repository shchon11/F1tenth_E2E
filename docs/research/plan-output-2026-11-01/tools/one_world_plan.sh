#!/bin/bash
# 재설계 planner 학습(11-07, 사용자와 단계마다 합의): f1sim.learn.plan_rl -- plan = 점 10개 x 0.1 s(걸음마다 꺾기·속도, 차 한계 안),
# 학습은 plan대로 정확히 실행 + 시뮬 접촉, 보상 = 랩타임 + 결승선 정산 중간 점수(plan 점 10개) + 접촉·plan 충돌 감점, 입력 = LiDAR 3장 +
# 속도·핸들·자이로 + 직전 plan. 평가(5060)는 합의한 그대로 두 조건으로: 100 update마다 실차 조건(물리 + MAP) 빈 트랙 60 s(콘솔), 상자 8배치
# (콘솔), plan 품질과 두 조건 차이(scripted_eval: ideal / map x 빈 트랙 / 상자), 영상(빈 트랙 + 상자 8배치); 200마다 상자 30배치.
# 실패 기준(학습 전 합의, u800까지, 실차 조건): 0.3 s 점 뒤집힘 < 0.3/s, 0.5 s 점 plan과 실제 < 30 cm, 타이어 한계 넘는 plan < 5 %,
# 0.3 s 안 벽·상자 지나가는 plan < 2 % / 빈 트랙 <= 7.79 s, 30배치 멈춤 <= 13 %·세게 <= 8, 1:1 지금보다 나쁘지 않게 / 두 조건 랩 차이 < 5 %.
# 인자: 이름 학습GPU 평가GPU 업데이트 [시드]
NAME=$1; GPU=$2; EGPU=$3; U=$4; S=${5:-1}
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
L=$D/one_${NAME}.log
echo "== ${NAME}: 재설계 planner(plan_rl), 처음부터" > $L
( for UU in $(seq 100 100 $U); do
    C=$R/${NAME}/plan_u${UU}.pt
    until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
    [ -e $C ] || break
    sleep 10
    echo "  u${UU} 빈 트랙(실차 조건): $(CUDA_VISIBLE_DEVICES=$EGPU timeout 900 python3 $T/console_laps.py $C real/iccas25 60 2>&1 | grep '주행' | cut -c1-200)" >> $L
    echo "  u${UU} 상자 8배치(실차 조건): $(CUDA_VISIBLE_DEVICES=$EGPU timeout 1800 python3 $T/console_layouts.py $C real/iccas25 40 8 777 $D/layouts/q_${NAME}_u${UU} 2>&1 | grep '합계' | cut -c1-200)" >> $L
    for M in map ideal; do for PR in 0 1; do
      CUDA_VISIBLE_DEVICES=$EGPU timeout 1500 python3 $T/scripted_eval.py $M 32 40 $PR $C 2>&1 | grep "^\[" | sed "s/^/  u${UU} /" >> $L
    done; done
    if [ $((UU % 200)) = 0 ]; then
      echo "  u${UU} 상자 30배치(실차 조건): $(CUDA_VISIBLE_DEVICES=$EGPU timeout 7200 python3 $T/console_layouts.py $C real/iccas25 40 30 777 $D/layouts/o_${NAME}_u${UU} 2>&1 | grep '합계' | cut -c1-200)" >> $L
    fi
    V=$D/videos/${NAME}_u${UU}
    F1SIM_VIDEO_EMPTY=1 CUDA_VISIBLE_DEVICES=$EGPU timeout 3000 python3 -u $T/layouts_video.py $C real/iccas25 40 1 777 ${V}_empty.mp4 > ${V}_empty.log 2>&1
    CUDA_VISIBLE_DEVICES=$EGPU timeout 6000 python3 -u $T/layouts_video.py $C real/iccas25 40 8 777 ${V}_8layouts.mp4 > ${V}.log 2>&1
    printf "file '%s'\nfile '%s'\n" ${V}_empty.mp4 ${V}_8layouts.mp4 > ${V}_list.txt
    ffmpeg -v error -y -f concat -safe 0 -i ${V}_list.txt -c copy ${V}_empty_8layouts.mp4
    echo done > ${V}.done
    echo done > $D/one_${NAME}_u${UU}.done
  done ) &
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.plan_rl --name ${NAME} --updates $U --seed $S --save-every 25 > $D/${NAME}.log 2>&1 \
  || { echo fail > $D/one_${NAME}.done; exit 1; }
wait
echo done > $D/one_${NAME}.done
