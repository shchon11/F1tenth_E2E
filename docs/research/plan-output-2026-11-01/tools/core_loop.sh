#!/bin/bash
# 핵심만 (11-14 04:00, 사용자: "평가 분석 영상은 조금만 핵심적이게만 하고"). analysis_loop10 + video_loop7 + eval_loop_plan5 + exam 반복을 대신함.
# 5060에서 한 번에 하나씩(작업 하나가 시뮬레이터를 통째로 띄워 RAM ~3 GB; 학습도 ~3 GB, 전체 14 GB).
#   25 update마다: 학습 조건 평가 (학습 표의 둘째~넷째 칸; plan_validate ideal, 학습 차 1024대 x 40 s)
#   u25 / u50 / u100 / u150, 그 뒤 50마다: frozen exam1 (판정: plan25y 같은 update와 세션끼리 짝지어; exam_judge.sh)
#   50마다: 네 경우 영상 하나 (빈 트랙 · 상대 차 · 상자 · 상자+상대; train_video.py)
#   100마다: 기억을 지울 때와 비교 (학습 표 다섯째 칸; lap_memory.py)
# 뺀 것: 제어기(MAP·시간 추종기) 평가와 그 배치 영상, 라인 비교 그림, 상자 반응 기록, 주행 기록·회복 그림, 작은·큰 트랙 평가, 가속 사용,
#   최단 시간 라인, plan 뒷부분 치우침, 시간 추종기 네 경우 영상. 필요할 때 따로 돌림.
# 인자: 이름 [시작 25] [간격 25] [끝 1000]
NAME=$1; U0=${2:-25}; STEP=${3:-25}; U1=${4:-1000}
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
A=$D/analysis; mkdir -p $A $D/videos
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
L=$A/${NAME}_summary.log
S="bash $T/slot5060.sh"
for UU in $(seq $U0 $STEP $U1); do
  C=$R/${NAME}/plan_u${UU}.pt
  [ -e $A/${NAME}_u${UU}.done ] && continue
  until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 10
  echo "== u${UU} $(date +%H:%M)" >> $L
  python3 $T/trend_plot.py $NAME $A/${NAME}_trend.png > /dev/null 2>&1 && cp $A/${NAME}_trend.png $A/${NAME}_u${UU}_trend.png
  CUDA_VISIBLE_DEVICES=1 $S timeout 2400 python3 $T/plan_validate.py $C ideal 1024 40 "  u${UU} " 2>&1 | grep "^  u${UU}" >> $D/one_${NAME}_i.log
  case " 25 50 100 150 " in *" $UU "*) EX=1 ;; *) EX=$(( UU > 150 && UU % 50 == 0 )) ;; esac
  [ "$EX" = 1 ] && bash $T/exam_judge.sh $NAME $UU
  if [ $((UU % 50)) -eq 0 ]; then
    V=$D/videos/${NAME}_u${UU}_train_4cases
    CUDA_VISIBLE_DEVICES=1 $S timeout 2400 python3 -u $T/train_video.py $C ${V}.mp4 40 256 > ${V}.log 2>&1
    echo done > ${V}.done
  fi
  if [ $((UU % 100)) -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=1 $S timeout 2400 python3 -u $T/lap_memory.py $C 256 60 > $A/${NAME}_u${UU}_lapmem.log 2>&1; grep "^요약" $A/${NAME}_u${UU}_lapmem.log >> $L
  fi
  echo done > $A/${NAME}_u${UU}.done
done
