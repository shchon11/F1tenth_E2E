#!/bin/bash
# plan 학습 초반 추이 분석 (plan25n부터: (6) plan 뒷부분 - 실제) (11-09, 사용자: "초반이라 추이 확인하는게 엄청 중요해보여서 시각화등등 분석 많이 해봐"): checkpoint(25 update)마다
# 5060에서 차례로 (1) 학습 추이 그림 (2) 기억 들여다보기 (3) 가속 한계 사용 (4) 같은 라인 최단 시간 (5) 빈 트랙 라인 그림.
# 평가 반복(plan_validate·영상)이 5060을 쓰는 동안은 기다림. 결과: out/analysis/, 요약 out/analysis/NAME_summary.log
# 인자: 이름 시작 간격 끝
NAME=$1; U0=$2; STEP=$3; U1=$4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
A=$D/analysis; mkdir -p $A
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
L=$A/${NAME}_summary.log
for UU in $(seq $U0 $STEP $U1); do
  C=$R/${NAME}/plan_u${UU}.pt
  until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 10
  while pgrep -f "plan_[v]alidate.py|layouts_[v]ideo.py" > /dev/null; do sleep 20; done
  echo "== u${UU} $(date +%H:%M)" >> $L
  python3 $T/trend_plot.py $NAME $A/${NAME}_trend.png plan25b2 0 > /dev/null 2>&1
  cp $A/${NAME}_trend.png $A/${NAME}_u${UU}_trend.png
  CUDA_VISIBLE_DEVICES=1 timeout 1500 python3 -u $T/mem_probe.py $C $A/${NAME}_u${UU}_mem.png 60 256 2>&1 | grep "맞힘" >> $L
  CUDA_VISIBLE_DEVICES=1 timeout 1200 python3 -u $T/accel_usage.py $C 64 60 2>&1 | grep "올릴 때\|회전이 막지\|내릴 때" >> $L
  CUDA_VISIBLE_DEVICES=1 timeout 1200 python3 -u $T/line_mintime.py $C 64 2>&1 | grep "실제 바퀴\|plan 모델 한계" >> $L
  CUDA_VISIBLE_DEVICES=1 timeout 1200 python3 -u $T/line_compare.py $C ideal 32 40 $A/${NAME}_u${UU}_line.png 2>&1 | grep "png" >> $L
  CUDA_VISIBLE_DEVICES=1 timeout 1200 python3 -u $T/tail_bias.py $C 64 40 2>&1 | grep "plan - 실제" >> $L
  echo done > $A/${NAME}_u${UU}.done
done
