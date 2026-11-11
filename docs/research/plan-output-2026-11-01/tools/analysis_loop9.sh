#!/bin/bash
# plan 학습 초반 추이 분석 (plan25n부터: (6) plan 뒷부분 - 실제) (11-09, 사용자: "초반이라 추이 확인하는게 엄청 중요해보여서 시각화등등 분석 많이 해봐"): checkpoint(25 update)마다
# 5060에서 (1) 학습 추이 그림(이어 온 단계 전체, trend_plot --chain) (2) 기억 들여다보기 (3) 가속 한계 사용 (4) 같은 라인 최단 시간 (5) 빈 트랙 라인 그림.
# (11-09 v3) 무거운 단계는 5060 자리 2개(slot5060.sh)로, 끝난 checkpoint(.done)는 건너뜀.
# (11-09 v4, 사용자: "뭐이리 표가 난잡해. 지금 돌리고 있는 학습만 보여줘") 실제 차(물리 + 새 제어기) 빠른 평가를 checkpoint마다(차 128대 x 40 s,
# out/one_NAME_q.log) -- 표의 모든 행이 같은 숫자를 갖게. 추이 그림은 기본으로 지금 run만.
# (v7, 11-10) 5060이 밀려서(한 번에 하나만 도는 때가 많음) 가속 사용·같은 라인 최단·plan 뒷부분은 100 update마다만.
# (v6, 11-10 사용자: "학습 환경 평가나 잘해. 제어기까지 돌린건 서브로 두고") 표의 본 칸 = 학습 조건(plan대로 정확히) 평가,
# 환경 1024(학습 차 512대) x 40 s -> out/one_NAME_i.log. 실제 차(새 제어기)는 100 update 평가(eval_loop_plan4)에만.
# (v5) 빠른 평가를 100 update 평가의 새 제어기 조건과 같은 크기(환경 256, 학습 차 128대)로 -- 128 환경은 빈 트랙 차가 3대뿐이라
# 벽 한 번이 0.5/분으로 잡혀 시작 행과 비교가 안 됐음(u25: 세게 0.97 vs 시작 0.50). 결과: out/analysis/, 요약 out/analysis/NAME_summary.log
# (v8, 11-11 사용자 동의: 기억의 판정은 "기억을 지우면 운전이 나빠지는가") 기억 들여다보기(mem_probe: 기억이 뭘 아는가) 대신
# lap_memory(혼자·상자 계속, 기억 그대로 / 바퀴마다 지움, 환경 256 x 60 s: 512는 5060에서 약 30분이라 checkpoint 간격 50분에 안 맞음) -> 요약 줄이 표의 다섯째 칸. 표의 평가를 먼저 돌림.
# (v9, 11-11 기억은 보류: 사용자 동의) 기억 지우기 비교(lap_memory)를 뺌 -- 표의 다섯째 칸은 세게 닿은 것의 갈래(plan_table).
# 인자: 이름 시작 간격 끝 [추이 그림 시작 run]
NAME=$1; U0=$2; STEP=$3; U1=$4; ROOT=${5:-$1}
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
A=$D/analysis; mkdir -p $A
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
L=$A/${NAME}_summary.log
S="bash $T/slot5060.sh"   # at most two heavy jobs on the 5060 (11-09 out-of-memory)
for UU in $(seq $U0 $STEP $U1); do
  C=$R/${NAME}/plan_u${UU}.pt
  [ -e $A/${NAME}_u${UU}.done ] && continue
  until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 10
  echo "== u${UU} $(date +%H:%M)" >> $L
  python3 $T/trend_plot.py $NAME $A/${NAME}_trend.png --chain $ROOT > /dev/null 2>&1
  cp $A/${NAME}_trend.png $A/${NAME}_u${UU}_trend.png
  CUDA_VISIBLE_DEVICES=1 $S timeout 2400 python3 $T/plan_validate.py $C ideal 1024 40 "  u${UU} " 2>&1 | grep "^  u${UU}" >> $D/one_${NAME}_i.log
  if [ $((UU % 100)) -eq 0 ]; then CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/accel_usage.py $C 64 60 2>&1 | grep "올릴 때\|회전이 막지\|내릴 때" >> $L; fi
  if [ $((UU % 100)) -eq 0 ]; then CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/line_mintime.py $C 64 2>&1 | grep "실제 바퀴\|plan 모델 한계" >> $L; fi
  CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/line_compare.py $C ideal 32 40 $A/${NAME}_u${UU}_line.png 2>&1 | grep "png" >> $L
  if [ $((UU % 100)) -eq 0 ]; then CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/tail_bias.py $C 64 40 2>&1 | grep "plan - 실제" >> $L; fi
  echo done > $A/${NAME}_u${UU}.done
done
