#!/bin/bash
# plan 학습 분석 반복 v10 (11-12, plan25x부터). analysis_loop9에 상황별 측정을 더함 (사용자: "Edge case 관리. 기억 활용 등등 내가 말했던
# 것들 하나하나 잊지마"): checkpoint마다 5060에서(무거운 것은 slot5060.sh로 한 번에 둘까지)
#   25 update마다: 학습 추이 그림, 표의 평가(학습 세계 17개 트랙, 고정 시드, 학습 차 512대 x 40 s -> out/one_NAME_i.log), 빈 트랙 주행 그림
#   50 update마다: 상자 통과를 상황별로(늦게 보인 상자 / 라인을 크게 바꿔야 하는 상자 / 급하게 피한 통과 / 지난 바퀴에 닿은 상자)
#                  -> out/edge_cases_NAME_uUU.txt (box_response 기록 out/rec/resp_NAME_uUU.npz)
#   100 update마다: 기억을 바퀴마다 지우면 나빠지나(lap_memory) -> out/analysis/NAME_uUU_lapmem.log,
#                  세게 닿은 뒤 다시 달리기까지(drive_record -> recovery) -> out/analysis/NAME_uUU_recovery.txt(.png),
#                  크기 바꾼 트랙에서의 평가(작은 6개 / 큰 5개, 학습 차 256대씩) -> out/one_NAME_scale.log,
#                  가속 한계 사용, 같은 라인 최단 시간, plan 뒷부분 - 실제
# LiDAR 자리(F1SIM_LIDAR_MOUNT_X)와 타이어 한계(F1SIM_PLAN_LAT_MARGIN)는 실행 스크립트가 export한 것을 그대로 씀.
# 인자: 이름 시작 간격 끝 [추이 그림 시작 run]
NAME=$1; U0=$2; STEP=$3; U1=$4; ROOT=${5:-$1}
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
A=$D/analysis; mkdir -p $A $D/rec
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
L=$A/${NAME}_summary.log
S="bash $T/slot5060.sh"   # at most two heavy jobs on the 5060 (11-09 out-of-memory)
P=real:korea_2025_iccas
SMALL="$P+warp0+scale060,$P+warp7+scale063,$P+warp14+scale065,$P+warp5+scale068,$P+warp12+scale071,$P+warp3+scale074"
LARGE="$P+warp13+scale108,$P+warp4+scale111,$P+warp11+scale114,$P+warp2+scale117,$P+warp9+scale120"
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
  CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/line_compare.py $C ideal 32 40 $A/${NAME}_u${UU}_line.png 2>&1 | grep "png" >> $L
  if [ $((UU % 50)) -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=1 $S timeout 3000 python3 -u $T/box_response.py $C $D/rec/resp_${NAME}_u${UU}.npz 1024 40 > $D/rec/resp_${NAME}_u${UU}.log 2>&1
    python3 $T/edge_cases.py $D/rec/resp_${NAME}_u${UU}.npz > $D/edge_cases_${NAME}_u${UU}.txt 2>&1
  fi
  if [ $((UU % 100)) -eq 0 ]; then
    CUDA_VISIBLE_DEVICES=1 $S timeout 2400 python3 -u $T/lap_memory.py $C 256 60 > $A/${NAME}_u${UU}_lapmem.log 2>&1; grep "^요약" $A/${NAME}_u${UU}_lapmem.log >> $L
    CUDA_VISIBLE_DEVICES=1 $S timeout 2400 python3 -u $T/drive_record.py $C $D/rec/drv_${NAME}_u${UU}.npz 512 40 > $D/rec/drv_${NAME}_u${UU}.log 2>&1
    python3 $T/recovery.py $D/rec/drv_${NAME}_u${UU}.npz $A/${NAME}_u${UU}_recovery.png > $A/${NAME}_u${UU}_recovery.txt 2>&1
    F1SIM_WORLD_TRACKS="$SMALL" CUDA_VISIBLE_DEVICES=1 $S timeout 2400 python3 $T/plan_validate.py $C ideal 512 40 "  u${UU} 작은 트랙(x0.58-0.75) " 2>&1 | grep "^  u${UU}" >> $D/one_${NAME}_scale.log
    F1SIM_WORLD_TRACKS="$LARGE" CUDA_VISIBLE_DEVICES=1 $S timeout 2400 python3 $T/plan_validate.py $C ideal 512 40 "  u${UU} 큰 트랙(x1.11-1.19) " 2>&1 | grep "^  u${UU}" >> $D/one_${NAME}_scale.log
    CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/accel_usage.py $C 64 60 2>&1 | grep "올릴 때\|회전이 막지\|내릴 때" >> $L
    CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/line_mintime.py $C 64 2>&1 | grep "실제 바퀴\|plan 모델 한계" >> $L
    CUDA_VISIBLE_DEVICES=1 $S timeout 1200 python3 -u $T/tail_bias.py $C 64 40 2>&1 | grep "plan - 실제" >> $L
  fi
  echo done > $A/${NAME}_u${UU}.done
done
