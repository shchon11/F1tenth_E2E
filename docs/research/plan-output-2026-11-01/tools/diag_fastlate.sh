#!/bin/bash
# (11-15 03:05) '빠르게 들어가다 늦게 보인 상자' 칸이 왜 안 움직이나: (가) 4070: plan26l u200 시험을 policy 입력(스캔·기억)까지 기록 ->
#   box_react.py(상자 빔을 지운 스캔으로 plan을 다시 계산: 상자 때문에 plan이 비킨 거리·줄인 속도, 언제부터). (나) 5060: 같은 차 복제 +
#   2도 x 0.3 s 일찍 비키기(box_replay.py L/R 갈래) -> nudge_value.py(상자까지 거리별: 일찍 비키면 덜 닿나, 보상으로 이득인가).
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; O=$R/out/diag1115; mkdir -p $O
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
CK=/home/shchon11/f1sim_runs/plan26l/plan_u200.pt
cd /home/shchon11/F1tenth_E2E/f1sim
( F1SIM_EXAM_SEED=20261115 BOX_REPLAY_ARMS="2,0,0,2,2" BOX_REPLAY_NUDGE="2.0,3" CUDA_VISIBLE_DEVICES=1 timeout 5400 python3 -u -W ignore $T/box_replay.py $CK $O/nudge_26l_u200.npz 64 90 3.0 uniform > $O/nudge_26l_u200.log 2>&1
  echo "rc $?" >> $O/nudge_26l_u200.log
  cd $T && python3 -W ignore nudge_value.py $O/nudge_26l_u200.npz > $O/nudge_value_26l_u200.txt 2>&1; touch $O/nudge.done ) &
F1SIM_EXAM_SAVE_MEM=1 F1SIM_EXAM_SAVE_SCAN=1 CUDA_VISIBLE_DEVICES=0 timeout 2400 python3 -u -W ignore $T/exam_record.py $CK $O/react_26l_u200.npz ideal 256 uniform > $O/react_rec.log 2>&1
echo "rc $?" >> $O/react_rec.log
cd $T
EXAM_ROWS=$O/react_boxrows.npy python3 -W ignore exam_boxes.py $O/react_26l_u200.npz > $O/react_boxes.txt 2>&1
CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 -W ignore box_react.py $CK $O/react_26l_u200.npz $O/box_react_26l_u200 > $O/box_react_26l_u200.txt 2>&1
touch $O/react.done
wait
