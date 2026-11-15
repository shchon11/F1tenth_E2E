#!/bin/bash
# (11-15 09:45) 느린 차를 따라잡을 때 일찍 비키기/줄이기 복제 실험 (plan26r u550, 상대 = 선 따라가는 0.5-0.75배 차, 상자 없음). 4070, 학습은 그동안 멈춤.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; O=$R/out/diag1115
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
F1SIM_EXAM_SEED=20261117 BOX_REPLAY_OPP=1 BOX_REPLAY_ARMS="2,0,0,2,2,2" BOX_REPLAY_NUDGE="2.0,3" BOX_REPLAY_SLOW="0.8,10" CUDA_VISIBLE_DEVICES=0 \
  timeout 3600 python3 -u -W ignore $T/box_replay.py /home/shchon11/f1sim_runs/plan26r/plan_u550.pt $O/carnudge_26r_u550.npz 64 90 3.0 uniform > $O/carnudge_26r_u550.log 2>&1
echo "rc $?" >> $O/carnudge_26r_u550.log
cd $T && python3 -W ignore car_nudge_value.py $O/car_nudge_value_26r_u550 $O/carnudge_26r_u550.npz
