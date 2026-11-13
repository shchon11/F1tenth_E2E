#!/bin/bash
# 측정한 gradient 방향으로 checkpoint를 학습 없이 옮겨 보고 frozen exam(ideal, 상자 uniform, 256)으로 채점 (11-13).
# 인자: GPU 표지 dump_dir 거리 [rollout 4-10] [steer+speed]   -> /home/shchon11/f1sim_runs/plan25y_gstep150/표지.pt, out/exam/exam1_gstep150_표지.npz
GPU=$1; LAB=$2; DUMP=$3; DIST=$4; RR=${5:-4-10}; PARTS=${6:-steer+speed}
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools
C=/home/shchon11/f1sim_runs/plan25y_gstep150/$LAB.pt; O=$R/out/exam/exam1_gstep150_$LAB
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
[ -e $O.npz ] && exit 0
python3 -W ignore $T/grad_step.py /home/shchon11/f1sim_runs/plan25y/plan_u150.pt $DUMP $C $DIST $RR $PARTS > $O.log 2>&1 || { echo "rc step" >> $O.log; exit 1; }
if [ "$GPU" = 1 ]; then
  CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 3000 python3 -u -W ignore $T/exam_record.py $C $O.npz ideal 256 uniform >> $O.log 2>&1
else
  CUDA_VISIBLE_DEVICES=$GPU timeout 3000 python3 -u -W ignore $T/exam_record.py $C $O.npz ideal 256 uniform >> $O.log 2>&1
fi
echo "rc $?" >> $O.log
