#!/bin/bash
# 10-27: does a policy drive the same with its plan HELD ON THE TRACK for K steps (EnvConfig.decision_every)? Korea solo,
# no props, rep floor, the checkpoint's MAP, cap 9, 16 envs x 2400 steps, seed 81 -- ppo_scan.sh's protocol, MAIN repo code.
# usage: hold_test.sh GPU NAME CKPT K
D=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research/teacher_speed
O=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research/rl_speed/out/hold_test
G=$1; N=$2; CK=$3; K=$4; F=$O/solo_${N}_k$K.json
REPO=${REPO:-/home/shchon11/F1tenth_E2E}; cd $REPO/f1sim && export PYTHONPATH=$REPO/f1sim:$REPO/f1sim_ros
CUDA_VISIBLE_DEVICES=$G OUT=$O/solo_${N}_k$K.bins.json WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u \
  $D/teacher_speed_chain.py $CK --action-mode plan --tracks real:korea_2026_competition --budget-laps 0 --teacher-a-lat 7.0 \
  --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --device cuda:0 --collision-mode soft \
  --speed-cap ${CAP:-9.0} --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 --output $F \
  --envs 16 --steps 2400 --seed 81 --race-size 1 --procedural-obstacles 0.0 --decision-every $K > $F.log 2>&1 || { echo "[$(date +%H:%M)] $N k$K FAILED"; exit 1; }
echo "[$(date +%H:%M)] $N k$K done"
