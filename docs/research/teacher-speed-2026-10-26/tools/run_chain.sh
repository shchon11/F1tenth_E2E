#!/bin/bash
# usage: run_chain.sh GPU|cpu NAME <extra evaluate args...>   -> out/NAME.txt, out/NAME.json
D=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research/teacher_speed
G=$1; N=$2; shift 2
mkdir -p $D/out
if [ "$G" = cpu ]; then export CUDA_VISIBLE_DEVICES=""; DEV=cpu; else export CUDA_VISIBLE_DEVICES=$G; DEV=cuda:0; fi
cd /home/shchon11/F1tenth_E2E/f1sim
OUT=$D/out/$N.json WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u $D/teacher_speed_chain.py \
  --action-mode plan --tracks real:korea_2026_competition --envs 4 --steps 1600 --budget-laps 0 \
  --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 \
  --device $DEV --seed 81 --collision-mode soft --race-size 1 --procedural-obstacles 0.0 --output $D/out/$N.eval.json \
  --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 "$@" > $D/out/$N.log 2>&1
echo "exit $?" >> $D/out/$N.log
grep -v -i warn $D/out/$N.log | grep -E "^steps|^profile|lap_time|exit|preview model" > $D/out/$N.txt
