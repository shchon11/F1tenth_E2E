#!/bin/bash
# 10-26 19:00: WHICH VESC SETTING MATTERS? The competition car's speed loop costs the entry ~4 % lap time and +33 % hard contacts (cap
# 2.5 m/s^2 + 0.1 s delay, braking default). Split it: cap only (RAMP 2.5, DELAY 0) vs delay only (RAMP 0, DELAY 0.1), no limits,
# props on, 81-82 races + solo. usage: vesc_split.sh GPU cap|delay
R=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools/research
CK=/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar_slow.pt
GPU=$1; MODE=$2
if [ $MODE = cap ]; then export RAMP=2.5 DELAY=0.0; else export RAMP=0.0 DELAY=0.1; fi
export TAU=""
O=$R/out/vesc_$MODE; mkdir -p $O
cd /home/shchon11/F1tenth_E2E/f1sim
COMMON="--action-mode plan --tracks real:korea_2026_competition --envs 32 --steps 2400 --budget-laps 0 --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 --raceline-objective min_time --dial-offset 0.0 --speed-cap 9.0 --device cuda:0 --graphs --collision-mode soft --movable-obstacles --spawn-runway 3.0 --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 --procedural-raceline-corridor off --dial-value 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92"
for S in 81 82; do for K in forzaeth forzaeth_pred lane_switch interactive; do
  F=$O/race_s${S}_${K}.json; [ -f $F ] && continue
  CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u $R/realspeed_eval.py $CK $COMMON \
    --seed $S --race-size 2 --opp-slots "[{\"kind\": \"$K\", \"speed_scale\": [0.7, 1.0]}]" --output $F > $F.log 2>&1 || { echo "[$(date +%H:%M)] $MODE $S $K FAILED"; exit 1; }
done; done
for S in 81 82; do
  F=$O/solo_s${S}.json; [ -f $F ] && continue
  CUDA_VISIBLE_DEVICES=$GPU WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 /home/shchon11/F1tenth_E2E/.venv/bin/python -u $R/realspeed_eval.py $CK $COMMON \
    --seed $S --race-size 1 --output $F > $F.log 2>&1 || { echo "[$(date +%H:%M)] $MODE solo $S FAILED"; exit 1; }
done
echo "[$(date +%H:%M)] vesc $MODE done"
