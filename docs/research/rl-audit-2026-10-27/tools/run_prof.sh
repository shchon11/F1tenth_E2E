#!/bin/bash
# usage: GPU=0 ENVS=256 WARM=40 MEAS=40 run_prof.sh [extra ppo args...]
SP=$(cd $(dirname $0) && pwd)
T=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools; C=/home/shchon11/F1tenth_E2E/checkpoints; RR=~/f1sim_runs
INIT=${INIT:-$RR/v2PH_s2421/ppo_u512.pt}
MAPJ=$(cat $T/out/map_k26_round5.json); MAP=real:korea_2026_competition
TAG=v2PHc80_s2438
LEAGUE="[\"$RR/${TAG}_seats/seat0.pt\",\"$RR/${TAG}_seats/seat1.pt\",\"$C/iccas_constrained_v2R.pt\",\"$C/iccas_constrained_safe_s976.pt\",\"$C/iccas_league_c_s949.pt\",\"$RR/ffR_lin/ppo_final.pt\",\"$RR/spec_korea_c2new_s947/ppo_final.pt\",\"$RR/spec_korea_contact2_s942/ppo_final.pt\",\"$RR/spec_korea_contact_s915/ppo_u768.pt\"]"
SEATS="$RR/${TAG}_seats/seat0.pt,$RR/${TAG}_seats/seat1.pt"
TRACKS="$MAP"; for k in $(seq 0 15); do TRACKS="$TRACKS,$MAP+warp$k"; done
SLOT='{"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"],"speed_scale":[0.7,1.05],"ckpt_mix":'"$LEAGUE"',"ckpt_p":0.5,"present":0.5,"spawn":"random","reactive":{"defend":0.3,"line":0.2,"oblivious":0.1},"events":["brake","stop","shift","weave"],"event_rate":0.5}'
OPP_SLOTS="${OPP_SLOTS:-[$SLOT,$SLOT,$SLOT]}"
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
cd /home/shchon11/F1tenth_E2E/f1sim && export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros TORCHINDUCTOR_COMPILE_THREADS=2 CUDA_VISIBLE_DEVICES=${GPU:-0}
python3 $SP/prof_step.py ${WARM:-40} ${MEAS:-40} -- --arch v2 --name prof_tmp --init "$INIT" --cond dial \
   --grip-budget-penalty 0 \
   --tracks "$TRACKS" --envs ${ENVS:-256} --epochs 3 --total 1e6 \
   --gamma 0.997 --cap0 10 --cap1 10 --cap-steps 5e6 --episode-s 40 \
   --collision-penalty 1 --collision-speed-penalty 0 --steer-penalty 0.02 \
   --proximity-penalty 0 --sideslip-penalty 0 \
   --lap-bonus 0 --lap-time-bonus 6 --plan-clearance-penalty 0 --wrong-way-penalty 0.4 \
   --race-size ${RACE:-4} --opp-slots "$OPP_SLOTS" --rear-end-penalty 0 --merge-penalty 0 \
   --overtake-bonus 0 --car-proximity-penalty 0 --car-contact-penalty 1 --lagrange wall=1.0,car=1.0 --contact-severity-min 1.5 --lam 0.97 \
   --lagrange-init 80 --lagrange-lr 0 --lagrange-max 300 \
   --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 18 \
   --procedural-raceline-corridor off --spawn-runway 3.0 \
   --collision-mode soft --movable-obstacles \
   --raceline-objective min_time --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 \
   --level-replay 0.5 --league-slots "$SEATS" --league-every 64 \
   --grad-accum 1 --minibatch 1024 --lr 2e-5 --lr-end 1e-5 --horizon 32 \
   --dial-constant 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 \
   --plan-follower map --plan-follower-map "$MAPJ" \
   --critic-warmup 10 --amp --sim-backend graphs --save-every 1000 --seed 7 --wandb disabled "$@"
