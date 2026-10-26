#!/bin/bash
# 10-27 RL research, H (clean signal): in races the hard-contact cost of the traffic swings the returns and buries the small gain of a
# faster straight; solo, the same objective should climb faster. = the race recipe (v2PH_run.sh / ppo_boot_v2rep.sh: reward terms,
# lambda held, lr, horizon, minibatch, MAP Korea, rep floor, dial held, props on the line) with race-size 1 and 64 envs, so an update
# has the same 2048 learner samples. Falsifier: the straights climb no faster per update than the race PPO's (lambda 20: +0.18 /
# +0.36 / +0.49 m/s at 64 / 128 / 256 updates over the entry's 6.38).
# usage: GPU=.. ppo_solo_v2same.sh RUN INIT SEED UPDATES LAMBDA
set -u
RUN=$1; INIT=$2; SEED=$3; UPD=$4; LAMB=$5
OUT=~/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline; T=$OUT/tools; WT=${WT:-/data/wt/ft}
RUNNER=${RUNNER:-$OUT/tools/research/rl_speed/grad_snr.py}
MAPJ=$(cat ${MAPJ_FILE:-$T/out/map_k26_round5.json}); MAP=${MAP:-real:korea_2026_competition}
TRACKS="$MAP"; for k in $(seq 0 15); do TRACKS="$TRACKS,$MAP+warp$k"; done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
cd "$WT/f1sim" && export PYTHONPATH=$WT/f1sim:$WT/f1sim_ros:${PYTHONPATH:-} TORCHINDUCTOR_COMPILE_THREADS=2 CUDA_VISIBLE_DEVICES=${GPU:-0} WANDB_MODE=offline
echo "[$(date +%H:%M:%S)] $RUN start (code $(git -C $WT rev-parse --short HEAD), init $INIT, lambda $LAMB)" >> "$OUT/$RUN.log"
python3 -u $RUNNER --arch v2 --name "$RUN" --init "$INIT" --cond dial --grip-budget-penalty 2.0 \
   --tracks "$TRACKS" --envs ${ENVS:-64} --epochs 3 --total $((UPD * ${ENVS:-64} * 32)) \
   --gamma 0.997 --cap0 ${CAP:-9} --cap1 ${CAP:-9} --cap-steps 5e6 --episode-s 40 \
   --collision-penalty 1 --collision-speed-penalty 0 --steer-penalty 0.02 --proximity-penalty 0 --sideslip-penalty 0 \
   --lap-bonus 5 --lap-time-bonus 6 --plan-clearance-penalty 0 --wrong-way-penalty 0.4 \
   --lagrange "wall=1.0" --contact-severity-min 1.5 --lam 0.97 --lagrange-init $LAMB --lagrange-lr 0 \
   --procedural-obstacles ${PROPS:-1.0} --procedural-density 1.5 --procedural-max-props 18 --procedural-raceline-corridor off \
   --spawn-runway 3.0 --collision-mode soft --movable-obstacles \
   --raceline-objective min_time --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 \
   --critic-warmup 10 --grad-accum 1 --minibatch 1024 --lr ${LR:-1e-5} --lr-end ${LR_END:-${LR:-1e-5}} --horizon 32 --dial-constant 1.0489 \
   --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 --plan-follower map --plan-follower-map "$MAPJ" \
   --amp --sim-backend graphs --save-every 64 --seed "$SEED" --wandb offline ${SOLO_EXTRA:-} >> "$OUT/$RUN.log" 2>&1
echo "[$(date +%H:%M:%S)] $RUN exit $?" | tee -a "$OUT/$RUN.log"
