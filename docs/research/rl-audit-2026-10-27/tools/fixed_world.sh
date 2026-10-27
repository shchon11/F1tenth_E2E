#!/bin/bash
# 2026-10-27 rl-audit: does PPO improve its OWN objective? A checkpoint driven FROZEN (actor lr 0, with its training noise) in the
# rlc training world held fixed: the same league files for every checkpoint (seat0 = v2PH u512, seat1 = iccas safest, never
# refreshed), no self-play seat, no level replay (both make the world follow the policy under test). Everything else is rl_clean.sh.
# The objective is computed by fixed_world_j.py from progress.jsonl: progress + steer + wrong-way - 80 x charged hard onsets.
# usage: GPU=0 [UPD=40] fixed_world.sh NAME CKPT SEED [extra ppo args]
set -u
NAME=$1; CK=$2; SEED=$3; shift 3
T=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools; C=/home/shchon11/F1tenth_E2E/checkpoints; RR=~/f1sim_runs
MAPJ=$(cat $T/out/map_k26_round5.json); MAP=real:korea_2026_competition
S=$RR/fixedworld_seats; mkdir -p $S
[ -f $S/seat0.pt ] || cp $RR/v2PH_s2421/ppo_u512.pt $S/seat0.pt; [ -f $S/seat1.pt ] || cp $C/iccas_bootcamp_safest.pt $S/seat1.pt
LEAGUE="[\"$S/seat0.pt\",\"$S/seat1.pt\",\"$C/iccas_constrained_v2R.pt\",\"$C/iccas_constrained_safe_s976.pt\",\"$C/iccas_league_c_s949.pt\",\"$RR/ffR_lin/ppo_final.pt\",\"$RR/spec_korea_c2new_s947/ppo_final.pt\",\"$RR/spec_korea_contact2_s942/ppo_final.pt\",\"$RR/spec_korea_contact_s915/ppo_u768.pt\"]"
TRACKS="$MAP"; for k in $(seq 0 15); do TRACKS="$TRACKS,$MAP+warp$k"; done
SLOT='{"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"],"speed_scale":[0.7,1.05],"ckpt_mix":'"$LEAGUE"',"ckpt_p":0.5,"present":0.5,"spawn":"random","reactive":{"defend":0.3,"line":0.2,"oblivious":0.1},"events":["brake","stop","shift","weave"],"event_rate":0.5}'
OUT=$(cd $(dirname $0)/.. && pwd)/out; mkdir -p $OUT
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
cd /home/shchon11/F1tenth_E2E/f1sim && export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros TORCHINDUCTOR_COMPILE_THREADS=2 CUDA_VISIBLE_DEVICES=${GPU:-0}
ENVS=${ENVS:-1024}; LR_ROWS=$((ENVS / 4))
echo "[$(date +%H:%M:%S)] $NAME fixed-world start ($CK, seed $SEED, $(git -C /home/shchon11/F1tenth_E2E rev-parse --short HEAD))" >> $OUT/$NAME.log
# ENTRY: another entry point taking the same argv (e.g. ENTRY="$D/hit_anticipation.py 2400 --"); default the PPO trainer
python3 -u ${ENTRY:--m f1sim.learn.ppo} --arch v2 --name "$NAME" --init "$CK" --cond dial \
   --tracks "$TRACKS" --envs $ENVS --epochs 1 --total $(( ${UPD:-40} * LR_ROWS * 64 )) \
   --horizon 64 --minibatch 4096 --grad-accum 2 --lr 0 --lr-end 0 --critic-lr 3e-4 \
   --gamma 0.997 --lam 0.97 --cap0 10 --cap1 10 --cap-steps 5e6 --episode-s 40 \
   --collision-penalty 1 --collision-speed-penalty 0 --steer-penalty 0.02 --proximity-penalty 0 --sideslip-penalty 0 \
   --lap-bonus 0 --lap-time-bonus 0 --grip-budget-penalty 0 --plan-clearance-penalty 0 --wrong-way-penalty 0.4 \
   --race-size 4 --opp-slots "[$SLOT,$SLOT,$SLOT]" --rear-end-penalty 0 --merge-penalty 0 \
   --overtake-bonus 0 --car-proximity-penalty 0 --car-contact-penalty 1 --lagrange wall=1.0,car=1.0 --contact-severity-min 1.5 \
   --lagrange-init 80 --lagrange-lr 0 --lagrange-max 300 \
   --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 18 --procedural-raceline-corridor off --spawn-runway 3.0 \
   --collision-mode soft --movable-obstacles \
   --raceline-objective min_time --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 \
   --level-replay 0 --league-slots "" \
   --dial-constant 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 \
   --plan-follower map --plan-follower-map "$MAPJ" \
   --pg-dead-dims 3,5,6,7 \
   --critic-warmup 1 --amp --sim-backend graphs --save-every 100000 --seed "$SEED" --wandb offline \
   "$@" >> $OUT/$NAME.log 2>&1
echo "[$(date +%H:%M:%S)] $NAME exit $?" | tee -a $OUT/$NAME.log
