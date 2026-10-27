#!/bin/bash
# 2026-10-27 rl-audit: the race PPO with the measured fixes and nothing else changed. From the entry's PPO base (v2PH_s2421 u512),
# Korea + 16 warps, races of 4 vs the 4 planner kinds + the 9-entry league, props on the line (movable), level replay, MAP, rep floor,
# dial held, hard hits (>= 1.5 m/s) at a fixed 80 each, cap 10 -- the v2PH recipe (ppo_boot_v2rep.sh + v2PH_run.sh + clean80) except:
#   * 16384 learner samples an update (1024 envs, horizon 64) instead of 2048 -- the per-update "faster" signal was noise
#     (SNR 0.15, t 0.96 over 40 updates, rl-audit section 2); possible since the throughput fixes;
#   * actor lr LR -> LR_END (calibrated so an update moves the policy by approx_kl ~0.01; 2e-5 gave 5e-4);
#   * --pg-dead-dims 3,5,6,7 (MAP never executes kappa_4, v2..v4; their PG terms were half the trunk's gradient);
#   * --lap-time-bonus 0 (the sector bonus raced an EMA of every car's times, opponents included: 38 % of the reward, a moving bar);
#   * SPEED_RHO (env) > 0: AR(1) exploration on the plan speeds (only if the SNR measurement supports it).
# usage: GPU=0 LR=.. LR_END=.. [SPEED_RHO=0.95] [TOTAL=..] rl_clean.sh NAME SEED
set -u
NAME=$1; SEED=$2
T=/home/shchon11/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline/tools; C=/home/shchon11/F1tenth_E2E/checkpoints; RR=~/f1sim_runs
INIT=${INIT:-$RR/v2PH_s2421/ppo_u512.pt}
MAPJ=$(cat $T/out/map_k26_round5.json); MAP=real:korea_2026_competition
mkdir -p $RR/${NAME}_seats; cp $INIT $RR/${NAME}_seats/seat0.pt; cp $C/iccas_bootcamp_safest.pt $RR/${NAME}_seats/seat1.pt
LEAGUE="[\"$RR/${NAME}_seats/seat0.pt\",\"$RR/${NAME}_seats/seat1.pt\",\"$C/iccas_constrained_v2R.pt\",\"$C/iccas_constrained_safe_s976.pt\",\"$C/iccas_league_c_s949.pt\",\"$RR/ffR_lin/ppo_final.pt\",\"$RR/spec_korea_c2new_s947/ppo_final.pt\",\"$RR/spec_korea_contact2_s942/ppo_final.pt\",\"$RR/spec_korea_contact_s915/ppo_u768.pt\"]"
SEATS="$RR/${NAME}_seats/seat0.pt,$RR/${NAME}_seats/seat1.pt"
TRACKS="$MAP"; for k in $(seq 0 15); do TRACKS="$TRACKS,$MAP+warp$k"; done
SLOT='{"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"],"speed_scale":[0.7,1.05],"ckpt_mix":'"$LEAGUE"',"ckpt_p":0.5,"present":0.5,"spawn":"random","reactive":{"defend":0.3,"line":0.2,"oblivious":0.1},"events":["brake","stop","shift","weave"],"event_rate":0.5}'
# speed_cap 10: a `self` car is otherwise capped at speed_scale x selfplay_pace_ref (5 m/s, EnvConfig.selfplay_front_cap) --
# an old trick to make the self-play front car slow. Here the self car is a LEARNER row: 35 % of learner steps were clipped
# by a cap neither the actor nor the critic observes (rlc2 first attempt, 10-27 18:28-18:38).
SELF='{"kind":"self","present":'"${SELF_PRESENT:-1.0}"',"spawn":"random","speed_cap":10.0}'
# SELF_PLAY=1: the first opponent seat is the learner's own current weights (a second learner car in every race when present 1)
OPP_SLOTS="[$SLOT,$SLOT,$SLOT]"; [ -n "${SELF_PLAY:-}" ] && OPP_SLOTS="[$SELF,$SLOT,$SLOT]"
OUT=$(cd $(dirname $0)/.. && pwd)/out; mkdir -p $OUT
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
cd /home/shchon11/F1tenth_E2E/f1sim && export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros TORCHINDUCTOR_COMPILE_THREADS=2 CUDA_VISIBLE_DEVICES=${GPU:-0}
echo "[$(date +%H:%M:%S)] $NAME start (code $(git -C /home/shchon11/F1tenth_E2E rev-parse --short HEAD), init $INIT, lr ${LR} -> ${LR_END}, rho ${SPEED_RHO:-0}, gamma ${GAMMA:-0.997}, self-play ${SELF_PLAY:-off} ${SELF_PRESENT:-1.0})" >> $OUT/$NAME.log
python3 -u -m f1sim.learn.ppo --arch v2 --name "$NAME" --init "$INIT" --cond dial \
   --tracks "$TRACKS" --envs ${ENVS:-1024} --epochs 3 --total ${TOTAL:-30000000} \
   --horizon ${HORIZON:-64} --minibatch ${MB:-4096} --grad-accum ${ACCUM:-2} --lr $LR --lr-end $LR_END --critic-lr 3e-4 \
   --gamma ${GAMMA:-0.997} --lam 0.97 --cap0 10 --cap1 10 --cap-steps 5e6 --episode-s 40 \
   --collision-penalty 1 --collision-speed-penalty 0 --steer-penalty 0.02 --proximity-penalty 0 --sideslip-penalty 0 \
   --lap-bonus 0 --lap-time-bonus 0 --grip-budget-penalty 0 --plan-clearance-penalty 0 --wrong-way-penalty 0.4 \
   --race-size 4 --opp-slots "$OPP_SLOTS" --rear-end-penalty 0 --merge-penalty 0 \
   --overtake-bonus 0 --car-proximity-penalty 0 --car-contact-penalty 1 --lagrange wall=1.0,car=1.0 --contact-severity-min 1.5 \
   --lagrange-init 80 --lagrange-lr 0 --lagrange-max 300 \
   --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 18 --procedural-raceline-corridor off --spawn-runway 3.0 \
   --collision-mode soft --movable-obstacles \
   --raceline-objective min_time --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 \
   --level-replay 0.5 --league-slots "$SEATS" --league-every ${LEAGUE_EVERY:-64} \
   --dial-constant 1.0489 --rand-range vehicle.mu=1.0,1.0 --rand-range vehicle.mu_f_scale=0.92,0.92 \
   --plan-follower map --plan-follower-map "$MAPJ" \
   --pg-dead-dims 3,5,6,7 --speed-noise-rho ${SPEED_RHO:-0} \
   --critic-warmup ${CW:-5} --amp --sim-backend graphs --save-every ${SAVE_EVERY:-32} --seed "$SEED" --wandb offline \
   ${EXTRA:-} >> $OUT/$NAME.log 2>&1
echo "[$(date +%H:%M:%S)] $NAME exit $?" | tee -a $OUT/$NAME.log
