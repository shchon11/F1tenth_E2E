#!/bin/bash
# 2026-10-27 rl-audit: in the fixed training world the follower (rlc2 u256) out-scores the passer (v2PH u512) on PPO's own objective
# (J/step 0.0907 vs 0.0805), while in the evaluation races the passer wins (~4.6 vs ~3.8 per s). Which part of the training world makes
# passing a loss? The same J for one checkpoint in variants that each remove one part (fixed_world.sh, extra args override):
#   noreact  -- opponents drive their own line: no reactive modes (defend/line/oblivious), no events (brake/stop/shift/weave)
#   noleague -- planner opponents only (no league checkpoints), reactive + events kept
#   evallike -- the evaluation's race: 1 planner opponent at 0.7-1.0, nothing reactive, props <= 10
# usage: GPU=0 fw_variants.sh TAG CKPT [extra ppo args, e.g. --grad-accum 4]
set -u
TAG=$1; CK=$2; shift 2
D=$(cd $(dirname $0) && pwd); C=/home/shchon11/F1tenth_E2E/checkpoints; RR=~/f1sim_runs; S=$RR/fixedworld_seats
LEAGUE="[\"$S/seat0.pt\",\"$S/seat1.pt\",\"$C/iccas_constrained_v2R.pt\",\"$C/iccas_constrained_safe_s976.pt\",\"$C/iccas_league_c_s949.pt\",\"$RR/ffR_lin/ppo_final.pt\",\"$RR/spec_korea_c2new_s947/ppo_final.pt\",\"$RR/spec_korea_contact2_s942/ppo_final.pt\",\"$RR/spec_korea_contact_s915/ppo_u768.pt\"]"
KINDS='"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"]'
NOREACT='{'"$KINDS"',"speed_scale":[0.7,1.05],"ckpt_mix":'"$LEAGUE"',"ckpt_p":0.5,"present":0.5,"spawn":"random"}'
NOLEAGUE='{'"$KINDS"',"speed_scale":[0.7,1.05],"present":0.5,"spawn":"random","reactive":{"defend":0.3,"line":0.2,"oblivious":0.1},"events":["brake","stop","shift","weave"],"event_rate":0.5}'
EVAL='{'"$KINDS"',"speed_scale":[0.7,1.0]}'
bash $D/fixed_world.sh fw_${TAG}_noreact $CK 4001 --opp-slots "[$NOREACT,$NOREACT,$NOREACT]" "$@"
bash $D/fixed_world.sh fw_${TAG}_noleague $CK 4001 --opp-slots "[$NOLEAGUE,$NOLEAGUE,$NOLEAGUE]" "$@"
bash $D/fixed_world.sh fw_${TAG}_evallike $CK 4001 --race-size 2 --opp-slots "[$EVAL]" --procedural-max-props 10 "$@"
