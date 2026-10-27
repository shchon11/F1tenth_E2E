#!/bin/bash
# 2026-10-27 rl-audit: the evaluation races (eval_ckpt.sh) put props OFF the racing line (evaluate.py leaves
# procedural_raceline_corridor at its default "on"), training puts them ON it ("off"). Two checks per checkpoint in the fixed world:
#   evalon -- the evaluation's race (1 planner opponent at 0.7-1.0, props <= 10) with the corridor ON (props off the line), as eval_ckpt.sh
#   solobox -- no opponent (the seat present 0), props on the line as in training: how a car does with nobody to follow
# usage: GPU=0 fw_boxes.sh TAG CKPT [extra ppo args]
set -u
TAG=$1; CK=$2; shift 2
D=$(cd $(dirname $0) && pwd)
EVAL='{"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"],"speed_scale":[0.7,1.0]}'
NONE='{"kind":"forzaeth","present":0.0}'
bash $D/fixed_world.sh fw_${TAG}_evalon $CK 4001 --race-size 2 --opp-slots "[$EVAL]" --procedural-max-props 10 --procedural-raceline-corridor on "$@"
bash $D/fixed_world.sh fw_${TAG}_solobox $CK 4001 --race-size 2 --opp-slots "[$NONE]" "$@"
