#!/bin/bash
# 2026-10-28 rl-audit: where does the hit chaos come from? Twin pairs (paired_branch.py, mean action, no shared noise) in the
# fixed training world and in variants that each remove one source: whether a pair's hard-hit count differs ("flip") over
# H steps, per kind (wall / prop / car). Full world: 1.8 % of pairs over 3 s, 58 % of them contact vs no contact at all.
#   full      -- fixed_world.sh as it is
#   noreact   -- opponents drive their own line: no reactive modes, no events
#   static    -- props on the line but immovable (no --movable-obstacles)
#   noprops   -- no props
#   solo      -- no opponents, props kept
#   solo0     -- no opponents, no props
# usage: GPU=0 [ENVS=512] [TRIALS=12] [H=120] [ONLY="noprops solo"] chaos_variants.sh TAG CKPT [extra ppo args]
set -u
TAG=$1; CK=$2; shift 2
D=$(cd $(dirname $0) && pwd); C=/home/shchon11/F1tenth_E2E/checkpoints; RR=~/f1sim_runs; S=$RR/fixedworld_seats
OUT=$(cd $D/.. && pwd)/out
LEAGUE="[\"$S/seat0.pt\",\"$S/seat1.pt\",\"$C/iccas_constrained_v2R.pt\",\"$C/iccas_constrained_safe_s976.pt\",\"$C/iccas_league_c_s949.pt\",\"$RR/ffR_lin/ppo_final.pt\",\"$RR/spec_korea_c2new_s947/ppo_final.pt\",\"$RR/spec_korea_contact2_s942/ppo_final.pt\",\"$RR/spec_korea_contact_s915/ppo_u768.pt\"]"
KINDS='"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"]'
NOREACT='{'"$KINDS"',"speed_scale":[0.7,1.05],"ckpt_mix":'"$LEAGUE"',"ckpt_p":0.5,"present":0.5,"spawn":"random"}'
sed 's/ --movable-obstacles//' $D/fixed_world.sh > $OUT/fixed_world_static.sh
# no props: the procedural flags must go too (ppo refuses density / max-props / corridor without a layout to draw)
sed 's/--procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 18 --procedural-raceline-corridor off //; s/ --movable-obstacles//' \
    $D/fixed_world.sh > $OUT/fixed_world_noprops.sh
# solo: race size 1 and no --opp-slots at all (an empty table is refused: it is not "no slots")
sed 's/--race-size 4 --opp-slots "\[$SLOT,$SLOT,$SLOT\]"/--race-size 1/' $D/fixed_world.sh > $OUT/fixed_world_solo.sh
sed 's/--race-size 4 --opp-slots "\[$SLOT,$SLOT,$SLOT\]"/--race-size 1/' $OUT/fixed_world_noprops.sh > $OUT/fixed_world_solo0.sh
export TRIALS=${TRIALS:-12} W0=${W0:-400} GAP=${GAP:-200} H=${H:-120} VERIFY=${VERIFY:-3} ENVS=${ENVS:-512} \
       SETS=${SETS:-"4:0.1,4:-0.1,0:0.05,0:-0.05"} ENTRY="$D/paired_branch.py --"
run() { local name=$1 script=$2; shift 2
  bash $script chaos_${TAG}_$name $CK 4001 "$@" > /dev/null 2>&1
  echo "== $name"; grep -v wandb $OUT/chaos_${TAG}_$name.log | grep "hit flips\|what the flipped\|Traceback\|Error" | cut -c1-260; }
for v in ${ONLY:-full noreact static noprops solo solo0}; do
  case $v in
    full) run full $D/fixed_world.sh "$@";;
    noreact) run noreact $D/fixed_world.sh --opp-slots "[$NOREACT,$NOREACT,$NOREACT]" "$@";;
    static) run static $OUT/fixed_world_static.sh "$@";;
    noprops) run noprops $OUT/fixed_world_noprops.sh "$@";;
    solo) run solo $OUT/fixed_world_solo.sh "$@";;
    solo0) run solo0 $OUT/fixed_world_solo0.sh "$@";;
  esac
done
