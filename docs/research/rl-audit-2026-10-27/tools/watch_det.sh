#!/bin/bash
# 2026-10-27 rl-audit: watch a run and, for every EVERY-th checkpoint, (1) eval_ckpt.sh with props ON the racing line (CORRIDOR=off,
# as training and the venue), (2) the deterministic objective J in the fixed training world (fixed_world.sh with
# F1SIM_ROLLOUT_MEAN=1, UPD updates) -- the deployed policy's own score, which the training log cannot show (section 9). Then the
# tables. Stops when the run's log says "exit". usage: G=1 EVERY=128 [UPD=20] [BASE=out/line_base] watch_det.sh NAME
D=/home/shchon11/F1tenth_E2E/docs/research/rl-audit-2026-10-27/tools; NAME=$1; G=${G:-1}; EVERY=${EVERY:-128}
RUN=~/f1sim_runs/$NAME; LOG=$D/../out/$NAME.log; EX=""; [ "$G" = 1 ] && EX="--grad-accum 4"
# the fixed-world J loads the checkpoint's critic too: give the env the same privileged columns the run trained with
FWX=$(python3 -c "import json,os; a=json.load(open(os.path.expanduser('~/f1sim_runs/$NAME/args.json'))); "\
"print(' '.join(f for k, f in (('race_end_terminal', '--race-end-terminal'), ('priv_props', '--priv-props')) if a.get(k)))" 2>/dev/null)
done_u=""
while true; do
  for f in $(ls $RUN/ppo_u*.pt 2>/dev/null | sort -V); do
    u=$(basename $f .pt | sed 's/ppo_u//')
    [ $((u % EVERY)) -eq 0 ] || continue
    [ $u -ge ${MIN_U:-0} ] || continue                 # MIN_U: skip checkpoints before it (e.g. a critic-only warm-up)
    case " $done_u " in *" $u "*) continue;; esac
    CORRIDOR=off bash $D/eval_ckpt.sh $G ${NAME}_u$u $f both || continue
    F1SIM_ROLLOUT_MEAN=1 GPU=$G UPD=${UPD:-20} bash $D/fixed_world.sh fwd_${NAME}_u$u $f 4001 $EX $FWX > /dev/null
    done_u="$done_u $u"
    args=""; [ -n "${BASE:-}" ] && args="base=$BASE"; names="fwd_base"
    for v in $done_u; do args="$args u$v=$D/../out/${NAME}_u$v"; names="$names fwd_${NAME}_u$v"; done
    python3 $D/summarize.py $args; python3 $D/fixed_world_j.py $names
  done
  if grep -q "$NAME exit" $LOG 2>/dev/null; then echo "[$(date +%H:%M)] watch $NAME finished"; break; fi
  sleep 120
done
