#!/bin/bash
# Evaluate a run's checkpoints as they appear (every EVERY updates, plus the init once) with eval_ckpt.sh on GPU G, then print the
# summary table. Stops when the run's log says "exit". usage: G=1 EVERY=128 watch_eval.sh NAME [INIT]
D=$(cd $(dirname $0) && pwd); NAME=$1; INIT=${2:-}; G=${G:-1}; EVERY=${EVERY:-128}
RUN=~/f1sim_runs/$NAME; LOG=$D/../out/$NAME.log
if [ -n "$INIT" ]; then bash $D/eval_ckpt.sh $G ${NAME}_init $INIT both; fi
done_u=""
while true; do
  for f in $(ls $RUN/ppo_u*.pt 2>/dev/null | sort -V); do
    u=$(basename $f .pt | sed 's/ppo_u//')
    [ $((u % EVERY)) -eq 0 ] || continue
    case " $done_u " in *" $u "*) continue;; esac
    bash $D/eval_ckpt.sh $G ${NAME}_u$u $f both && done_u="$done_u $u"
    args=""; [ -n "$INIT" ] && args="init=$D/../out/${NAME}_init"; [ -n "${BASE:-}" ] && args="$args base=$BASE"
    for v in $done_u; do args="$args u$v=$D/../out/${NAME}_u$v"; done
    python3 $D/summarize.py $args
  done
  if grep -q "$NAME exit" $LOG 2>/dev/null; then
    f=$RUN/ppo_final.pt; [ -f $f ] && bash $D/eval_ckpt.sh $G ${NAME}_final $f both
    echo "[$(date +%H:%M)] watch $NAME finished"; break
  fi
  sleep 120
done
