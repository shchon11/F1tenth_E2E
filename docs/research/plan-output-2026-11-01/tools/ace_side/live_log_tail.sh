#!/bin/bash
# (11-27) ON ACE: gzip of "<size> <name> <seconds since the last tap>\n" + the bytes of the current log (live_log.cur names it) from offset $2 to that size
# (ace_live_play.py's pull)
DIR=${3:-/nas/workspace/shchon11_f1sim/runs}/$1; OFF=${2:-0}
N=$(cat $DIR/live_log.cur 2>/dev/null); L=$DIR/$N
S=$( [ -n "$N" ] && stat -c %s $L 2>/dev/null || echo 0)
A=$(( $(date +%s) - $(stat -c %Y $DIR/live_tap.json 2>/dev/null || echo 0) ))      # seconds since the run last wrote a tap
{ echo "$S ${N:--} $A"; [ "$S" -gt "$OFF" ] && tail -c +$((OFF + 1)) $L | head -c $((S - OFF)); } | gzip -1
