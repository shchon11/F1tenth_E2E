#!/bin/bash
# (11-27) ON ACE: gzip of "<size of live_log.tsv>\n" + its bytes from offset $2 to that size (ace_live_play.py's pull)
L=/nas/workspace/shchon11_f1sim/runs/$1/live_log.tsv; OFF=${2:-0}
S=$(stat -c %s $L 2>/dev/null || echo 0)
{ echo $S; [ "$S" -gt "$OFF" ] && tail -c +$((OFF + 1)) $L | head -c $((S - OFF)); } | gzip -1
