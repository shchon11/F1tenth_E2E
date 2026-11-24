#!/bin/bash
# the representation test's recordings, one at a time; each waits until no exam of the race loop runs and > 6 GB RAM is free
T=$HOME/F1tenth/reprtest/tools; O=$HOME/F1tenth/reprtest/out
waitfree() {
  while true; do
    ex=$(pgrep -f "exam_lab/tools_race/exam_record.py" | wc -l)
    av=$(free -m | awk '/^Mem:/{print $7}')
    [ "$ex" = "0" ] && [ "$av" -gt 6000 ] && break
    echo "$(date +%T) waiting (exam procs $ex, avail $av MB)" >> $O/chain1.log; sleep 30
  done
}
one() { waitfree; echo "$(date +%T) start $1" >> $O/chain1.log; bash $T/rec.sh "$@"; echo "$(date +%T) end $1 $(tail -1 $O/$1.log)" >> $O/chain1.log; }
one r4a race4 256 90 50 20261125 "F1SIM_EXAM4_PACES=0.5,0.6,0.8,0.95 F1SIM_EXAM_SAVE_SCAN=1 F1SIM_EXAM_PRED_SAVE=1"
one r4b race4 256 90 50 20261126 "F1SIM_EXAM4_PACES=0.55,0.7,0.85,1.0 F1SIM_EXAM_SAVE_SCAN=1 F1SIM_EXAM_PRED_SAVE=1"
one l3a line3 256 60 999 20261125 "F1SIM_EXAM_SAVE_SCAN=1 F1SIM_EXAM_SAVE_BOXVIS=1 F1SIM_EXAM_ALL_BOXES=1"
one l3b line3 256 60 999 20261126 "F1SIM_EXAM_SAVE_SCAN=1 F1SIM_EXAM_SAVE_BOXVIS=1 F1SIM_EXAM_ALL_BOXES=1"
one r4c race4 256 90 50 20261127 "F1SIM_EXAM4_PACES=0.5,0.65,0.75,0.9 F1SIM_EXAM_SAVE_SCAN=1 F1SIM_EXAM_PRED_SAVE=1"
echo "$(date +%T) chain done" >> $O/chain1.log
