#!/bin/bash
# after master.sh: the optional box model (c), same gate (one job at a time, exams first)
O=$HOME/F1tenth/reprtest/out; T=$HOME/F1tenth/reprtest/tools; R=$O/res; L=$O/master.log
until grep -q "all done" $L; do sleep 30; done
while true; do
  ex=$(pgrep -f "exam_lab/tools_race/exam_record.py" | wc -l)
  av=$(free -m | awk '/^Mem:/{print $7}')
  gf=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$ex" = "0" ] && [ "$av" -gt 6000 ] && [ "$gf" -gt 5000 ] && break
  echo "$(date +%T) box box_bevgru waits (exam procs $ex, RAM avail $av MB, GPU free $gf MB)" >> $L; sleep 60
done
if grep -q "^rc 0" $O/l3b.log; then BB="$O/l3a $O/l3b"; else BB="$O/l3a"; fi
echo "$(date +%T) start box box_bevgru" >> $L
cd $HOME/F1tenth/reprtest
systemd-run --user --scope -q -p MemoryMax=10G /usr/bin/time -v -o $R/box_box_bevgru.rss \
  $HOME/F1tenth/.venv/bin/python3 -u -W ignore $T/lidar_repr_test.py box box_bevgru $R $BB > $R/box_box_bevgru.log 2>&1
echo "$(date +%T) end box box_bevgru rc $? $(grep 'Maximum resident' $R/box_box_bevgru.rss)" >> $L
echo "$(date +%T) after done" >> $L
