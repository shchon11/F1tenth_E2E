#!/bin/bash
# after after.sh: the two GRU box models again with 36 epochs (convergence check; box_gru was still improving at 12)
O=$HOME/F1tenth/reprtest/out; T=$HOME/F1tenth/reprtest/tools; R=$O/res_long; L=$O/master.log; mkdir -p $R
until grep -q "after done" $L; do sleep 30; done
if grep -q "^rc 0" $O/l3b.log; then BB="$O/l3a $O/l3b"; else BB="$O/l3a"; fi
for m in box_gru box_bevgru; do
  while true; do
    ex=$(pgrep -f "exam_lab/tools_race/exam_record.py" | wc -l)
    av=$(free -m | awk '/^Mem:/{print $7}')
    gf=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$ex" = "0" ] && [ "$av" -gt 6000 ] && [ "$gf" -gt 5000 ] && break
    echo "$(date +%T) long $m waits (exam procs $ex, RAM avail $av MB, GPU free $gf MB)" >> $L; sleep 60
  done
  echo "$(date +%T) start long box $m" >> $L
  cd $HOME/F1tenth/reprtest
  REPR_BOX_EPOCHS=36 systemd-run --user --scope -q -p MemoryMax=10G /usr/bin/time -v -o $R/box_$m.rss \
    $HOME/F1tenth/.venv/bin/python3 -u -W ignore $T/lidar_repr_test.py box $m $R $BB > $R/box_$m.log 2>&1
  echo "$(date +%T) end long box $m rc $? $(grep 'Maximum resident' $R/box_$m.rss)" >> $L
done
echo "$(date +%T) after2 done" >> $L
