#!/bin/bash
# the second representation test (net_redesign_v2 §5), ONE job at a time on the shared lab PC. Before each job: no race exam
# (exam_lab) runs, > 6 GB RAM available, > 5 GB GPU memory free. Results -> out/res2, log out/queue2.log
O=$HOME/F1tenth/reprtest/out; T=$HOME/F1tenth/reprtest/tools; R=$O/res2; mkdir -p $R
L=$O/queue2.log
export REPR_GPU_FRAC=0.6
gate() {
  while true; do
    ex=$(pgrep -f "exam_lab/tools_race/exam_record.py" | wc -l)
    av=$(free -m | awk '/^Mem:/{print $7}')
    gf=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$ex" = "0" ] && [ "$av" -gt 6000 ] && [ "$gf" -gt 5000 ] && break
    echo "$(date +%T) $1 waits (exam procs $ex, RAM avail $av MB, GPU free $gf MB)" >> $L; sleep 60
  done
  echo "$(date +%T) start $1" >> $L
}
train() {   # MODE MODEL BASES...
  local MODE=$1 M=$2; shift 2
  gate "$MODE $M"
  cd $HOME/F1tenth/reprtest
  systemd-run --user --scope -q -p MemoryMax=10G /usr/bin/time -v -o $R/${MODE}_$M.rss \
    $HOME/F1tenth/.venv/bin/python3 -u -W ignore $T/lidar_repr_test.py $MODE $M $R "$@" > $R/${MODE}_$M.log 2>&1
  echo "$(date +%T) end $MODE $M rc $? $(grep 'Maximum resident' $R/${MODE}_$M.rss)" >> $L
}
R4="$O/r4a $O/r4b $O/r4c"; L3="$O/l3a $O/l3b"
train opp pts5 $R4
for m in clr_pts5 clr_bev10b clr_bev10 clr_polar_gru; do train clr $m $L3; done
for m in box_ptmem_noise box_ptmemd_noise; do train box $m $L3; done
train opp pts5_noise $R4
train opp pts10 $R4
echo "$(date +%T) all done" >> $L
