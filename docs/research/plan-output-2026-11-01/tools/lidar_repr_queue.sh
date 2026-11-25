#!/bin/bash
# ONE job of the representation test at a time on the shared lab PC. Before each job: the recording chain is done, no race
# exam (exam_lab) runs, > 6 GB RAM available, > 5 GB GPU memory free (room for an exam next to it). Order: the box smoke on
# l3a (bugs before training), the opponent models, the l3b recording again (it died of a CUDA OOM), the box models.
O=$HOME/F1tenth/reprtest/out; T=$HOME/F1tenth/reprtest/tools; R=$O/res; mkdir -p $R $O/smoke_box
L=$O/master.log
until grep -q "chain done" $O/chain1.log; do sleep 30; done
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
train() {   # MODE MODEL OUTDIR BASES... (extra env from the caller)
  local MODE=$1 M=$2 OD=$3; shift 3
  gate "$MODE $M"
  cd $HOME/F1tenth/reprtest
  systemd-run --user --scope -q -p MemoryMax=10G /usr/bin/time -v -o $OD/${MODE}_$M.rss \
    $HOME/F1tenth/.venv/bin/python3 -u -W ignore $T/lidar_repr_test.py $MODE $M $OD "$@" > $OD/${MODE}_$M.log 2>&1
  echo "$(date +%T) end $MODE $M rc $? $(grep 'Maximum resident' $OD/${MODE}_$M.rss)" >> $L
}
REPR_BOX_EPOCHS=1 train box box_mem $O/smoke_box $O/l3a
[ -e $O/hold ] && { echo "$(date +%T) held after the box smoke" >> $L; until [ ! -e $O/hold ]; do sleep 30; done; }
for m in bev10 polar_gru polar_stack10 bev5 polar_stack5 polar_wide10 bev10_noise; do
  train opp $m $R $O/r4a $O/r4b $O/r4c
done
gate "record l3b"
bash $T/rec.sh l3b line3 256 60 999 20261126 "F1SIM_EXAM_SAVE_SCAN=1 F1SIM_EXAM_SAVE_BOXVIS=1 F1SIM_EXAM_ALL_BOXES=1"
echo "$(date +%T) end record l3b $(tail -1 $O/l3b.log)" >> $L
if grep -q "^rc 0" $O/l3b.log; then BB="$O/l3a $O/l3b"; else BB="$O/l3a"; fi
for m in box_mem box_gru box_now box_mem_noise; do
  train box $m $R $BB
done
echo "$(date +%T) all done" >> $L
