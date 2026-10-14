#!/bin/bash
# One arm of the copycat study (docs/research/copycat-2026-10-14.md PREREG), on GPU 0, every step
# sequential: the v2smoke_b DAgger smoke through the harness -> ICCAS solo trials of it0/it1/it2 ->
# held-out BC (buffers saved) -> the copycat probe on both held-out buffers.
# usage: arm.sh NAME [trainer flags of the arm ...]
source /tmp/claude-1000/-home-shchon11-F1tenth-E2E/d98b5699-f98e-4674-9a12-ab5a57bf35f1/scratchpad/v2cc/common.sh
NAME=$1; shift; EXTRA="$*"
R=/home/shchon11/f1sim_runs/$NAME
export CUDA_VISIBLE_DEVICES=0 SMOKE_OUT=$D/$NAME
memok() {
  local a; a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  echo "$(date +%T) $1: MemAvailable $a MB" >> $D/$NAME.steps
  [ "$a" -ge 2500 ] || { echo "$(date +%T) $1: MemAvailable $a MB < 2500, stopping" >> $D/$NAME.steps; exit 3; }
}
memok train
timeout 3500 python3 $D/run_smoke.py --arch v2 --name $NAME --tracks train --envs 512 --iters 3 --steps 240 \
  --keep-iters 3 --epochs 3 --batch 2160 --eval-every 2 --eval-steps 200 --teacher-eval-steps 200 \
  $TEACH $INITSTEM $OPP $WORLD --wandb disabled --seed 991 $EXTRA > $D/$NAME.log 2>&1
rc=$?; echo "exit $rc" >> $D/$NAME.log; echo "$(date +%T) train exit $rc" >> $D/$NAME.steps
[ -f $R/student_it2.pt ] || exit 4
memok iccas
timeout 1800 python3 $D/eval_multi.py $D/iccas_$NAME.json real:korea_2025_iccas \
  $R/student_it0.pt $R/student_it1.pt $R/student_it2.pt > $D/iccas_$NAME.log 2>&1
rc=$?; echo "exit $rc" >> $D/iccas_$NAME.log; echo "$(date +%T) iccas exit $rc" >> $D/$NAME.steps
memok heldout
HELD_CKPTS=$R/student_it0.pt,$R/student_it1.pt,$R/student_it2.pt HELD_OUT=$D/heldout_$NAME.json HELD_STEPS=400 \
  HELD_NSEQ=48 HELD_NBATCH=8 HELD_SAVE=$D/heldbuf_$NAME \
  timeout 2400 python3 $D/heldout_bc.py --arch v2 --name ${NAME}_held --tracks "$HELD" --envs 128 \
  $TEACH $OPP $WORLD --wandb disabled --seed 4242 $EXTRA > $D/heldout_$NAME.log 2>&1
rc=$?; echo "exit $rc" >> $D/heldout_$NAME.log; echo "$(date +%T) heldout exit $rc" >> $D/$NAME.steps
for who in last_student_drives teacher_drives; do
  timeout 1200 python3 $D/copycat_cc.py $D/heldbuf_${NAME}_$who.pt $D/copycat_${NAME}_$who.json \
    $R/student_it0.pt $R/student_it1.pt $R/student_it2.pt > $D/copycat_${NAME}_$who.log 2>&1
  rc=$?; echo "exit $rc" >> $D/copycat_${NAME}_$who.log; echo "$(date +%T) copycat $who exit $rc" >> $D/$NAME.steps
done
echo "$(date +%T) done" >> $D/$NAME.steps
