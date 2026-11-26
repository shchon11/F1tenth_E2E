#!/bin/bash
# (11-26, the user: "그 정보들이 네트워크에 있는지 디버깅용 디코더 ... 사이트나 콘솔에서도") every exam record on the lab PC that kept the
# policy's state (F1SIM_EXAM_SAVE_MEM=1: <record>_mem.npz -- lab_exam2m_loop.sh, lab_race_exam_loop.sh's exam3) gets its state
# decoders trained there (state_decoder_train.py: feat / mem / feat+mem -> pose, ego state, walls, boxes, the other car; held-out
# box layouts), one record at a time, then the decoder, the test sessions' inputs and the metrics come here:
#   out/decoder/<record>_dec.npz, <record>_decin.npz, <record>_decode.txt (+ <record>.npz when this PC has no copy of the record)
#   out/analysis/<run>_u<N>_decode.txt (exam3) / <run>_u<N>_decode_<exam>.txt (others)    -- the site's /dec page reads out/decoder
#   ~/f1sim_runs/<run>/plan_u<N>_dec.npz here and on the lab, beside the checkpoint (the console's decoder overlay): the exam4
#   record's decoder when there is one (its other-car head saw a car), else the exam3 one
# Exams first: a decoder starts only while no exam_record.py runs on the lab and it has > 6 GB RAM free; each runs niced under
# systemd-run --user --scope -p MemoryMax=10G (another agent shares the lab). ~1 min a record on the lab's GPU.
# Which records: those whose _mem.npz is newer than the watcher's start (or than $1, any `date -d` time, e.g. "2026-11-26 00:00"),
# plus every one whose name matches DEC_BACKFILL (a regex, e.g. 'cp36c_u9[0-9]{2}'), oldest first; a record is done when the lab
# has dec/<record>.dec.done (failed twice: .dec.fail). Runs until killed (by PID).
# The code it sends to the lab (state_decoder_train.py, f1sim/learn/state_decoder.py) is this script's own checkout's; the outputs
# always go to the main checkout's out/ (the site's). Start (from the checkout to use):
#   cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01 && mkdir -p out/decoder && \
#   setsid nohup bash tools/lab_decoder_watch.sh > out/decoder/watch.log 2>&1 < /dev/null &
H=${LAB_H:-shchon11@10.100.0.2}
 SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
SELF=$(cd "$(dirname "$0")" && pwd)
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out; RUNS=/home/shchon11/f1sim_runs   # the site reads this out/
MOD=$(cd "$SELF/../../../../f1sim/f1sim/learn" && pwd)/state_decoder.py
LX=/home/shchon11/F1tenth/exam_lab/out; LT=/home/shchon11/F1tenth/exam_lab/tools_dec
PYL=/home/shchon11/F1tenth/.venv/bin/python3
SINCE=$(date -d "${1:-now}" +%s) || exit 1
BACK=${DEC_BACKFILL:-}
mkdir -p $D/decoder $D/analysis
for try in 1 2 3; do
  tar -cf - -C $SELF state_decoder_train.py -C $(dirname $MOD) state_decoder.py | $SSH $H "mkdir -p $LT $LX/dec && tar -x -C $LT" && break
  sleep 30
done
echo "$(date +%T) 시작: $(date -d @$SINCE +'%m-%d %H:%M') 뒤의 기억 기록${BACK:+ + '$BACK'}"
lab_free() {          # 0 when no exam runs on the lab and it has room; prints why not
  $SSH $H 'n=$(ps -eo comm=,args= | awk "\$1 ~ /^python/ && /exam_record[.]py/" | wc -l)
    m=$(awk "/MemAvailable/ {print int(\$2/1048576)}" /proc/meminfo)
    [ "$n" -eq 0 ] && [ "$m" -gt 6 ] || { echo "시험 ${n}개 · RAM 여유 ${m} GB"; exit 1; }'
}
while true; do
  # every finished record with a memory file and no decoder yet: "mtime path" (exam/ = the exam loops', dec/ = made for this)
  list=$($SSH $H "cd $LX && for f in exam/*_mem.npz dec/*_mem.npz; do [ -e \"\$f\" ] || continue; b=\${f%_mem.npz}; n=\${b##*/}
      [ -e \$b.npz ] && grep -q '^rc 0' \$b.log 2>/dev/null || continue
      [ -e dec/\$n.dec.done ] || [ -e dec/\$n.dec.fail ] && continue
      echo \$(stat -c %Y \$f) \$b; done" 2>/dev/null | sort -n)
  todo=()
  while read -r mt b; do
    [ -n "$b" ] || continue
    n=${b##*/}
    if [ "$mt" -ge "$SINCE" ] || { [ -n "$BACK" ] && [[ $n =~ $BACK ]]; }; then todo+=("$b"); fi
  done <<< "$list"
  for b in "${todo[@]}"; do
    n=${b##*/}
    [[ $n =~ ^(exam[0-9]+)_([A-Za-z0-9]+)_u([0-9]+)_ ]] || continue
    EX=${BASH_REMATCH[1]}; RUN=${BASH_REMATCH[2]}; U=${BASH_REMATCH[3]}
    until why=$(lab_free); do echo "$(date +%T) $n: 기다림 ($why)"; sleep 60; done
    # beside the checkpoint (the console): the exam4 record's decoder wins, an exam3 one only where there is none yet
    CK=/home/shchon11/f1sim_runs/$RUN/plan_u$U.pt; CC=/home/shchon11/f1sim_runs/$RUN/plan_u${U}_dec.npz
    COPY=$($SSH $H "[ -e $CK ] && { [ $EX = exam4 ] || [ ! -e $CC ]; } && echo --ckpt-copy=$CC")
    $SSH $H "cd $LX && rm -f dec/$n.declog && setsid nohup bash -c 'systemd-run --user --scope -q -p MemoryMax=10G nice -n 10 $PYL -u -W ignore $LT/state_decoder_train.py $b.npz --out dec/$n $COPY > dec/$n.declog 2>&1; echo rc \$? >> dec/$n.declog' > /dev/null 2>&1 < /dev/null &"
    for k in $(seq 1 120); do sleep 15; $SSH $H "grep -q '^rc ' $LX/dec/$n.declog" 2>/dev/null && break; done
    if $SSH $H "grep -q '^rc 0' $LX/dec/$n.declog"; then
      extra=""; [ -e $D/exam/$n.npz ] || extra="$b.npz"
      for t in 1 2 3; do $SSH $H "cd $LX && tar -cf - dec/${n}_dec.npz dec/${n}_decin.npz dec/${n}_decode.txt $extra" \
        | tar -x -C $D/decoder --strip-components=1 && break; sleep 20; done
      if [ -e $D/decoder/${n}_decode.txt ]; then
        A=$D/analysis/${RUN}_u${U}_decode$([ $EX = exam3 ] || echo _$EX).txt
        cp $D/decoder/${n}_decode.txt $A
        [ -n "$COPY" ] && [ -e $RUNS/$RUN/plan_u$U.pt ] && cp $D/decoder/${n}_dec.npz $RUNS/$RUN/plan_u${U}_dec.npz
        $SSH $H "touch $LX/dec/$n.dec.done"
        echo "$(date +%T) $n 끝 -> $A"
      fi
    else
      if $SSH $H "[ -e $LX/dec/$n.dec.try ]"; then $SSH $H "touch $LX/dec/$n.dec.fail"; else $SSH $H "touch $LX/dec/$n.dec.try"; fi
      echo "$(date +%T) $n 실패: $($SSH $H "tail -3 $LX/dec/$n.declog" | tr '\n' ' ')"
    fi
  done
  sleep 120
done
