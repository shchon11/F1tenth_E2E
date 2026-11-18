#!/bin/bash
# (11-18) exam2 = 3 boxes a race on the empty-track line (exam_record.py count "line3", user OK 11-18 "ㄱㅊ네"); copied from
# lab_exam_loop.sh (the running loops keep that file) -- records exam2_..._line3_n256, analysis <run>_u<N>_exam2.txt/.done.
# (11-17) frozen exam on the lab PC (RTX 4060 Ti) every STEP updates, so both GPUs here train (user: "5060 4070 다 써") and the
# collision maps come every 25 (user: "100부터 25단위로 쭈르륵"). Per checkpoint: copy it (+ args.json) to the lab, run exam_record.py
# there (exam_judge.sh's command; the world json is found by name in the lab's code, tools/world_of.py), fetch the record,
# write the analysis (.txt + .done) here. The remote exam runs under setsid and is polled, so a dropped tunnel does not kill it.
# Args: RUN U0 STEP U1
RUN=$1; U0=$2; STEP=${3:-25}; U1=$4
H=shchon11@10.100.0.2; SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; D=$R/out; RUNS=/home/shchon11/f1sim_runs
LX=/home/shchon11/F1tenth/exam_lab/out/exam
for UU in $(seq $U0 $STEP $U1); do
  C=$RUNS/$RUN/plan_u$UU.pt
  until [ -e $C ] || [ -e $D/one_$RUN.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 20
  B=exam2_${RUN}_u${UU}_ideal_line3_n256
  [ -e $D/exam/$B.npz ] && continue
  for try in 1 2 3; do
    tar -cf - -C $RUNS $RUN/args.json $RUN/plan_u$UU.pt | $SSH $H 'mkdir -p ~/f1sim_runs && tar -x -C ~/f1sim_runs' || { sleep 30; continue; }
    # (the lab's login shell is zsh: a glob with no match aborted this whole line -- the exam never started; plain names)
    $SSH $H "mkdir -p $LX && rm -f $LX/$B.npz $LX/$B.log && cd ~/F1tenth/console_main/f1sim && setsid nohup bash -c 'PYTHONPATH=\$PWD WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555 CUDA_VISIBLE_DEVICES=0 timeout 5400 ~/F1tenth/.venv/bin/python3 -u -W ignore ~/F1tenth/exam_lab/tools/exam_record.py ~/f1sim_runs/$RUN/plan_u$UU.pt $LX/$B.npz ideal 256 line3 > $LX/$B.log 2>&1; echo rc \$? >> $LX/$B.log' > /dev/null 2>&1 < /dev/null &"
    for k in $(seq 1 200); do sleep 30; $SSH $H "grep -q '^rc ' $LX/$B.log" 2>/dev/null && break; done
    $SSH $H "cd $LX && tar -cf - $B.npz $B.log" | tar -x -C $D/exam 2>/dev/null
    [ -e $D/exam/$B.npz ] && break
    sleep 60
  done
  if [ -e $D/exam/$B.npz ]; then
    (source /home/shchon11/F1tenth_E2E/.venv/bin/activate; cd /home/shchon11/F1tenth_E2E/f1sim
     { python3 -W ignore $T/exam_table.py $D/exam/$B.npz; python3 -W ignore $T/side_flip.py $D/exam/$B.npz; } > $D/analysis/${RUN}_u${UU}_exam2.txt 2>&1)
    echo done > $D/analysis/${RUN}_u${UU}_exam2.done
  fi
  echo "$(date +%T) $RUN u$UU 시험2 $([ -e $D/exam/$B.npz ] && echo 끝 || echo 실패)"
done
