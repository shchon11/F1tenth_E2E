#!/bin/bash
# (11-19) lab_exam2_loop.sh + F1SIM_EXAM_SAVE_MEM=1: every exam also keeps the policy's features and memory state per decision
# (<record>_mem.npz) -> whether --mem-aux-seen put 'a passed box is ahead' into the memory the policy drives with
# (box_memory_decode.py), measured on the same exam as the driving; the lab copy is removed after fetching (~190 MB each)
# (11-18) exam2 = 3 boxes a race on the empty-track line (exam_record.py count "line3", user OK 11-18 "ㄱㅊ네"); copied from
# lab_exam_loop.sh (the running loops keep that file) -- records exam2_..._line3_n256, analysis <run>_u<N>_exam2.txt/.done.
# (11-17) frozen exam on the lab PC (RTX 4060 Ti) every STEP updates, so both GPUs here train (user: "5060 4070 다 써") and the
# collision maps come every 25 (user: "100부터 25단위로 쭈르륵"). Per checkpoint: copy it (+ args.json) to the lab, run exam_record.py
# there (exam_judge.sh's command; the world json is found by name in the lab's code, tools/world_of.py), fetch the record,
# write the analysis (.txt + .done) here. The remote exam runs under setsid and is polled, so a dropped tunnel does not kill it.
# Args: RUN U0 STEP U1
# (11-23) EXAM=exam3: a run trained in the bag-scan world (its world json names the exam: exam_name "exam3", the exam3 bank) --
# records exam3_..._line3_n256, analysis <run>_u<N>_exam3.txt/.done; default exam2 as before
RUN=$1; U0=$2; STEP=${3:-25}; U1=$4; EX=${EXAM:-exam2}
H=${LAB_H:-shchon11@10.100.0.2};   # (11-20) WireGuard (lab side: interface 'home'); LAB_H=lab = Tailscale. When both were up
# Tailscale rode over the WireGuard path and crawled at ~3 KB/s; the WireGuard address itself did 6 MB/s
 SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; D=$R/out; RUNS=/home/shchon11/f1sim_runs
LX=/home/shchon11/F1tenth/exam_lab/out/exam
for UU in $(seq $U0 $STEP $U1); do
  C=$RUNS/$RUN/plan_u$UU.pt
  until [ -e $C ] || [ -e $D/one_$RUN.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 20
  B=${EX}_${RUN}_u${UU}_ideal_line3_n256
  [ -e $D/exam/$B.npz ] && continue
  for try in 1 2 3; do
    tar -cf - -C $RUNS $RUN/args.json $RUN/plan_u$UU.pt | $SSH $H 'mkdir -p ~/f1sim_runs && tar -x -C ~/f1sim_runs' || { sleep 30; continue; }
    # (the lab's login shell is zsh: a glob with no match aborted this whole line -- the exam never started; plain names)
    $SSH $H "mkdir -p $LX && rm -f $LX/$B.npz $LX/${B}_mem.npz $LX/$B.log && cd ~/F1tenth/console_main/f1sim && setsid nohup bash -c 'PYTHONPATH=\$PWD F1SIM_EXAM_SAVE_MEM=1 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555 CUDA_VISIBLE_DEVICES=0 timeout 5400 ~/F1tenth/.venv/bin/python3 -u -W ignore ~/F1tenth/exam_lab/tools/exam_record.py ~/f1sim_runs/$RUN/plan_u$UU.pt $LX/$B.npz ideal 256 line3 > $LX/$B.log 2>&1; echo rc \$? >> $LX/$B.log' > /dev/null 2>&1 < /dev/null &"
    for k in $(seq 1 200); do sleep 30; $SSH $H "grep -q '^rc ' $LX/$B.log" 2>/dev/null && break; done
    # (11-20) the memory file (126 MB) stays on the lab (fetch it by hand when needed): over Tailscale it crawled at ~1 KB/s
    # and held the loop 20 min after the exam was done
    $SSH $H "cd $LX && tar -cf - $B.npz $B.log" | tar -x -C $D/exam 2>/dev/null
    [ -e $D/exam/$B.npz ] && break
    sleep 60
  done
  if [ -e $D/exam/$B.npz ]; then
    (source /home/shchon11/F1tenth_E2E/.venv/bin/activate; cd /home/shchon11/F1tenth_E2E/f1sim
     # (11-23, the user: "평가도 장애물이 몇초전부터 보이는지에 따라서 충돌률 보이도록해") every exam also gets the box table
     # (exam_boxes.py: hard hits by how long before passing the box came into view, ...) -> analysis/RUN_uU_boxes.txt, which the site
     # shows per checkpoint; its by-visibility rows also go into the exam's own text
     python3 -W ignore $T/exam_boxes.py $D/exam/$B.npz > $D/analysis/${RUN}_u${UU}_boxes.txt 2>&1
     { python3 -W ignore $T/exam_table.py $D/exam/$B.npz; python3 -W ignore $T/side_flip.py $D/exam/$B.npz;
       sed -n '/(나)/,/(다)/p' $D/analysis/${RUN}_u${UU}_boxes.txt | grep -v '(다)'; } > $D/analysis/${RUN}_u${UU}_${EX}.txt 2>&1)
    echo done > $D/analysis/${RUN}_u${UU}_${EX}.done
  fi
  echo "$(date +%T) $RUN u$UU $EX $([ -e $D/exam/$B.npz ] && echo 끝 || echo 실패)"
done
