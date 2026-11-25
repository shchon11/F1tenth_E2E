#!/bin/bash
# (11-24, the race phase) the lab PC's exams for a race run, one checkpoint after the other: exam3 (the line exam -- a race world
# names its solo world, so this is the qualifier's exam3 exactly) and then exam4 (the race exam: exam_record.py count "race4",
# 128 sessions = 16 conditions x 8, the other car a frozen copy of the fixed reference policy). Copied from lab_exam2m_loop.sh
# (a running loop keeps that file); one exam at a time on the lab, the two in a row per checkpoint.
# Records exam3_<run>_u<N>_ideal_line3_n256 / exam4_<run>_u<N>_ideal_race4_n256, analysis <run>_u<N>_exam3.txt/.done and
# <run>_u<N>_exam4.txt/.done (exam_race_table.py).
# Args: RUN U0 STEP U1. The lab's code: LAB_CODE (its f1sim, default the console's copy that lab_sync.sh keeps at this repo's
# HEAD); its exam tools: LAB_TOOLS (default ~/F1tenth/exam_lab/tools_race, written by this loop from this repo at the start);
# the reference policy: ~/f1sim_runs/exam4_ref/plan_u500.pt (scan31b u500), sent if missing.
# (11-25, the ORACLE gate) EXAM4_ENV: extra environment for the exam4 record (e.g. F1SIM_EXAM_PRED_CLEAR_ORACLE=1 -- a
# --pred-clear-oracle run is examined with the inputs it trained on); EXAM4_HEAD=1: also a second exam4 with no EXAM4_ENV (the
# policy's own learned inputs), record exam4_<run>_u<N>_ideal_race4_n256_head, analysis <run>_u<N>_exam4head.txt/.done.
# Every exam runs under systemd-run --user --scope -p MemoryMax=10G (another agent shares the lab) and starts only while the lab
# has fewer than 2 heavy jobs (exam_record / lidar_repr_test / plan_rl / state_student), > 6 GB RAM and >= 3 GB GPU memory free.
RUN=$1; U0=$2; STEP=${3:-25}; U1=$4
H=${LAB_H:-shchon11@10.100.0.2}
 SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; D=$R/out; RUNS=/home/shchon11/f1sim_runs
LX=/home/shchon11/F1tenth/exam_lab/out/exam
LC=${LAB_CODE:-/home/shchon11/F1tenth/console_main/f1sim}
LT=${LAB_TOOLS:-/home/shchon11/F1tenth/exam_lab/tools_race}
for try in 1 2 3; do
  tar -cf - -C $T exam_record.py world_of.py | $SSH $H "mkdir -p $LT && tar -x -C $LT" && \
  tar -chf - -C $RUNS exam4_ref | $SSH $H 'mkdir -p ~/f1sim_runs && tar -x -C ~/f1sim_runs' && break
  sleep 30
done
ENVS="PYTHONPATH=$LC WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555 CUDA_VISIBLE_DEVICES=0"
lab_room() {          # 0 when the lab has room for one more exam (see the note above); prints why not
  $SSH $H 'n=$(ps -eo comm=,args= | awk "\$1 ~ /^python/ && /exam_record[.]py|lidar_repr_test[.]py|plan_rl|state_student/" | wc -l)
    m=$(awk "/MemAvailable/ {print int(\$2/1048576)}" /proc/meminfo)
    g=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, "{print int((\$1-\$2)/1024)}" | head -1)
    [ "$n" -lt 2 ] && [ "$m" -gt 6 ] && [ "$g" -ge 3 ] || { echo "작업 ${n}개 · RAM 여유 ${m} GB · GPU 여유 ${g} GB"; exit 1; }'
}
exam_one() {          # $1 record name, $2 count, $3 extra env -> 0 when the record is here
  local B=$1 CNT=$2 XE=$3
  [ -e $D/exam/$B.npz ] && return 0
  for try in 1 2 3; do
    tar -cf - -C $RUNS $RUN/args.json $RUN/plan_u$UU.pt | $SSH $H 'mkdir -p ~/f1sim_runs && tar -x -C ~/f1sim_runs' || { sleep 30; continue; }
    until why=$(lab_room); do echo "$(date +%T) $B: 랩 자리 기다림 ($why)"; sleep 60; done
    $SSH $H "mkdir -p $LX && rm -f $LX/$B.npz $LX/$B.log && cd $LC && setsid nohup bash -c 'systemd-run --user --scope -q -p MemoryMax=10G env $ENVS $XE timeout 5400 ~/F1tenth/.venv/bin/python3 -u -W ignore $LT/exam_record.py ~/f1sim_runs/$RUN/plan_u$UU.pt $LX/$B.npz ideal 256 $CNT > $LX/$B.log 2>&1; echo rc \$? >> $LX/$B.log' > /dev/null 2>&1 < /dev/null &"
    for k in $(seq 1 200); do sleep 30; $SSH $H "grep -q '^rc ' $LX/$B.log" 2>/dev/null && break; done
    $SSH $H "cd $LX && tar -cf - $B.npz $B.log" | tar -x -C $D/exam 2>/dev/null
    [ -e $D/exam/$B.npz ] && return 0
    sleep 60
  done
  return 1
}
for UU in $(seq $U0 $STEP $U1); do
  C=$RUNS/$RUN/plan_u$UU.pt
  until [ -e $C ] || [ -e $D/one_$RUN.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 20
  B3=exam3_${RUN}_u${UU}_ideal_line3_n256; B4=exam4_${RUN}_u${UU}_ideal_race4_n256
  if exam_one $B3 line3 "F1SIM_EXAM_SAVE_MEM=1"; then
    (source /home/shchon11/F1tenth_E2E/.venv/bin/activate; cd /home/shchon11/F1tenth_E2E/f1sim
     python3 -W ignore $T/exam_boxes.py $D/exam/$B3.npz > $D/analysis/${RUN}_u${UU}_boxes.txt 2>&1
     { python3 -W ignore $T/exam_table.py $D/exam/$B3.npz; python3 -W ignore $T/side_flip.py $D/exam/$B3.npz;
       sed -n '/(나)/,/(다)/p' $D/analysis/${RUN}_u${UU}_boxes.txt | grep -v '(다)'; } > $D/analysis/${RUN}_u${UU}_exam3.txt 2>&1)
    echo done > $D/analysis/${RUN}_u${UU}_exam3.done
  fi
  if exam_one $B4 race4 "${EXAM4_ENV:-}"; then
    (source /home/shchon11/F1tenth_E2E/.venv/bin/activate; cd /home/shchon11/F1tenth_E2E/f1sim
     python3 -W ignore $T/exam_race_table.py $D/exam/$B4.npz > $D/analysis/${RUN}_u${UU}_exam4.txt 2>&1)
    echo done > $D/analysis/${RUN}_u${UU}_exam4.done
  fi
  B4H=${B4}_head
  if [ "${EXAM4_HEAD:-}" = 1 ] && exam_one $B4H race4 ""; then
    (source /home/shchon11/F1tenth_E2E/.venv/bin/activate; cd /home/shchon11/F1tenth_E2E/f1sim
     python3 -W ignore $T/exam_race_table.py $D/exam/$B4H.npz > $D/analysis/${RUN}_u${UU}_exam4head.txt 2>&1)
    echo done > $D/analysis/${RUN}_u${UU}_exam4head.done
  fi
  echo "$(date +%T) $RUN u$UU exam3 $([ -e $D/exam/$B3.npz ] && echo 끝 || echo 실패) exam4${EXAM4_ENV:+($EXAM4_ENV)} $([ -e $D/exam/$B4.npz ] && echo 끝 || echo 실패)$([ "${EXAM4_HEAD:-}" = 1 ] && echo " exam4(head) $([ -e $D/exam/$B4H.npz ] && echo 끝 || echo 실패)")"
done
