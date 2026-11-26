#!/bin/bash
# (11-27) exam3 of a run trained on ACE, on asus: the lab loop (lab_exam2m_loop.sh) runs the checkpoint's own world json, which
# for an ACE run names NAS bank paths -> here F1SIM_EXAM_WORLD forces the asus copy of the qualifier world. Same exam and the same
# outputs as the lab loop (out/exam/exam3_RUN_uU_ideal_line3_n256.npz, analysis/RUN_uU_{boxes,exam3}.txt + .done), so the site
# and the tables read them as usual. Waits for each checkpoint (fetched by ace_run.sh fetch), then for an asus slot: no other
# plan_rl / exam_record on GPU $GPU's chain and >= 6 GB available.  usage: asus_exam3_loop.sh RUN U0 STEP U1 [GPU]
RUN=$1; U0=$2; STEP=$3; U1=$4; GPU=${5:-1}; EX=exam3
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; D=$R/out; RUNS=/home/shchon11/f1sim_runs
W=/home/shchon11/F1tenth_E2E/f1sim/f1sim/learn/configs/plan_world_korea26scan_qual.json
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
for UU in $(seq $U0 $STEP $U1); do
  C=$RUNS/$RUN/plan_u$UU.pt
  until [ -e $C ]; do sleep 60; done
  sleep 30
  B=${EX}_${RUN}_u${UU}_ideal_line3_n256
  [ -e $D/exam/$B.npz ] && continue
  while true; do
    n=$(ps -eo args | grep -c "[f]1sim.learn.plan_rl"); n2=$(ps -eo args | grep -c "[e]xam_record.py"); av=$(free -g | awk '/Mem:/{print $7}')
    [ "$n" -le 1 ] && [ "$n2" -eq 0 ] && [ "$av" -ge 6 ] && break; sleep 60
  done
  (cd /home/shchon11/F1tenth_E2E/f1sim && PYTHONPATH=$PWD F1SIM_EXAM_WORLD=$W WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 \
     F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555 CUDA_VISIBLE_DEVICES=$GPU timeout 5400 nice $PY -u -W ignore \
     $T/exam_record.py $C $D/exam/$B.npz ideal 256 line3 > $D/exam/$B.log 2>&1)
  if [ -e $D/exam/$B.npz ]; then
    (cd /home/shchon11/F1tenth_E2E/f1sim
     $PY -W ignore $T/exam_boxes.py $D/exam/$B.npz > $D/analysis/${RUN}_u${UU}_boxes.txt 2>&1
     { $PY -W ignore $T/exam_table.py $D/exam/$B.npz; $PY -W ignore $T/side_flip.py $D/exam/$B.npz;
       sed -n '/(나)/,/(다)/p' $D/analysis/${RUN}_u${UU}_boxes.txt | grep -v '(다)'; } > $D/analysis/${RUN}_u${UU}_${EX}.txt 2>&1)
    echo done > $D/analysis/${RUN}_u${UU}_${EX}.done
  fi
  echo "$(date +%T) $RUN u$UU $EX $([ -e $D/exam/$B.npz ] && echo 끝 || echo 실패)"
done
