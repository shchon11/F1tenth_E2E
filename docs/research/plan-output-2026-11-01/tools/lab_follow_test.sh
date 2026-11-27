#!/bin/bash
# (11-27) the race stage's 'brake earlier' test: exam4 of RUN/plan_uUU on the lab with F1SIM_EXAM_FOLLOW=G,DV per variant (the
# other car ahead within G m -> the plan's speed <= its speed + DV), records exam4f_<G>_<DV>_<run>_u<UU>_ideal_race4_n256 here.
# The baseline is the ordinary exam4 record of the same checkpoint (same seed). usage: lab_follow_test.sh RUN UU "G,DV" ["G,DV" ...]
RUN=$1; UU=$2; shift 2
H=shchon11@10.100.0.2; SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
T=$(cd "$(dirname "$0")" && pwd); D=$T/../out; RUNS=/home/shchon11/f1sim_runs
LX=/home/shchon11/F1tenth/exam_lab/out/exam; LC=/home/shchon11/F1tenth/console_main/f1sim; LT=/home/shchon11/F1tenth/exam_lab/tools_follow
tar -cf - -C $T exam_record.py world_of.py | $SSH $H "mkdir -p $LT && tar -x -C $LT"
tar -chf - -C $RUNS exam4_ref $RUN/args.json $RUN/plan_u$UU.pt | $SSH $H 'mkdir -p ~/f1sim_runs && tar -x -C ~/f1sim_runs'
ENVS="PYTHONPATH=$LC WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555 CUDA_VISIBLE_DEVICES=0"
for V in "$@"; do
  B=exam4f_${V/,/_}_${RUN}_u${UU}_ideal_race4_n256; B=${B//./p}
  [ -e $D/exam/$B.npz ] && continue
  $SSH $H "mkdir -p $LX && rm -f $LX/$B.npz $LX/$B.log && cd $LC && setsid nohup bash -c 'systemd-run --user --scope -q -p MemoryMax=10G env $ENVS F1SIM_EXAM_FOLLOW=$V timeout 5400 ~/F1tenth/.venv/bin/python3 -u -W ignore $LT/exam_record.py ~/f1sim_runs/$RUN/plan_u$UU.pt $LX/$B.npz ideal 256 race4 > $LX/$B.log 2>&1; echo rc \$? >> $LX/$B.log' > /dev/null 2>&1 < /dev/null &"
  for k in $(seq 1 200); do sleep 30; $SSH $H "grep -q '^rc ' $LX/$B.log" 2>/dev/null && break; done
  $SSH $H "cd $LX && tar -cf - $B.npz $B.log" | tar -x -C $D/exam 2>/dev/null
  echo "$(date +%T) $B $( [ -e $D/exam/$B.npz ] && echo 끝 || echo 실패)"
done
