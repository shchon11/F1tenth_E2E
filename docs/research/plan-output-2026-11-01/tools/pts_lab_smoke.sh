#!/bin/bash
# (11-25) plan_rl --arch pts smoke on the lab PC: two ranks of --dist-world 2 on its one GPU, small env counts, a few updates,
# from scratch, the proposed pts1a flags otherwise (copies join early, scenes from the start, so every path runs).
# usage (on the lab, from the code copy's root): bash docs/research/plan-output-2026-11-01/tools/pts_lab_smoke.sh NAME [ENVS0 ENVS1 UPDATES HORIZON EXTRA...]
#   ENVS1 = 0: one process, no --dist-world (when the lab's GPU has room for one only)
# Refuses while an exam runs; one 10 GB scope for both ranks; a watchdog stops both (by PID) below 3 GB of available RAM.
N=${1:?name}; E0=${2:-48}; E1=${3:-32}; U=${4:-6}; H=${5:-16}; shift $(( $# < 5 ? $# : 5 )); EXTRA=("$@")
R=$(cd "$(dirname "$0")/../../../.." && pwd)
O=$R/out; mkdir -p "$O"
PY=$HOME/F1tenth/.venv/bin/python3
W=$R/f1sim/f1sim/learn/configs/plan_world_korea26scan_race.json
if [ "$(ps -eo args | grep -c '[e]xam_record')" != 0 ]; then echo "exam running: not started"; exit 1; fi
PORT=$((54000 + RANDOM % 1000))
cat > "$O/$N.job" <<EOF
export PYTHONPATH=$R/f1sim F1SIM_RACELINE_CACHE_ONLY=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=(--world $W --arch pts --updates $U --horizon $H --epochs 4 --minibatches 2 --accum-actor-auto --accum-ratchet
   --horizon-auto-mult 2.0 --horizon-auto 4 --horizon-auto-min 2 --horizon-auto-down --mem-chunk-rollout --mem-prev 20
   --noise-probe-every 1 --opp-every 4 --accum-auto-frac 0.5 --lr 3e-4 --lr-end 1e-4 --std0 0.6
   --noise-corr 0.9 --noise-corr-steer --noise-corr-box 1.5,4.0 --noise-corr-car 2.0,4.0 --online-boxes 0.5 --online-late 4.0
   --contact-scale 0.82 --plan-hit-share 0.0 --tail-consistency 1.0 --tail-target driven --tail-trunk --tail-trunk-ratio 0.5
   --memory 128 --plan-lat-margin 0.95 --box-range 12 --mem-value 0.15 --speed-std-floor 0.4 --steer-std-floor 0.3
   --shaping-fade 100,150 --opp-restart-s 2.0 --lidar-mount-x 0.12555 --align-episodes --compile-leaves
   --scene-replay 0.25 --scene-lag 20 --scene-size 1024 --live-tap 0 --live-taps 2 --scene-keep-unsolved --scene-episode-s 13
   --scene-from 0 --self-opp 1.0 --self-opp-every 2 --self-opp-pace 0.5,0.95,0.2 --scene-copies --scene-near-miss --car-fault
   --plan-car-cost 0.1 --plan-car-margin 0.05 --plan-car-w-end 0.3 --removal-p 0.5 --opp-present-ramp 0,3
   --pts-odom-noise 1.0 --free-lr 3e-4 --free-epochs 1 --save-every 3 ${EXTRA[*]})
if [ $E1 -gt 0 ]; then
  /usr/bin/time -v $PY -u -W ignore -m f1sim.learn.plan_rl --name $N --envs $E1 --dist-rank 1 --dist-world 2 --dist-init tcp://127.0.0.1:$PORT "\${A[@]}" > $O/${N}_r1.log 2>&1 &
  P1=\$!
  DW="--dist-world 2 --dist-init tcp://127.0.0.1:$PORT"
else
  P1=""; DW=""; echo "one process" > $O/${N}_r1.log
fi
/usr/bin/time -v $PY -u -W ignore -m f1sim.learn.plan_rl --name $N --envs $E0 --dist-rank 0 \$DW "\${A[@]}" > $O/${N}_r0.log 2>&1 &
P0=\$!
while kill -0 \$P0 2>/dev/null || kill -0 \$P1 2>/dev/null; do
  AV=\$(awk '/MemAvailable/{print int(\$2/1024)}' /proc/meminfo)
  if [ "\$AV" -lt 3072 ]; then echo "WATCHDOG: available \${AV} MB < 3 GB: stopping \$P0 \$P1"; kill \$P0 \$P1 2>/dev/null; break; fi
  sleep 3
done
wait \$P0; echo "rank0 rc \$?"; wait \$P1; echo "rank1 rc \$?"
grep -h "Maximum resident\|Elapsed" $O/${N}_r0.log $O/${N}_r1.log
EOF
setsid nohup bash -c "systemd-run --user --scope -p MemoryMax=10G -- timeout 1800 bash $O/$N.job > $O/$N.log 2>&1; echo rc \$? >> $O/$N.log" > /dev/null 2>&1 < /dev/null &
echo "started $N (logs $O/${N}_r0.log, $O/${N}_r1.log, $O/$N.log)"
