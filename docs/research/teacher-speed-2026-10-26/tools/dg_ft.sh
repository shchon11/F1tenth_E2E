#!/bin/bash
# dg_ft (10-26): the GENERAL DAgger again, from the FIXED teacher (the user: "teacher부터 잘못됐네").
# What changed against dgv2A_s901 (10-15, the last general v2 DAgger; its recipe otherwise: 149 train tracks, 4-car races with 3
# scripted seats present 0.5, props off the line, min-curvature 7/6.5/4 line, 512 envs x 240 steps, 1600 updates/iter):
#   - no 8 m/s cap (DAgger's old default): the command range, 10 (c5cc9e9 / 7acb936)
#   - the env guard checks walls on the plan only (teacher_walls_within_plan): 9.15 -> 8.74 s solo, 0 contacts
#   - the teacher's speed profile corners at --teacher-profile-a-lat ALAT on the same lines (real car p90 6.4, p99 ~10 m/s^2)
#   - the plans are followed by MAP while collecting (what PPO, the console and the car use), POLICY_MAP on every track
#   - init: the Korea entry (the newest network and heads), as "start from the best checkpoint"
# usage: GPU=.. TEACHER=raceline|interactive ALAT=8.0 dg_ft.sh TAG SEED [ITERS]
set -u
TAG=$1; SEED=$2; ITERS=${3:-10}
OUT=~/f1sim_runs/_eval/mintime-teacher-2026-09-19/pipeline; WT=${WT:-/data/wt/ft}
INIT=${INIT:-/home/shchon11/F1tenth_E2E/checkpoints/korea26_v2_ppo_s2421_u512_nearcar_slow.pt}
SLOT='{"kind_mix":["forzaeth","forzaeth_pred","lane_switch","interactive"],"speed_scale":[0.7,1.05],"present":0.5,"spawn":"random","reactive":{"defend":0.3,"line":0.2,"oblivious":0.1},"events":["brake","stop","shift","weave"],"event_rate":0.5}'
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
cd "$WT/f1sim" && export PYTHONPATH=$WT/f1sim:$WT/f1sim_ros:${PYTHONPATH:-} TORCHINDUCTOR_COMPILE_THREADS=2 CUDA_VISIBLE_DEVICES=${GPU:-0} WANDB_MODE=offline F1SIM_RACELINE_CACHE_ONLY=1
echo "[$(date +%H:%M:%S)] $TAG start (code $(git -C $WT rev-parse --short HEAD), teacher ${TEACHER:-raceline}, profile a_lat ${ALAT:-8.0}, init $INIT)" >> "$OUT/$TAG.log"
python3 -m f1sim.learn.dagger --arch v2 --name "$TAG" --init "$INIT" --tracks train --envs ${ENVS:-512} --iters $ITERS \
   --steps 240 --keep-iters 3 --updates 1600 \
   --race-size 4 --kind-mix-assign partition --opp-slots "[$SLOT,$SLOT,$SLOT]" --opponent-speed-mode linear \
   --teacher ${TEACHER:-raceline} --raceline-objective min_curvature --teacher-a-lat 7.0 --teacher-a-acc 6.5 --teacher-a-brake 4.0 \
   --teacher-profile-a-lat ${ALAT:-8.0} --plan-follower map \
   --procedural-obstacles 1.0 --procedural-density 1.5 --procedural-max-props 10 \
   --cond dial --dial-margin 0.30 --dial-exact 0.30 --collision-mode soft --spawn-runway 3.0 \
   --seed "$SEED" --wandb offline ${DG_EXTRA:-} >> "$OUT/$TAG.log" 2>&1
echo "[$(date +%H:%M:%S)] $TAG exit $?" | tee -a "$OUT/$TAG.log"
grep "^iter" "$OUT/$TAG.log" | tail -3 | cut -c1-200
echo "${TAG}_DONE"
