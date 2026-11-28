#!/bin/bash
# Audit C chain 2 (after chain 1): baselines driving the ego on the same qualifier sessions, and 1v1 vs FTG / ForzaETH
S=/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad/C
PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
RUNS=/home/shchon11/f1sim_runs
export CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim
until [ -e $S/rec/chain1.done ]; do sleep 20; done
Q1=$RUNS/pick_scan31b_u475/state_latest.pt
R1=$RUNS/pick_race33b_u975/state_latest.pt
ego() {  # name kind stress
  local NAME=$1 K=$2 ST=$3
  [ -e $S/rec/$NAME.npz ] && return
  echo "$(date +%T) start $NAME"
  env C_EGO=$K C_STRESS=$ST F1SIM_EXAM_ALL_BOXES=1 timeout 3600 $PY -u -W ignore $S/run_ego.py $Q1 $S/rec/$NAME.npz 256 line3 120 60 > $S/rec/$NAME.log 2>&1
  echo "$(date +%T) end $NAME rc $?"
}
vs() {  # name ckpt slotjson
  local NAME=$1 CK=$2 SLOT=$3
  [ -e $S/rec/$NAME.npz ] && return
  echo "$(date +%T) start $NAME"
  env F1SIM_EXAM_ALL_BOXES=1 F1SIM_EXAM_HALF_LAP=1 F1SIM_EXAM_START_REST=1 F1SIM_EXAM_OPP_SLOT="$SLOT" timeout 3600 $PY -u -W ignore $S/run_exam.py $CK $S/rec/$NAME.npz map 256 line3 120 60 > $S/rec/$NAME.log 2>&1
  echo "$(date +%T) end $NAME rc $?"
}
FTG_RACE='{"kind": "ftg", "kind_mix": [], "reactive": {}, "events": [], "event_rate": 0.0, "speed_scale": [1.0, 1.0], "spawn": "ahead", "present": 1.0, "options": {"max_speed": 7.210195504919446, "safety_radius": 17, "max_lidar_dist": 8.086843513821734, "track_width": 1.691335212526323, "radius_mode": "stopping_passable", "r_min": 0.7126292306388392, "r_max": 4.532010673481452, "t_react": 0.39939999594014036, "a_brake": 7.091946230811433, "scan_quantile": 0.6041039357596871, "scan_gain": 0.4625910516959505, "gap_margin": 0.23808984327171676, "target": "middle"}}'
FZ_RACE='{"kind": "forzaeth", "kind_mix": [], "reactive": {}, "events": [], "event_rate": 0.0, "speed_scale": [1.0, 1.0], "spawn": "ahead", "present": 1.0}'
ego ego_ftg_map       ftg      0
ego ego_raceline_map  raceline 0
vs  race_race33b975_vsftg_map $R1 "$FTG_RACE"
vs  race_race33b975_vsfz_map  $R1 "$FZ_RACE"
ego ego_ftg_mapS      ftg      1
ego ego_raceline_mapS raceline 1
echo "$(date +%T) chain2 done"; touch $S/rec/chain2.done
