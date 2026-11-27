#!/bin/bash
# (11-27, user: "실시간 사이트에 보이게좀") a run training on ACE on the live page: every 3 s pull its live_tap*.json (and once
# args.json + live_tracks.npz) through the lab PC into ~/f1sim_runs/NAME/, where dash_server.py's /live reads them. SSH
# connections are kept open (ControlMaster on both hops) so a pull is one round trip.  usage: ace_live_sync.sh NAME
NAME=$1; [ -n "$NAME" ] || exit 1
LAB=shchon11@10.100.0.2; AR=/nas/workspace/shchon11_f1sim/runs; D=/home/shchon11/f1sim_runs/$NAME
CM="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/cm-ace-live-%r@%h:%p -o ControlPersist=600 -o BatchMode=yes -o ConnectTimeout=20"
RC="ssh -o ControlMaster=auto -o ControlPath=/tmp/cm-ace-live-%r@%h:%p -o ControlPersist=600 -o BatchMode=yes ACE"
mkdir -p "$D"; [ -e "$D/args.json" ] && [ ! -e "$D/.from_ace" ] && { echo "$D is not an ACE run"; exit 1; }
touch "$D/.from_ace"
once=0
while true; do
  if [ $once -eq 0 ]; then
    ssh $CM $LAB "$RC 'cd $AR/$NAME 2>/dev/null && tar -cf - args.json live_tracks.npz 2>/dev/null'" | tar -x -C "$D" 2>/dev/null && [ -e "$D/live_tracks.npz" ] && once=1
  fi
  ssh $CM $LAB "$RC 'cd $AR/$NAME 2>/dev/null && tar -cf - live_tap*.json 2>/dev/null'" | tar -x -C "$D" 2>/dev/null
  # the run on ACE is gone -> stop (its pid file names a dead process)
  if [ $((SECONDS % 300)) -lt 4 ]; then
    alive=$(ssh $CM $LAB "$RC 'p=\$(cat $AR/$NAME.pid 2>/dev/null); [ -n \"\$p\" ] && ps -p \$p >/dev/null && echo y'")
    [ "$alive" = y ] || { echo "$(date +%T) $NAME not running on ACE: stop"; exit 0; }
  fi
  sleep 3
done
