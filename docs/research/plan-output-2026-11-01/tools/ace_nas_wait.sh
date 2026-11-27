#!/bin/bash
# (11-27 22:10) ACE's NAS (10.77.0.200) stopped answering ~21:45 (ping 100 % loss; hs43a / nc45a blocked in D state). The fetch
# loops and live players were stopped so they do not pile more hung processes onto ACE; this waits until the NAS answers again,
# then restarts them. Checks every 5 min.
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools; O=$T/../out/rq
while true; do
  r=$(timeout 60 ssh -o BatchMode=yes shchon11@10.100.0.2 'timeout 50 ssh -o BatchMode=yes ACE "timeout 8 stat -c %Y /nas/workspace/shchon11_f1sim/runs/nc45a.log >/dev/null && echo up"' 2>/dev/null)
  if [ "$r" = up ]; then
    echo "$(date +%T) NAS 돌아옴"
    for n in hs43a nc45a; do
      setsid nohup bash $T/ace_run2.sh fetch $n 600 >> $O/${n}_fetch.out 2>&1 < /dev/null &
      setsid nohup /home/shchon11/F1tenth_E2E/.venv/bin/python -u $T/ace_live_play.py $n >> $O/${n}_play.out 2>&1 < /dev/null &
    done
    exit 0
  fi
  echo "$(date +%T) NAS 아직"; sleep 300
done
