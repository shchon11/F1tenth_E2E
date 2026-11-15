#!/bin/bash
# (11-15 15:58) plan26s가 u1000에서 끝나면 plan26t를 띄움
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
until [ -e /home/shchon11/f1sim_runs/plan26s/plan_u1000.pt ] && ! grep -q "plan26s" /proc/384969/cmdline 2>/dev/null; do sleep 20; done
sleep 30
touch $R/out/guard4070_1115s.done $R/out/one_plan26s.done
cd $R && bash tools/launch_plan26t.sh
echo "$(date +%T) plan26s 끝(u1000), plan26t 시작" >> $R/out/switch_plan26t.log
