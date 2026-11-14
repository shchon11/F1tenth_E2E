#!/bin/bash
# (11-15 01:45) plan26p가 u250 checkpoint를 남기면 멈추고(평가 반복은 남은 checkpoint까지 마저 함) plan26q를 띄움.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
until [ -e /home/shchon11/f1sim_runs/plan26p/plan_u250.pt ]; do sleep 15; done
sleep 20                                                   # the save is finished writing
for p in 58216 58217; do grep -q "plan26p" /proc/$p/cmdline 2>/dev/null && { kill $p; kill -CONT $p 2>/dev/null; }; done   # the two trainers, checked by PID
sleep 10
touch $R/out/guard4070_1114p.done $R/out/one_plan26p.done
for p in 58213 58214 58216 58217; do grep -q "plan26p" /proc/$p/cmdline 2>/dev/null && kill -CONT $p 2>/dev/null; done
sleep 20
cd $R && bash tools/launch_plan26q.sh
echo "$(date +%T) plan26p 멈춤(u250), plan26q 시작" >> $R/out/switch_plan26q.log
