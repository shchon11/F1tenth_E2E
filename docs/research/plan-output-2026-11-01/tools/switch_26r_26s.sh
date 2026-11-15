#!/bin/bash
# (11-15 09:52) plan26r u600 시험이 끝나면 plan26r을 멈추고(평가 반복은 남은 checkpoint까지) plan26s를 띄움. PID는 확인 후 멈춤.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
until [ -e $R/out/analysis/plan26r_u600_exam.done ]; do sleep 20; done
for p in 136774 136775; do grep -q "plan26r" /proc/$p/cmdline 2>/dev/null && { kill $p; kill -CONT $p 2>/dev/null; }; done
sleep 10
touch $R/out/guard4070_1115r.done $R/out/one_plan26r.done
for p in 136771 136772 136774 136775; do grep -q "plan26r" /proc/$p/cmdline 2>/dev/null && kill -CONT $p 2>/dev/null; done
sleep 20
cd $R && bash tools/launch_plan26s.sh
echo "$(date +%T) plan26r 멈춤(u600 뒤), plan26s 시작" >> $R/out/switch_plan26s.log
