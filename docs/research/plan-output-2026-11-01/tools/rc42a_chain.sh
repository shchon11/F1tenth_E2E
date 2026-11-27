#!/bin/bash
# (11-27) user OK'd ("굿") the race stage with the qualifier's hit price, started from the hc41a checkpoint at the end of its run:
# waits for hc41a u900 on asus (ace_run.sh fetch), launches rc42a on ACE GPU 3, fetches it every 10 min, and examines it on the
# lab PC (exam3 + exam4, lab_race_exam_loop.sh; world_of.py maps the *_ace.json world back to this machine's copy).
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools; O=$T/../out/rq; PY=/home/shchon11/F1tenth_E2E/.venv/bin/python
until [ -e /home/shchon11/f1sim_runs/hc41a/plan_u900.pt ]; do sleep 120; done
sleep 90
echo "$(date +%T) rc42a launch" >> $O/chain.log
cd $T && bash ace_run.sh launch rc42a $($PY rq_race_hc.py args) >> $O/rc42a_launch.out 2>&1
echo "$(date +%T) rc42a launched rc $?" >> $O/chain.log
setsid nohup bash $T/ace_run.sh fetch rc42a 600 > $O/rc42a_fetch.out 2>&1 < /dev/null &
setsid nohup bash $T/lab_race_exam_loop.sh rc42a 950 50 1200 > $O/rc42a_exam.out 2>&1 < /dev/null &
echo "$(date +%T) rc42a fetch + lab exams started" >> $O/chain.log
