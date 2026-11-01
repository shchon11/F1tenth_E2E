#!/bin/bash
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
until [ -f $D/layouts_sst25.done ]; do sleep 10; done
bash $T/layouts_run.sh tch25 /home/shchon11/f1sim_runs/srp25h2_propsmix/state_u200.pt real/iccas25 40 30 777 > $D/layouts_tch25.log 2>&1
