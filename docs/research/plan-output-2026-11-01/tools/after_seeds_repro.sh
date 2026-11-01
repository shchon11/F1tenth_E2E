#!/bin/bash
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
until [ -f $D/seeds_25.done ]; do sleep 15; done
bash $T/gui_repro.sh pick_sst25h2 /home/shchon11/f1sim_runs/sst_srp25h2_props/student_i80.pt real/iccas25 40 $D/gui_sst25
echo done > $D/gui_repro.done
