#!/bin/bash
# Korea 2025: 학생 vs 선생(상자+혼합), 학습 장애물 시드 1~6, 60 s
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
bash $T/console_seeds.sh 학생 /home/shchon11/f1sim_runs/sst_srp25h2_props/student_i80.pt real/iccas25 60 1 2 3 4 5 6
bash $T/console_seeds.sh 선생 /home/shchon11/f1sim_runs/srp25h2_propsmix/state_u200.pt real/iccas25 60 1 2 3 4 5 6
echo done > $D/seeds_25.done
