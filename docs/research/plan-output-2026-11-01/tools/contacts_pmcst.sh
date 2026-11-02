#!/bin/bash
# pmcst(앞차를 상자 통로에) 0.6배속 20판에서 세게 부딪힌 순간마다 확대 그림 (11-02 11:15). 원인 분류용.
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E
CUDA_VISIBLE_DEVICES=1 python3 $D/tools/race_contacts.py /home/shchon11/f1sim_runs/srp25h2_pmcst/state_u200.pt real/iccas25 30 20 forzaeth 0.6 777 $D/out/races/zoom_pmcst_x0.6 2>&1 | grep -v Warning > $D/out/contacts_pmcst.log
echo done > $D/out/contacts_pmcst.done
