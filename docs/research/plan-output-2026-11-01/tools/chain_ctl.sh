#!/bin/bash
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
bash $T/gpu_guard3.sh "state_[r]l --name srp25h2_propctl" $D/srp25h2_propctl.done > $D/gpu_guard_ctl.log 2>&1 &
bash $T/props_ctl.sh real:korea_2025_iccas srp25h2 0 95 || exit 1
bash $T/noise_touch.sh $R/srp25h2_propctl/state_u100.pt real:korea_2025_iccas 1 > $D/noise_tch25c.log 2>&1
bash $T/layouts_run.sh tch25c $R/srp25h2_propctlmix/state_u100.pt real/iccas25 40 30 777 > $D/layouts_tch25c.log 2>&1
echo done > $D/chain_ctl.done
