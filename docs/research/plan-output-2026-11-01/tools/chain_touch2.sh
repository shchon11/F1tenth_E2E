#!/bin/bash
# 4070: ×1.2 상자 접촉 대가 100회가 끝나면 Korea 2025를 200회 더 (지킴이), 끝나면 5060에서 같은 30배치 + 혼자 60 s
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
until [ -f $D/chain_touch_4070.done ]; do sleep 30; done
bash $T/gpu_guard3.sh "state_[r]l --name srp25h2_propst2" $D/srp25h2_propst2.done > $D/gpu_guard_touch2.log 2>&1 &
bash $T/props_touch2.sh real:korea_2025_iccas srp25h2 0 93 || exit 1
until [ -f $D/chain_eval_5060.done ]; do sleep 30; done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
C=$R/srp25h2_propst2mix/state_u200.pt
cd /home/shchon11/F1tenth_E2E/f1sim
echo "tch25t2 혼자: $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py $C real/iccas25 60 2>&1 | grep '주행\|Error')" > $D/solo_tch25t2.log
bash $T/layouts_run.sh tch25t2 $C real/iccas25 40 30 777 > $D/layouts_tch25t2.log 2>&1
echo done > $D/chain_touch2.done
