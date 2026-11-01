#!/bin/bash
# 4070(지킴이): Korea 2025 밀기 대가 100회 -> 5060: 같은 30배치 + 혼자 60 s + 학습 세계 평균/노이즈 평가
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
bash $T/gpu_guard3.sh "state_[r]l --name srp25h2_proppush" $D/srp25h2_proppush.done > $D/gpu_guard_push.log 2>&1 &
bash $T/props_push.sh real:korea_2025_iccas srp25h2 0 94 || exit 1
until [ -f $D/chain_eval_5060.done ]; do sleep 30; done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
C=$R/srp25h2_proppushmix/state_u100.pt
cd /home/shchon11/F1tenth_E2E/f1sim
echo "tch25p 혼자: $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py $C real/iccas25 60 2>&1 | grep '주행\|Error')" > $D/solo_tch25p.log
bash $T/layouts_run.sh tch25p $C real/iccas25 40 30 777 > $D/layouts_tch25p.log 2>&1
bash $T/noise_touch.sh $R/srp25h2_proppush/state_u100.pt real:korea_2025_iccas 1 > $D/noise_tch25p.log 2>&1
echo done > $D/chain_push.done
