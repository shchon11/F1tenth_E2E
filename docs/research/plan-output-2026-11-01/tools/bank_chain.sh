#!/bin/bash
# 은행 두 개가 끝나면 (11-03 밤): 은행 세계 빈도 지도 + 통과 시험 (prop_freq.py, 두 지도), 그다음 두 지도 상자 단계 재학습 (props_bank.sh).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
until [ -e $D/bank_korea25.done ] && [ -e $D/bank_korea26.done ]; do sleep 30; done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
CUDA_VISIBLE_DEVICES=0 python3 tools/prop_freq.py $R/srp25h3gwt_props/state_u100.pt 256 60 out/prop_freq_korea25_bank.png "한국 2025" $R/prop_banks > out/prop_freq_bank25.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 tools/prop_freq.py $R/srp26gwt_props/state_u100.pt 256 60 out/prop_freq_korea26_bank.png "한국 2026" $R/prop_banks > out/prop_freq_bank26.log 2>&1 &
wait
echo done > $D/prop_freq_bank.done
setsid nohup bash $T/props_bank.sh 25 srp25bk_props $R/srp25h3gwt_props/state_u100.pt 0 1024 280 1 > /dev/null 2>&1 < /dev/null &
setsid nohup bash $T/props_bank.sh 26 srp26bk_props $R/srp26gwt_props/state_u100.pt 1 1024 281 0 > /dev/null 2>&1 < /dev/null &
