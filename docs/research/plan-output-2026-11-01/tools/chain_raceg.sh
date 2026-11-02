#!/bin/bash
# 4070(지킴이): gamma 0.999 상대 단계 100회 -> 5060: 같은 20판 x 30 s vs forzaeth 0.6 / 1.0, 혼자 60 s
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
bash $T/gpu_guard3.sh "state_[r]l --name srp25h2_raceg" $D/srp25h2_raceg.done > $D/gpu_guard_raceg.log 2>&1 &
bash $T/race_gamma.sh real:korea_2025_iccas srp25h2 0 96 1024 || exit 1
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E/f1sim
C=$R/srp25h2_raceg/state_u100.pt
echo "raceg 혼자: $(CUDA_VISIBLE_DEVICES=1 timeout 900 python3 $T/console_laps.py $C real/iccas25 60 2>&1 | grep '주행\|Error')" > $D/races_raceg.log
for SC in 0.6 1.0; do
  CUDA_VISIBLE_DEVICES=1 timeout 3000 python3 $T/console_races.py $C real/iccas25 30 20 forzaeth $SC 777 $D/races/raceg_x$SC 2>&1 | grep "판\|합계\|Error" | sed "s/^/raceg x$SC /" >> $D/races_raceg.log
done
echo done > $D/chain_raceg.done
