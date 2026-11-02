#!/bin/bash
# 11-02 12:00 연쇄: 4070(지킴이) srp25h2_racec 학습 150회 / 5060: 본 시험(corridor_car.sh) 끝나면 ×1.2 pmcst 학습 없는 시험,
# 그다음 racec u50·u100·u150이 나올 때마다 같은 20판(0.6배속, 1.0배속) + 혼자 60 s
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
T=$D/tools
R=/home/shchon11/f1sim_runs
rm -f $D/out/srp25h2_racec.done
bash $T/gpu_guard3.sh "state_[r]l --name srp25h2_racec" $D/out/srp25h2_racec.done > $D/out/gpu_guard_racec.log 2>&1 &
bash $T/race_corridor.sh real:korea_2025_iccas srp25h2 0 96 1024 &
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E
L=$D/out/races_racec.log
: > $L
run() {  # 이름 체크포인트 맵 배율
  CUDA_VISIBLE_DEVICES=1 python3 $T/console_races.py $2 $3 30 20 forzaeth $4 777 $D/out/races/$1_x$4 2>&1 | grep -v Warning | sed "s/^/$1 x$4 /" >> $L
}
until [ -f $D/out/corridor_car.done ]; do sleep 30; done
run pmcst26x $R/srp26xh2_pmcst/state_u200.pt real/korea26x12 0.6
run pmcst26x $R/srp26xh2_pmcst/state_u200.pt real/korea26x12 1.0
for U in 50 100 150; do
  until [ -f $R/srp25h2_racec/state_u$U.pt ] || [ -f $D/out/srp25h2_racec.done ] || ! pgrep -f "state_[r]l --name srp25h2_racec" > /dev/null; do sleep 60; done
  [ -f $R/srp25h2_racec/state_u$U.pt ] || continue
  sleep 20
  run racec_u$U $R/srp25h2_racec/state_u$U.pt real/iccas25 0.6
  run racec_u$U $R/srp25h2_racec/state_u$U.pt real/iccas25 1.0
  CUDA_VISIBLE_DEVICES=1 python3 $T/console_laps.py $R/srp25h2_racec/state_u$U.pt real/iccas25 60 - 2>&1 \
    | grep -v Warning | grep "주행" | sed "s/^/racec_u$U 혼자 /" >> $L
done
echo done > $D/out/chain_racec.done
