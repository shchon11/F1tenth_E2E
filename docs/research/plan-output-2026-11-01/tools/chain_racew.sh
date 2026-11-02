#!/bin/bash
# 11-02 13:45 연쇄 (racew): 4070(지킴이) srp25h2_racew 학습 100회 / 5060: u50·u100이 나올 때마다 같은 20판(0.6배속, 1.0배속)
# + 혼자 60 s, 끝에 u100 상자 30배치
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
T=$D/tools
R=/home/shchon11/f1sim_runs
rm -f $D/out/srp25h2_racew.done
bash $T/gpu_guard3.sh "state_[r]l --name srp25h2_racew" $D/out/srp25h2_racew.done > $D/out/gpu_guard_racew.log 2>&1 &
bash $T/race_world.sh real:korea_2025_iccas srp25h2 0 96 1024 &
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E
L=$D/out/races_racew.log
: > $L
run() {  # 이름 체크포인트 맵 배율
  CUDA_VISIBLE_DEVICES=1 python3 $T/console_races.py $2 $3 30 20 forzaeth $4 777 $D/out/races/$1_x$4 2>&1 | grep -v Warning | sed "s/^/$1 x$4 /" >> $L
}
sleep 180                                   # the trainer is up before the loop asks pgrep
for U in 50 100; do
  until [ -f $R/srp25h2_racew/state_u$U.pt ] || [ -f $D/out/srp25h2_racew.done ] || ! pgrep -f "state_[r]l --name srp25h2_racew" > /dev/null; do sleep 60; done
  [ -f $R/srp25h2_racew/state_u$U.pt ] || continue
  sleep 20
  run racew_u$U $R/srp25h2_racew/state_u$U.pt real/iccas25 0.6
  run racew_u$U $R/srp25h2_racew/state_u$U.pt real/iccas25 1.0
  CUDA_VISIBLE_DEVICES=1 python3 $T/console_laps.py $R/srp25h2_racew/state_u$U.pt real/iccas25 60 - 2>&1 \
    | grep -v Warning | grep "주행" | sed "s/^/racew_u$U 혼자 /" >> $L
done
# 상자 30배치(콘솔 리셋과 같음, 시드 777, 40 s) -- 지금 추천(상자 선생)과 같은 판
CUDA_VISIBLE_DEVICES=1 python3 $T/console_layouts.py $R/srp25h2_racew/state_u100.pt real/iccas25 40 30 777 $D/out/layouts_racew_u100 2>&1 \
  | grep -v Warning | sed "s/^/racew_u100 배치 /" >> $L
echo done > $D/out/chain_racew.done
