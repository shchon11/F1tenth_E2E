#!/bin/bash
# 11-02 15:40 연쇄 (racel, 순위 보상 + 라인 예측, 5060): 4070(지킴이) srp25h2_racel 학습 100회 / 5060: u50·u100이 나올 때마다 같은 20판(0.6배속, 1.0배속)
# + 혼자 60 s, 끝에 u100 상자 30배치
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
T=$D/tools
R=/home/shchon11/f1sim_runs
rm -f $D/out/srp25h2_racel.done
bash $T/race_line.sh real:korea_2025_iccas srp25h2 1 96 1024
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E
L=$D/out/races_racel.log
: > $L
run() {  # 이름 체크포인트 맵 배율
  CUDA_VISIBLE_DEVICES=1 python3 $T/console_races.py $2 $3 30 20 forzaeth $4 777 $D/out/races/$1_x$4 2>&1 | grep -v Warning | sed "s/^/$1 x$4 /" >> $L
}
for U in 50 100; do
  until [ -f $R/srp25h2_racel/state_u$U.pt ] || [ -f $D/out/srp25h2_racel.done ] || ! pgrep -f "state_[r]l --name srp25h2_racel" > /dev/null; do sleep 60; done
  [ -f $R/srp25h2_racel/state_u$U.pt ] || continue
  sleep 20
  run racel_u$U $R/srp25h2_racel/state_u$U.pt real/iccas25 0.6
  run racel_u$U $R/srp25h2_racel/state_u$U.pt real/iccas25 1.0
  CUDA_VISIBLE_DEVICES=1 python3 $T/console_laps.py $R/srp25h2_racel/state_u$U.pt real/iccas25 60 - 2>&1 \
    | grep -v Warning | grep "주행" | sed "s/^/racel_u$U 혼자 /" >> $L
done
# 상자 30배치(콘솔 리셋과 같음, 시드 777, 40 s) -- 지금 추천(상자 선생)과 같은 판
CUDA_VISIBLE_DEVICES=1 python3 $T/console_layouts.py $R/srp25h2_racel/state_u100.pt real/iccas25 40 30 777 $D/out/layouts_racel_u100 2>&1 \
  | grep -v Warning | sed "s/^/racel_u100 배치 /" >> $L
echo done > $D/out/chain_racel.done
