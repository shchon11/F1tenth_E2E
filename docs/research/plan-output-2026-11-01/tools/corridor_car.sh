#!/bin/bash
# 앞차를 상자 통로에 (11-02 12:00): 상대 단계는 할인율과 무관하게 따라가기만 배움(0.6배속 차 추월 raceb 1/20, raceg 0/20). 상자 선생의
# "추월"은 차 입력 없이(car_channel 0) 라이다로만 보고 들이받은 것이고, 상대 단계는 새 차 입력(0 가중치)으로 차를 처음부터 배워 물러서기만
# 익혔다. 가설: 차를 "우리가 닿을 때 있을 자리의 상자"(car_spacetime_block)로 상자 통로에 넣으면 상자 선생의 비켜 가기가 학습 없이 추월이
# 된다. 비교: 상대 정보만 빈 트랙처럼 비운 것(pmbo) / 그 위에 통로에 차(pmcst) / 원래 상자 선생(propsmix, 기존 기록) / 상대 단계(raceb).
# 같은 콘솔 경주 20판 x 30 s, 시드 777, 5060. 0.05배속 = 거의 선 차(사고 차).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E
T=$D/tools/console_races.py
L=$D/out/races_corridor.log
: > $L
run() {  # 이름 체크포인트 배율
  CUDA_VISIBLE_DEVICES=1 python3 $T $2 real/iccas25 30 20 forzaeth $3 777 $D/out/races/$1_x$3 2>&1 | grep -v Warning | sed "s/^/$1 x$3 /" >> $L
}
run pmcst $R/srp25h2_pmcst/state_u200.pt 0.6
run pmcst $R/srp25h2_pmcst/state_u200.pt 1.0
run pmbo $R/srp25h2_pmbo/state_u200.pt 0.6
run pmbo $R/srp25h2_pmbo/state_u200.pt 1.0
run pmcst $R/srp25h2_pmcst/state_u200.pt 0.05
run props $R/srp25h2_propsmix/state_u200.pt 0.05
run raceb $R/srp25h2_raceb/state_u100.pt 0.05
echo done > $D/out/corridor_car.done
