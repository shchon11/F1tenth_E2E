#!/bin/bash
# 앞차를 레이싱 라인 위로 예측 (11-02 11:40): pmcst(앞차 = 지금 속도·지금 가로 위치 유지)는 0.6배속 차를 20/20 추월했지만 세게 15번,
# 모두 머리핀에서 -- 앞차는 레이싱 라인을 0.01~0.07 m로 따라가며 2 m 안에 0.9 m 안쪽으로 꺾어 우리 차로리 들어왔다. pmcsl = 같은 선생,
# 앞차를 "라인 위 지금 오프셋, 라인 속도의 지금 비율"로 예측(--car-st-line). 같은 20판(시드 777), 5060.
# 반증: 0.6배속 첫 추월 15/20 이상이면서 세게 부딪힘이 추월당 0.2 이하로 줄지 않으면 기각(머리핀 접촉의 원인이 예측이 아님).
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
cd /home/shchon11/F1tenth_E2E
T=$D/tools/console_races.py
L=$D/out/races_line.log
: > $L
run() {  # 이름 체크포인트 맵 배율
  CUDA_VISIBLE_DEVICES=1 python3 $T $2 $3 30 20 forzaeth $4 777 $D/out/races/$1_x$4 2>&1 | grep -v Warning | sed "s/^/$1 x$4 /" >> $L
}
run pmcsl $R/srp25h2_pmcsl/state_u200.pt real/iccas25 0.6
run pmcsl $R/srp25h2_pmcsl/state_u200.pt real/iccas25 1.0
run pmcsl26x $R/srp26xh2_pmcsl/state_u200.pt real/korea26x12 0.6
run pmcsl26x $R/srp26xh2_pmcsl/state_u200.pt real/korea26x12 1.0
run pmcsl $R/srp25h2_pmcsl/state_u200.pt real/iccas25 0.05
echo done > $D/out/corridor_line.done
