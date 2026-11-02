#!/bin/bash
# 순위 보상 + 앞차를 레이싱 라인 위로 예측 (11-02 17:55): racep·racev는 0.6배속 추월을 지키지만 접촉이 100회 내내 안 줆(세게 10~11,
# 1.0배속 12~13) -- 장면은 모두 머리핀: 앞차가 라인을 따라 출구 안쪽으로 꺾어 우리 차로 들어옴. 지금 표현(가로 위치 고정 예측)엔 그 닫힘이
# 안 보여 학습이 피하는 법을 못 배움. 보상 규칙 대신 표현: --car-st-line 1(라인 위 지금 오프셋, 라인 속도의 지금 비율; 접촉 1.5 s 전
# 닫히는 안쪽 칸을 막음 확인). 나머지는 racep와 같음. 기각(같은 20판): 0.6배속 추월 < 15/20, 추월당 세게 > 0.2, 1.0배속 세게 > 2.
# 인자: 맵 이름머리 GPU 시드 환경수
M=$1; TAG=$2; GPU=$3; S=$4; ENVS=$5
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 25 \
  --std-floor 0.05 0.2 --soft-bound 1 --bound-loss 1.0 --noise-corr 0.9 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.3 --removal-p 0.5 --removal-s 10 30"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_racel --stage race $PL $COMMON --updates 100 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed ${S}7 \
  --opp-present 1.0 --car-channel 0 --corridor-car-st 1 --car-st-line 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.05]}' \
  --base $R/${TAG}_solo2/state_u300.pt --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_racel.log 2>&1 || exit 1
echo done > $D/${TAG}_racel.done
