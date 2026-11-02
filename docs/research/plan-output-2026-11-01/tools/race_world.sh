#!/bin/bash
# 상대 단계 세계를 경주답게 (11-02 13:40): racec 진단(race_world_audit.py) -- 상대 단계 정책 운전 시간의 68 %가 상자 처리, 차 10 m 안은 37 %,
# 학습 목표 하락은 모든 상황에서 고르게 느려진 것(γ 0.999 두 학습만 무너짐). 바꿈: 매 판 상대 차(opp-present 1.0), 상자는 30 % 판에만,
# γ 0.995(원래 값). 나머지(관측, 시작점 = 상자 u200, 보상, 상대 종류·성향·돌발 행동, 혼합)는 racec와 같음.
# 기각(u50·u100, 같은 20판): 0.6배속 추월 < 15/20, 또는 학습 목표가 떨어짐, 또는 상자 30배치에서 지금 추천보다 나빠짐.
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
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_racew --stage race $PL $COMMON --updates 100 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed ${S}7 \
  --opp-present 1.0 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --opp-slot '{"speed_scale": [0.4, 1.05]}' \
  --base $R/${TAG}_solo2/state_u300.pt --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_racew.log 2>&1 || exit 1
echo done > $D/${TAG}_racew.done
