#!/bin/bash
# 상대 단계 강화학습을 추월하는 선생에서 (11-02 12:00): 지금까지 상대 단계는 차를 못 보는 정책(새 차 입력 0 가중치)에서 시작해 "물러서기"만
# 배웠다(0.6배속 추월 raceb 1/20, raceg 0/20). 앞차를 상자 통로에 넣은 상자 선생(pmcst)은 학습 없이 20/20 추월하지만 추월당 세게 0.38,
# 1.0배속 차도 무리하게 추월(세게 12). 가설: 추월하는 정책에서 출발하고 지평을 추월값이 지금 위험보다 크게(γ 0.999: 추월 ~52-66 > 0.38 x 80)
# 하면 학습이 추월을 버리지 않고 깨끗하게 다듬고 빠른 차는 따라가는 판단을 배운다. 관측 = pmcst와 같음(차는 상자 통로에, 상대 칸 비움).
# 기각(u100·u150, console_races 같은 20판): 0.6배속 추월 < 15/20, 또는 추월당 세게가 0.38 아래로 안 줆, 또는 1.0배속 세게 > 2.
# 인자: 맵 이름머리 GPU 시드 환경수
M=$1; TAG=$2; GPU=$3; S=$4; ENVS=$5
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.999 --lam 0.95 --clip 0.2 --save-every 25 \
  --std-floor 0.05 0.2 --soft-bound 1 --bound-loss 1.0 --noise-corr 0.9 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_racec --stage race $PL $COMMON --updates 150 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed ${S}7 \
  --opp-present 0.5 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --opp-slot '{"speed_scale": [0.4, 1.05]}' \
  --base $R/${TAG}_solo2/state_u300.pt --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_racec.log 2>&1 || exit 1
echo done > $D/${TAG}_racec.done
