#!/bin/bash
# 순위 보상 (11-02 15:40): racec·racew 모두 u50에 0.6배속 차 20/20 추월 -> u100 4~5/20(따라가기). 보상이 진행 - 80 x 세게뿐이라 0.6배속
# 추월값(~22, 10 s 지평) = 추월당 0.3번 세게의 기대 손해(~24): 학습이 안전한 쪽(따라가기)으로 감. 대회 목표는 앞서기인데 환경이 순위를 안 줌.
# 바꿈: --pos-bonus 80 -- 8 m 안의 차를 1 m 문턱으로 앞지르면 +80, 추월당하면 -80(제로섬, 범위 밖·리셋·진입 땐 0). 나머지는 racew와 같음.
# 계산: 느린 차(0.3번) +78 -> 추월, 같은 속도 차(1.3번) -24 -> 따라감. 가설: 학습이 추월을 지우지 않고 부딪힘을 줄이는 쪽으로 감.
# 기각(u50·u100, 같은 20판): 0.6배속 추월 < 15/20, 또는 추월당 세게 > 0.2, 또는 1.0배속 세게 > 2.
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
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_racep --stage race $PL $COMMON --updates 100 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed ${S}7 \
  --opp-present 1.0 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.05]}' \
  --base $R/${TAG}_solo2/state_u300.pt --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_racep.log 2>&1 || exit 1
echo done > $D/${TAG}_racep.done
