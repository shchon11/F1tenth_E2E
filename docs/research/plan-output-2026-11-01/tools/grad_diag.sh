#!/bin/bash
# gradient conflict 측정 (11-04 21:45, 사용자: "gradient conflict ?"). srp25ows가 느려지던 구간을 그대로(빠른 solo policy 시작, srp25ows 설정, 네 경우 world)
# 20 update, --grad-diag 1로 update마다 네 경우별 policy gradient의 크기, 서로의 cosine, 각 경우를 반으로 나눈 두 gradient의 cosine을 기록.
# 판정(미리 정함): 빈 트랙 반쪽끼리 cosine > 0.3 이고 빈 트랙 vs obstacle·차 cosine < 0 -> 충돌(PCGrad로 고침). 반쪽끼리 < 0.1 -> 빈 트랙 신호 부족.
# 인자: 이름 GPU 환경수 업데이트
NAME=$1; GPU=$2; ENVS=$3; U=$4
M=real:korea_2025_iccas
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage race --action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 \
  --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0 --envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 \
  --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 --noise-corr 0 --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.5 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 --severity-min 1.0 \
  --prop-touch-price 40 --prop-bank $R/prop_banks --updates $U --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 \
  --seed 5701 --opp-present 0.5 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.2]}' \
  --grad-diag 1 --init $R/srp25h2_solo2/state_u300.pt > $D/${NAME}.log 2>&1
echo done > $D/${NAME}.done
