#!/bin/bash
# 추월과 할인율 (11-02 10:10): 상대 단계(raceb, gamma 0.995 = 20 Hz에서 지평 ~10 s)는 안전하게 따라가기만 배움 -- 0.6배속 차도 20판 중
# 19판을 6~13 m 뒤에서 따라만 감(console_races.py). 0.6배속 차를 앞지르면 남은 시간 내내 2.4 m/s씩 앞서는데, 10 s 지평에선 그 값이 ~24로
# 세게 한 번 부딪힘(80)보다 작아, 시도가 30 % 넘게 부딪히면 따라가기가 낫다. 가설: 지평을 에피소드(40 s)만큼 -> gamma 0.999(~66).
# raceb와 같은 설정에서 gamma만 바꿔 100회. 인자: 맵 이름머리 GPU 시드 환경수
M=$1; TAG=$2; GPU=$3; S=$4; ENVS=$5
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.999 --lam 0.95 --clip 0.2 --save-every 50 \
  --std-floor 0.05 0.2 --soft-bound 1 --bound-loss 1.0 --noise-corr 0.9 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_raceg --stage race $PL $COMMON --updates 100 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed ${S}3 \
  --opp-present 0.5 --car-channel 1 --car-st 1 --scan-nocars 1 --opp-slot '{"speed_scale": [0.4, 1.05]}' \
  --base $R/${TAG}_solo2/state_u300.pt --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_raceg.log 2>&1 || exit 1
echo done > $D/${TAG}_raceg.done
