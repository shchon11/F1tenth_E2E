#!/bin/bash
# 학습의 "세게" 문턱을 판정 기준에 맞춤 (11-02 22:30): racep·racev·racel 모두 0.6배속 추월은 지키지만 세게 부딪힘이 100회 내내 안 줆.
# 측정(tools/car_impacts.py, racep u100 vs 1.0배속 20판): 콘솔이 세게(>= 1.0 m/s, contact_severity.HARD_MPS)로 센 19번 중 학습이 값을
# 매긴 것은 9번 -- 9번은 1.0~1.5 m/s라 학습에서 공짜(contact_severity_min 1.5), 1번은 우리가 0.6 m 넘게 앞서 공짜. 판정 기준의 절반을
# 학습이 안 보는 것. 바꿈: --severity-min 1.0 하나(보상·충돌 예측 머리·통계가 모두 이 문턱을 따름). 나머지는 racep와 같음.
# 기전 기각: u100 콘솔 세게(같은 20판)가 racep u100(0.6배속 10, 1.0배속 12)의 절반 아래로 안 내려가면 기각.
# 쓸 만함 기준(전과 같음): 0.6배속 추월 >= 15/20, 추월당 세게 <= 0.2, 1.0배속 세게 <= 2.
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
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_raceh --stage race $PL $COMMON --updates 100 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed ${S}7 --severity-min 1.0 \
  --opp-present 1.0 --car-channel 0 --corridor-car-st 1 --blank-opp 1 --scan-nocars 0 --pos-bonus 80 --opp-slot '{"speed_scale": [0.4, 1.05]}' \
  --base $R/${TAG}_solo2/state_u300.pt --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_raceh.log 2>&1 || exit 1
echo done > $D/${TAG}_raceh.done
