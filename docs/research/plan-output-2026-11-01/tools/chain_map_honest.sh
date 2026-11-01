#!/bin/bash
# 정직한 궤적 연쇄 (11-01 자정): 궤적(wp4) -> MAP 20 Hz, 계획 0.6 s(0.3 s는 MAP 조준점이 계획 밖 연장선에 놓여 정직한 궤적을 못 따라감:
# 레이싱 라인 랩 8.2~10.4 s, 0.6 s에서 6.93~7.33 s), --plan-relabel 1.0(경로를 그 뒤 실제로 달린 길에 맞춤; 속도는 맞추지 않음)
# 혼자 150 (지렛대 선생 u250에서) -> 상자 200 -> 상자 + 상대 300 -> 라이다 격자 학생 80
# 인자: 맵  이름 머리  선생 GPU  시드 머리  시작 체크포인트
M=$1; TAG=$2; GPU=$3; S=$4; INIT=$5
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs 2048 --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --std-floor 0.05 0.2 --soft-bound 1 --bound-loss 1.0 --noise-corr 0.9 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_solo --stage solo $PL --envs 2048 --warps 16 --beams 36 --episode-s 40 \
  --updates 150 --horizon 64 --epochs 4 --minibatches 8 --lr 1e-4 --lr-end 5e-5 --gamma 0.995 --lam 0.95 --clip 0.2 \
  --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --save-every 50 --seed ${S}1 --init $INIT > $D/${TAG}_solo.log 2>&1 || exit 1
echo solo > $D/chain_${TAG}.solo_done
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_props --stage props $PL $COMMON --updates 200 \
  --lr 1e-4 --lr-end 5e-5 --price 80 --price-start 20 --price-hold 0 --price-ramp 100 --seed ${S}2 \
  --init $R/${TAG}_solo/state_u150.pt > $D/${TAG}_props.log 2>&1 || exit 1
until [ -f $D/racelines_k.done ]; do sleep 60; done
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_race --stage race $PL $COMMON --updates 300 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed ${S}3 \
  --opp-present 0.5 --car-channel 1 --car-st 1 --scan-nocars 1 --opp-slot '{"speed_scale": [0.4, 1.05]}' \
  --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_race.log 2>&1 || exit 1
flock $D/student.lock env CUDA_VISIBLE_DEVICES=1 python3 -u -m f1sim.learn.state_student --name sst_${TAG} --teacher $R/${TAG}_race/state_u300.pt \
  --stage race --envs 768 --iters 80 --steps 256 --buffer 400000 --buffer-cpu 1 --epochs 3 --frames 4 --stride 4 --danger-weight 4 \
  --opp-present 0.5 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30 --unwedge 0 --grid 1 --seed ${S}9 > $D/sst_${TAG}.log 2>&1 || exit 1
echo done > $D/chain_${TAG}.done
