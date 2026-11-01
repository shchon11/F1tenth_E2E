#!/bin/bash
# 대조 (11-02 08:00): 상자 대가 두 가지(모든 접촉 propst, 느린 접촉만 proppush)가 똑같이 학습 세계 진행을 4.94 -> 4.58 m/s(평균 행동)로
# 늦춤 -- 대가 탓인가, u200에서 새 실행으로 이어 학습(새 Adam, 학습률 1e-4로 되돌림; 원래 단계 끝은 5e-5) 탓인가.
# 같은 설정에서 상자 대가만 뺀 100회. 인자: 맵 이름머리 GPU 시드
M=$1; TAG=$2; GPU=$3; S=$4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs 2048 --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --std-floor 0.05 0.2 --soft-bound 1 --bound-loss 1.0 --noise-corr 0.9 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_propctl --stage props $PL $COMMON --updates 100 \
  --lr 1e-4 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed ${S}7 \
  --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_propctl.log 2>&1 || exit 1
mkdir -p $R/${TAG}_propctlmix
python3 - <<PY
import torch
ck = torch.load("$R/${TAG}_propctl/state_u100.pt", map_location="cpu", weights_only=False)
ck["args"]["base"] = "$R/${TAG}_solo2/state_u300.pt"; ck["args"]["base_reach"] = 10.0; ck["args"]["base_hold"] = 10
torch.save(ck, "$R/${TAG}_propctlmix/state_u100.pt")
PY
echo done > $D/${TAG}_propctl.done
