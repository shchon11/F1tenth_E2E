#!/bin/bash
# 상자 접촉 대가 이어 학습 2단계 (11-02 06:45): 100회 결과(Korea 2025 30배치) -- 상자 밀기 0(26회 -> 0, 멈춤 없음)이지만
# 깨끗한 배치 랩 평균 +0.7 s, 벽에 세게 10회(3배치, 상자를 크게 피하다 반대편 벽). 대가가 바뀐 직후의 조심스러움이 더 학습하면
# 풀리는지(혼자 단계도 수백 회 뒤 속도가 돌아왔음): 같은 설정으로 200회 더. 인자: 맵 이름머리 GPU 시드
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
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_propst2 --stage props $PL $COMMON --updates 200 \
  --lr 1e-4 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --prop-touch-price 40 --seed ${S}5 \
  --init $R/${TAG}_propst/state_u100.pt > $D/${TAG}_propst2.log 2>&1 || exit 1
mkdir -p $R/${TAG}_propst2mix
python3 - <<PY
import torch
ck = torch.load("$R/${TAG}_propst2/state_u200.pt", map_location="cpu", weights_only=False)
ck["args"]["base"] = "$R/${TAG}_solo2/state_u300.pt"; ck["args"]["base_reach"] = 10.0; ck["args"]["base_hold"] = 10
torch.save(ck, "$R/${TAG}_propst2mix/state_u200.pt")
print("혼합 저장", "$R/${TAG}_propst2mix/state_u200.pt")
PY
echo done > $D/${TAG}_propst2.done
