#!/bin/bash
# 상자 밀기에만 대가 (11-02 07:00): 모든 상자 접촉에 초당 40(props_touch.sh)은 밀기를 없앴지만(Korea 2025 30배치 26회 -> 0)
# 스치는 접촉까지 값이 붙어 정책이 탐색 노이즈만큼 상자와 거리를 두고 느려짐(깨끗한 배치 랩 +0.7 s, 평균 행동 진행 -7 %,
# 상자를 크게 피하다 반대편 벽에 세게 10회). 사용자: "스치는 건 괜찮다" -> 차가 1.5 m/s보다 느릴 때(밀 때 0.3~0.8 m/s,
# 스칠 때 3~5 m/s)만 초당 40 (--prop-touch-vmax 1.5). 상자 200회(u200)에서 이어 100회. 인자: 맵 이름머리 GPU 시드
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
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_proppush --stage props $PL $COMMON --updates 100 \
  --lr 1e-4 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --prop-touch-price 40 --prop-touch-vmax 1.5 --seed ${S}6 \
  --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_proppush.log 2>&1 || exit 1
mkdir -p $R/${TAG}_proppushmix
python3 - <<PY
import torch
ck = torch.load("$R/${TAG}_proppush/state_u100.pt", map_location="cpu", weights_only=False)
ck["args"]["base"] = "$R/${TAG}_solo2/state_u300.pt"; ck["args"]["base_reach"] = 10.0; ck["args"]["base_hold"] = 10
torch.save(ck, "$R/${TAG}_proppushmix/state_u100.pt")
print("혼합 저장", "$R/${TAG}_proppushmix/state_u100.pt")
PY
echo done > $D/${TAG}_proppush.done
