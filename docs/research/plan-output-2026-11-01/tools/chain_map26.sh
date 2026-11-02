#!/bin/bash
# Korea 2026 원본 크기로 (11-02 17:30, 사용자 "원본으로 혀~"): 대회 bag의 /map은 실제 크기(라이다 정합 축척 1.00, 다른 팀 IFAC 맵과
# 겹침 IoU 0.95)라 ×1.2 맵은 실제보다 20 % 큼. Korea 2025와 같은 정직한 궤적 연쇄를 원본 맵(real:korea_2026_competition)에서:
# 혼자 300회(×1.2 혼자 u300에서 -- 같은 모양, 크기만 다름) -> 상자 200회 -> 혼자 선생 혼합(propsmix) -> 라이다 학생(상자 단계).
# 4070, 지킴이 아래. 인자: GPU 시드머리
GPU=$1; S=$2
until [ -f /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/srp25h2_racev.done ] || ! pgrep -f "state_[r]l --name srp25h2_racev" > /dev/null; do sleep 30; done   # the 4070 is free
TAG=srp26h2; M=real:korea_2026_competition
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs 2048 --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --std-floor 0.05 0.2 --soft-bound 1 --bound-loss 1.0 --noise-corr 0.9 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30"
bash $T/gpu_guard3.sh "state_[r]l --name ${TAG}_|state_[s]tudent --name sst_${TAG}" $D/chain_${TAG}.done > $D/gpu_guard_${TAG}.log 2>&1 &
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_solo2 --stage solo $PL --envs 2048 --warps 16 --beams 36 --episode-s 40 \
  --updates 300 --horizon 64 --epochs 4 --minibatches 8 --lr 1e-4 --lr-end 5e-5 --gamma 0.995 --lam 0.95 --clip 0.2 \
  --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --save-every 50 --seed ${S}5 \
  --init $R/srp26xh2_solo2/state_u300.pt > $D/${TAG}_solo2.log 2>&1 || exit 1
echo solo > $D/chain_${TAG}.solo2_done
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_props --stage props $PL $COMMON --updates 200 \
  --lr 1e-4 --lr-end 5e-5 --price 80 --price-start 20 --price-hold 0 --price-ramp 100 --seed ${S}2 \
  --init $R/${TAG}_solo2/state_u300.pt > $D/${TAG}_props.log 2>&1 || exit 1
echo props > $D/chain_${TAG}.props_done
# the solo teacher drives where nothing is near (SoloMix), as the other picks
python3 - <<EOF
import torch, os, shutil
ck = torch.load("$R/${TAG}_props/state_u200.pt", map_location="cpu", weights_only=False)
a = dict(ck["args"]); a.update(base="$R/${TAG}_solo2/state_u300.pt", base_reach=10.0, base_hold=10); ck["args"] = a
os.makedirs("$R/${TAG}_propsmix", exist_ok=True)
torch.save(ck, "$R/${TAG}_propsmix/state_u200.pt")
shutil.copy("$R/${TAG}_props/args.json", "$R/${TAG}_propsmix/args.json")
EOF
bash $T/student_propsmix.sh $TAG $GPU ${S}9 || exit 1
echo done > $D/chain_${TAG}.done
