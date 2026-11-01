#!/bin/bash
# 상자 접촉 시간 대가 (11-02 06:00): 상자 단계 선생이 라인 위 큰 상자 앞에서 1.5 m/s 아래로 감속해 밀고 지나감(콘솔 30배치 중
# 2~3배치, 4~11 s 멈춤). 1.5 m/s 미만 접촉은 학습에서 공짜였음(contact_severity_min) -> --prop-touch-price 40(초당, 속도 무관).
# 상자 200회(u200)에서 이어 100회, 바꾼 것은 이 대가와 고정 가격(이어 학습은 업데이트를 1부터 세므로 상승 구간 없이 80) 뿐.
# 인자: 맵 이름머리 GPU 시드
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
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_propst --stage props $PL $COMMON --updates 100 \
  --lr 1e-4 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --prop-touch-price 40 --seed ${S}4 \
  --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_propst.log 2>&1 || exit 1
# 콘솔·학생용 혼합(빈 길은 혼자 선생): 같은 모델에 base만 붙임 (*_propsmix와 같은 방식)
mkdir -p $R/${TAG}_propstmix
python3 - <<PY
import torch
ck = torch.load("$R/${TAG}_propst/state_u100.pt", map_location="cpu", weights_only=False)
ck["args"]["base"] = "$R/${TAG}_solo2/state_u300.pt"; ck["args"]["base_reach"] = 10.0; ck["args"]["base_hold"] = 10
torch.save(ck, "$R/${TAG}_propstmix/state_u100.pt")
print("혼합 저장", "$R/${TAG}_propstmix/state_u100.pt")
PY
echo done > $D/${TAG}_propst.done
