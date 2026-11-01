#!/bin/bash
# 상대 단계 + 학생 (11-02 02:05): 상자 단계가 혼자 랩을 늦춤(Korea 2025 6.5 -> 7.2 s, ×1.2 8.6 -> 9.0 s) -> 빈 길은 혼자 선생이 운전
# (--base: 상자·차가 10 m 안에 없으면 혼자 단계 정책의 궤적, srl58에서 검증된 혼합; 궤적 출력용으로 11-02 확장), 경주 정책은 가까울 때만.
# 상대 300회(상자 200회에서) -> 라이다 격자 학생 80회(혼합 선생을 따라 배움). 같은 GPU, 학생은 한 번에 하나.
# 인자: 맵  이름 머리  GPU  시드 머리  환경 수
M=$1; TAG=$2; GPU=$3; S=$4; ENVS=$5
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --std-floor 0.05 0.2 --soft-bound 1 --bound-loss 1.0 --noise-corr 0.9 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30"
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${TAG}_raceb --stage race $PL $COMMON --updates 300 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --seed ${S}3 \
  --opp-present 0.5 --car-channel 1 --car-st 1 --scan-nocars 1 --opp-slot '{"speed_scale": [0.4, 1.05]}' \
  --base $R/${TAG}_solo2/state_u300.pt --init $R/${TAG}_props/state_u200.pt > $D/${TAG}_raceb.log 2>&1 || exit 1
echo race > $D/chain_${TAG}b.race_done
flock $D/student.lock env CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_student --name sst_${TAG}b --teacher $R/${TAG}_raceb/state_u300.pt \
  --stage race --envs 768 --iters 80 --steps 256 --buffer 400000 --buffer-cpu 1 --epochs 3 --frames 4 --stride 4 --danger-weight 4 \
  --opp-present 0.5 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30 --unwedge 0 --grid 1 --seed ${S}9 > $D/sst_${TAG}b.log 2>&1 || exit 1
echo done > $D/chain_${TAG}b.done
