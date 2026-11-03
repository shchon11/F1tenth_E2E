#!/bin/bash
# 상자 단계 두 갈래 (11-03): 배치 자리 학습(srp25h3g, --base + 상관 0.9)은 상자 하나 1/12로 비켜 가기를 배웠지만 상자 많은 30배치에서
# 랩 중앙 7.25 -> 8.21 s, 세게 38, 멈춘 배치 5곳 -> 기각. 상자 하나에서도 최소 거리 0.56~1.18 m(선 기준 정책 0.34~0.81) -- 여유를 크게 둠.
# 가설 (가) 잡음이 여유를 정함: PPO는 잡음 낀 정책을 최적화하므로 상관 잡음의 가로 흔들림(1 s에 p90 ~0.4 m)만큼 상자·벽에서 떨어짐 ->
#   g에서 --base 그대로 상관 0(독립) 100회면 여유가 줄고 빨라지며, --base 덕에 빈 길 기울기가 없어 비켜 가기는 남아야 함 (이름 g2).
#   (--base 없이 한 같은 시도 srp25h3e는 비켜 가기를 지웠음: 상자 정책이 빈 길까지 몰았음)
# 가설 (나) 모델이 작음(사용자): 2 x 256 하나에 혼자 선과 상자 다루기를 함께 담지 못해 이어 학습마다 빠르기를 내줌 -> g와 같은 학습을
#   은닉 512로(srp25h3 u200을 함수 그대로 넓혀 시작, 같은 시드·환경 수) -- 같은 비켜 가기에 더 빠르거나 덜 부딪혀야 함 (이름 w).
# 기각(미리 정함, 둘 다): 상자 하나 접촉 > 2/12, 30배치 랩 중앙 > 7.3 s, 접촉 난 배치 > 5/30, 세게 > 15, 1 s 넘게 멈춘 배치가 하나라도.
# (나)는 추가로 g u100보다 30배치 랩 중앙·세게가 둘 다 낫지 않으면 용량 가설 기각.
# 인자: 이름 시작체크포인트 GPU 환경수 시드 지킴이(1/0) 추가인자...
NAME=$1; INIT=$2; GPU=$3; ENVS=$4; S=$5; GUARD=$6; shift 6; EXTRA="$*"
M=real:korea_2025_iccas; MC=real/iccas25
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --severity-min 1.0"
SOLO=$(python3 -c "import torch; print(torch.load('$R/srp25h3_propsmix/state_u200.pt', map_location='cpu', weights_only=False)['args']['base'])")
if [ "$GUARD" = 1 ]; then
  bash $T/gpu_guard3.sh "state_[r]l --name ${NAME}|one_[c]rate.py.*${NAME}|console_[l]a.*${NAME}" $D/cap_${NAME}.done > $D/gpu_guard_${NAME}.log 2>&1 &
fi
L=$D/cap_${NAME}.log
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage props $PL $COMMON --updates 100 \
  --base $SOLO --base-reach 10 --base-hold 10 --seed $S --init $INIT $EXTRA > $D/${NAME}.log 2>&1 || { echo fail > $D/cap_${NAME}.done; exit 1; }
echo "== ${NAME} u100 (시작 $(basename $(dirname $INIT))/$(basename $INIT), 추가: $EXTRA)" > $L
CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 -u $T/one_crate.py $MC 30x34 lineref=$R/srp25h2_propsmix/state_u200.pt \
  ${NAME}_u100=$R/${NAME}/state_u100.pt 2>&1 | grep -A 12 "== ${NAME}" >> $L
echo "  선생(혼합) 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/${NAME}/state_u100.pt $MC 60 2>&1 | grep '주행' | cut -c1-160)" >> $L
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $R/${NAME}/state_u100.pt $MC 40 30 777 $D/layouts/t_${NAME} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_t_${NAME}.log
echo "  선생(혼합) 30배치: $(tail -n 1 $D/layouts_t_${NAME}.log)" >> $L
echo done > $D/cap_${NAME}.done
