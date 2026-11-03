#!/bin/bash
# gw(라인 상자 + 독립 잡음) 다음 (11-03 14:00): gw u100 -- 30배치 랩 중앙 8.14 s, 세게 9, 접촉 난 배치 4/30, 상자 하나 1/12, 그러나
# 배치 12(머리핀 출구 직선의 큰 상자)·19(머리핀 입구 작은 원통)에서 천천히 붙어 밀며 멈춤(세게 문턱 아래 접촉은 보상에서 공짜).
# (가) 밀기 값: 독립 잡음 + --prop-touch-price 40(상자에 닿은 시간마다 값). 11-02엔 멈춤 0이 됐으나 크게 피하다 벽 세게 10, 느려짐 --
#   그땐 학습 세계가 피하기를 못 가르쳤음(라인 막는 상자 구간별 10배 차). 가설: 이제는 미는 전략만 없어지고 속도는 그대로.
#   기각: 1 s 넘게 멈춘 배치 하나라도, 세게(벽 포함) > 15, 랩 중앙 > gw + 0.2 s(8.34).
# (나) 잡음 더 줄이기: 독립 잡음 + 크기 절반(--log-std-shift -0.693, 하한 0.025/0.1). 가설: 여유는 잡음이 정함(gl 9.02 -> gw 8.14).
#   기각: 랩 중앙 >= 7.9 s, 또는 상자 하나 > 2/12, 세게 > 15.
# 최종 통과선은 그대로: 상자 하나 <= 2/12, 랩 중앙 <= 7.3 s, 접촉 난 배치 <= 5/30, 세게 <= 15, 멈춘 배치 없음.
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
