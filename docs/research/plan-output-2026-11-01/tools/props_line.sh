#!/bin/bash
# 라인 위 상자를 넣은 학습 세계 (11-03): 일곱 선생이 모두 같은 상자(배치 5 #1 s 37.9, 배치 0 #14 s 39.4, 상자 하나 자리 5, 머리핀 입구
# 배치 11·20)에 박음. 정보는 입력에 있고(통로 칸 58/59), 반응이 작음(상자 입력을 지워도 kappa_1 차 0.01~0.05). 원인: 학습 세계의 상자
# 생성기는 패턴을 한쪽 벽에서 재는 띠에 놓아 넓은 구간에서는 가운데쯤 라인에 못 닿음 -- 라인 막는 상자가 랩 2 % 칸마다 200~390개인데
# s 36.4~41.6 m는 21~42개(10배 드묾), 머리핀 입구 84~106개 (`tools/prop_s_hist.py`). 정책은 거기서 상자 피하기를 배운 적이 없음.
# 고침: `--prop-line 0.3` -- 패턴 자리의 30 %를 레이싱 라인 위(+-0.3 m) 상자 하나로(한쪽 틈 1.2 m 보장). 확인: 그 구간 172~238개로 고르게.
# 함께: 랩 위치를 행위자에서 뺌(ae4f860, 자동), --base 혼합, 상관 0.9 + 하한. 512(w u100에서) / 256(g u100에서) 두 갈래 = 용량 질문.
# 기각(미리 정함): 원인 -- 콘솔 30배치에서 배치 0·5의 s 36~42 상자 접촉, 또는 배치 5·11·20 중 하나라도 1 s 넘게 멈춤.
# 전체 -- 상자 하나 > 2/12, 30배치 랩 중앙 > 7.3 s, 접촉 난 배치 > 5/30, 세게 > 15, 멈춘 배치 하나라도. 용량 -- 512가 256보다 랩 중앙·세게 둘 다 낫지 않으면 기각.
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
