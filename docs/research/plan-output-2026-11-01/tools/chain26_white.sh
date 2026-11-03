#!/bin/bash
# Korea 2026, 상관 잡음 단계 빼기 (11-03 18:05): chain26_line.sh 단계별 30배치 -- gl(상관 0.9 + 라인 상자) 8.92 s 벽 세게 5,
# gw(독립) 8.76 s 상자 접촉 0 벽 세게 5, gwt(+밀기 값) 9.10 s 벽 7 멈춤 1. 벽 세게는 모두 배치 10·13의 같은 자리(오른쪽 직선 튀어나온 벽,
# 6.8 m/s) -- 시작점 srp26h3(벽 1)에는 없고 상관 잡음 단계(gl)에서 생김. 가설: 상관 잡음이 선을 벽 쪽으로 밀고 느리게 함 ->
# srp26h3 u200에서 바로 독립 잡음 + 라인 상자 100회(srp26w).
# 기각(미리 정함): 30배치 벽 세게 > 1, 상자 세게 > 4, 랩 중앙 > 8.00 s, 멈춤(1 s 넘게) 하나라도, 상자 하나 > 2/12, 혼자 > 7.5 s.
M=real:korea_2026_competition; MC=real/korea26; GPU=$1; ENVS=$2; S=$3
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
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --severity-min 1.0 --prop-line 0.3"
SOLO=$(python3 -c "import torch; print(torch.load('$R/srp26h3_propsmix/state_u200.pt', map_location='cpu', weights_only=False)['args']['base'])")
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name srp26w_props --stage props $PL $COMMON --updates 100 \
  --base $SOLO --base-reach 10 --base-hold 10 --seed ${S}4 --init $R/srp26h3_props/state_u200.pt --noise-corr 0 > $D/srp26w_props.log 2>&1 || { echo fail > $D/chain26_white.done; exit 1; }
L=$D/chain26_white.log
echo "== srp26w u100 (srp26h3 u200 -> 라인 상자 + 독립 잡음)" > $L
CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 -u $T/one_crate.py $MC 30x34 lineref=$R/srp26h3_propsmix/state_u200.pt \
  srp26w_u100=$R/srp26w_props/state_u100.pt 2>&1 | grep -A 12 "== srp26w" >> $L
echo "  선생(혼합) 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/srp26w_props/state_u100.pt $MC 60 2>&1 | grep '주행' | cut -c1-160)" >> $L
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $R/srp26w_props/state_u100.pt $MC 40 30 777 $D/layouts/t_srp26w 2>&1 | grep "배치\|합계\|Error" > $D/layouts_t_srp26w.log
echo "  선생(혼합) 30배치: $(tail -n 1 $D/layouts_t_srp26w.log)" >> $L
echo done > $D/chain26_white.done
