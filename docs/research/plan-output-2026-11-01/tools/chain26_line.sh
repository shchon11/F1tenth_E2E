#!/bin/bash
# Korea 2026에 Korea 2025에서 통한 상자 처방 (11-03 16:20): 2026 학습 세계는 라인 막는 상자가 상자의 8 %뿐, 랩 절반(s 2.8~4.7, 11~21,
# 41~46 m)은 2 % 칸마다 0~10개(`out/prop_s_hist_26.log`). 라인 위 상자(좁은 곳은 틈 나는 쪽으로 밀어 놓기)로 대부분 칸 100~340개
# (`out/prop_s_hist_26_line03b.log`). 2025와 같은 순서: (1) 라인 상자 + 상관 0.9·하한 (2) 독립 잡음 (3) 독립 잡음 + 상자 미는 시간 값 40,
# 각 100회, --base(혼자 정책), 랩 위치 없음. 시작 srp26h3 u200.
# 기각(미리 정함, 최종 체크포인트, 시작점 srp26h3 30배치: 랩 중앙 8.00 s, 접촉 난 배치 6/30, 상자 10(세게 4), 벽 1): 멈춤(1 s 넘게) 하나라도,
# 세게(벽 포함) > 5, 접촉 난 배치 > 6/30, 랩 중앙 > 8.00 s, 상자 하나 > 2/12, 혼자 랩이 7.5 s보다 느림.
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
run() {  # 이름 시작 시드 추가인자...
  N=$1; I=$2; SD=$3; shift 3
  CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name $N --stage props $PL $COMMON --updates 100 \
    --base $SOLO --base-reach 10 --base-hold 10 --seed $SD --init $I "$@" > $D/$N.log 2>&1 || { echo fail $N > $D/chain26_line.done; exit 1; }
}
run srp26gl_props $R/srp26h3_props/state_u200.pt ${S}1 --noise-corr 0.9 --std-floor 0.05 0.2
run srp26gw_props $R/srp26gl_props/state_u100.pt ${S}2 --noise-corr 0
run srp26gwt_props $R/srp26gw_props/state_u100.pt ${S}3 --noise-corr 0 --prop-touch-price 40
L=$D/chain26_line.log
echo "== srp26gwt u100 (srp26h3 u200 -> 라인 상자 상관 -> 독립 -> 밀기 값)" > $L
CUDA_VISIBLE_DEVICES=$GPU timeout 2400 python3 -u $T/one_crate.py $MC 30x34 lineref=$R/srp26h3_propsmix/state_u200.pt \
  srp26gwt_u100=$R/srp26gwt_props/state_u100.pt 2>&1 | grep -A 12 "== srp26gwt" >> $L
echo "  선생(혼합) 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $R/srp26gwt_props/state_u100.pt $MC 60 2>&1 | grep '주행' | cut -c1-160)" >> $L
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $R/srp26gwt_props/state_u100.pt $MC 40 30 777 $D/layouts/t_srp26gwt 2>&1 | grep "배치\|합계\|Error" > $D/layouts_t_srp26gwt.log
echo "  선생(혼합) 30배치: $(tail -n 1 $D/layouts_t_srp26gwt.log)" >> $L
echo done > $D/chain26_line.done
