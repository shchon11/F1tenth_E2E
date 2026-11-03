#!/bin/bash
# 경로 생성기로 판정한 균일 상자 배치 은행 세계에서 상자 단계 다시 학습 (11-03 밤). 사용자: "전혀 고르게 나오지 않잖아 ... 고르게 뽑히라고
# uniform하게. 다만 지나갈 차 폭 고려해서 아예 못지나가게만 배치하지 말고" + "그 주행가능한지는 path generator 로직 사용해서 판단하는거지?"
# 빈도 지도(prop_freq.py): 지금까지 학습 상자는 벽 0.3 m 안에 42 %(2025) / 71 %(2026), 라인 막음 25 % / 26 %(경주 단계 11 % / 8 %).
# 은행(layout_bank.py): 빈 칸에서 균일하게 뽑고 LayoutLines가 차 몸체 0.10 m 틈 + 조향 한계 곡률로 라인을 찾은 배치만, 리셋마다 1..전부.
# 시작 = 그 지도의 gwt 선생, 조리법 그대로(독립 잡음, 닿기 값 40, lr 5e-5, --base), 바뀌는 것은 상자 세계뿐.
# 기준 먼저: 시작 선생을 새 콘솔 30배치(은행, 시드 777)에서. 기각(미리 정함): 새 선생 u100이 같은 30배치에서
#   세게(벽+상자) > 시작 선생의 0.6배, 또는 1 s 넘게 멈춘 배치 수 > 시작 선생, 또는 혼자 랩 중앙 > 시작 선생 + 0.15 s.
# 인자: 지도(25|26) 이름 시작체크포인트 GPU 환경수 시드 지킴이(1/0) [업데이트]
MAPK=$1; NAME=$2; INIT=$3; GPU=$4; ENVS=$5; S=$6; GUARD=$7; U=${8:-100}
if [ "$MAPK" = 25 ]; then M=real:korea_2025_iccas; MC=real/iccas25; SOLO=/home/shchon11/f1sim_runs/srp25h2_solo2/state_u300.pt
else M=real:korea_2026_competition; MC=real/korea26; SOLO=/home/shchon11/f1sim_runs/srp26h2_solo2/state_u300.pt; fi
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
BANK=$R/prop_banks
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shchon11/F1tenth_E2E/f1sim
PL="--action plan --plan-lookahead 0.235 --plan-horizon 0.6 --repeat 2 --tracks $M --plan-relabel 1.0 --plan-relabel-speed 0"
COMMON="--envs $ENVS --warps 16 --beams 108 --episode-s 40 --horizon 64 --epochs 4 --minibatches 8 --gamma 0.995 --lam 0.95 --clip 0.2 --save-every 50 \
  --soft-bound 1 --bound-loss 1.0 --hit-head 1.0 --hit-shaping 1.0 --level-replay 0.5 \
  --corridor-obs 1 --scan-obs 1 --prop-fraction 0.8 --removal-p 0.5 --removal-s 10 30 --prop-count uniform --vis-actor 1 \
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --severity-min 1.0 \
  --noise-corr 0 --prop-touch-price 40 --prop-bank $BANK"
L=$D/bank_${NAME}.log
IN=$(basename $(dirname $INIT))
echo "== ${NAME}: 은행 세계 상자 단계 (시작 ${IN}/$(basename $INIT))" > $L
CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $INIT $MC 40 30 777 $D/layouts/b_${IN} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_b_${IN}.log
echo "  시작 선생 30배치(은행): $(tail -n 1 $D/layouts_b_${IN}.log)" >> $L
echo "  시작 선생 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $INIT $MC 60 2>&1 | grep '주행' | cut -c1-160)" >> $L
if [ "$GUARD" = 1 ]; then
  bash $T/gpu_guard3.sh "state_[r]l --name ${NAME}|console_[l]a.*${NAME}" $D/bank_${NAME}.done > $D/gpu_guard_${NAME}.log 2>&1 &
fi
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage props $PL $COMMON --updates $U \
  --base $SOLO --base-reach 10 --base-hold 10 --seed $S --init $INIT > $D/${NAME}.log 2>&1 || { echo fail > $D/bank_${NAME}.done; exit 1; }
for UU in 50 $U; do
  C=$R/${NAME}/state_u${UU}.pt
  CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $C $MC 40 30 777 $D/layouts/b_${NAME}_u${UU} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_b_${NAME}_u${UU}.log
  echo "  u${UU} 30배치(은행): $(tail -n 1 $D/layouts_b_${NAME}_u${UU}.log)" >> $L
  echo "  u${UU} 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $C $MC 60 2>&1 | grep '주행' | cut -c1-160)" >> $L
done
echo done > $D/bank_${NAME}.done
