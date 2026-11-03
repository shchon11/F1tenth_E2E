#!/bin/bash
# 은행 세계 상자 단계 + 추가 인자 (11-04 02:05, props_bank_more.sh에 추가인자 10번째부터). 상관 잡음 단계: 예전 세계에서 피하기를 가르친 건 gl(--noise-corr 0.9) 단계였고 은행 재학습은 그걸 건너뜀.
# 100 업데이트는 세계가 조금 바뀔 때 쓰던 미세 조정 길이. 학습 세계 곡선은 level replay(어려운 배치 다시 뽑기) 때문에 비교가 안 되므로
# 고정 30배치(은행, 시드 777)로 u50/u100/연장 u100/u200을 비교: 연장이 u100보다 세게(벽+상자) 0.8배 미만이면 학습 부족이 맞음.
# 조리법은 props_bank.sh 그대로, 시작만 그 u100. 인자: 지도(25|26) 이름 시작체크포인트 GPU 환경수 시드 지킴이(1/0) 업데이트 [기다릴파일]
MAPK=$1; NAME=$2; INIT=$3; GPU=$4; ENVS=$5; S=$6; GUARD=$7; U=$8; WAIT=$9; EXTRA="${@:10}"
if [ -n "$WAIT" ]; then until [ -e "$WAIT" ]; do sleep 30; done; fi
if [ "$MAPK" = 25 ]; then M=real:korea_2025_iccas; MC=real/iccas25; SOLO=/home/shchon11/f1sim_runs/srp25h2_solo2/state_u300.pt
else M=real:korea_2026_competition; MC=real/korea26; SOLO=/home/shchon11/f1sim_runs/srp26h2_solo2/state_u300.pt; fi
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
  --lr 5e-5 --lr-end 5e-5 --price 80 --price-start 80 --price-hold 0 --price-ramp 1 --severity-min 1.0 \
  --noise-corr 0 --prop-touch-price 40 --prop-bank $R/prop_banks"
L=$D/bank_${NAME}.log
echo "== ${NAME}: 은행 세계 상자 단계 연장 (시작 $(basename $(dirname $INIT))/$(basename $INIT), ${U} 업데이트)" > $L
if [ "$GUARD" = 1 ]; then
  bash $T/gpu_guard3.sh "state_[r]l --name ${NAME}|console_[l]a.*${NAME}" $D/bank_${NAME}.done > $D/gpu_guard_${NAME}.log 2>&1 &
fi
CUDA_VISIBLE_DEVICES=$GPU python3 -u -m f1sim.learn.state_rl --name ${NAME} --stage props $PL $COMMON --updates $U \
  --base $SOLO --base-reach 10 --base-hold 10 --seed $S --init $INIT $EXTRA > $D/${NAME}.log 2>&1 || { echo fail > $D/bank_${NAME}.done; exit 1; }
for UU in $((U / 2)) $U; do
  C=$R/${NAME}/state_u${UU}.pt
  CUDA_VISIBLE_DEVICES=$GPU timeout 3600 python3 $T/console_layouts.py $C $MC 40 30 777 $D/layouts/b_${NAME}_u${UU} 2>&1 | grep "배치\|합계\|Error" > $D/layouts_b_${NAME}_u${UU}.log
  echo "  u${UU} 30배치(은행): $(tail -n 1 $D/layouts_b_${NAME}_u${UU}.log)" >> $L
  echo "  u${UU} 혼자: $(CUDA_VISIBLE_DEVICES=$GPU timeout 900 python3 $T/console_laps.py $C $MC 60 2>&1 | grep '주행' | cut -c1-160)" >> $L
done
echo done > $D/bank_${NAME}.done
