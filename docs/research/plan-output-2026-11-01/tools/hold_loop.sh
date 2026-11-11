#!/bin/bash
# plan 하나를 0.1 / 0.2 / 0.3 / 0.5 s 그대로 달리는 측정을 checkpoint마다 (11-11, 사용자와 합의한 scratch 비교의 판정 숫자: 제어기 없이
# plan만으로 -- plan이 궤적으로 쓸 만한가). drive_record.py F1SIM_HOLD(학습의 붙잡기와 같은 입력 갱신), 학습 세계, 고정 시드, 학습 차
# 256대 x 40 s, 5060 자리(slot5060.sh). 표 out/hold/NAME_uUU.txt (hold_table.py), 0.1 s와 0.3 s 기록만 남김(장면 그림용, 하나 30 MB).
# 인자: 이름 시작 간격 끝
NAME=$1; U0=$2; STEP=$3; U1=$4
D=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out
T=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
R=/home/shchon11/f1sim_runs
H=$D/hold; mkdir -p $H $D/rec
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform
cd /home/shchon11/F1tenth_E2E/f1sim
for UU in $(seq $U0 $STEP $U1); do
  C=$R/${NAME}/plan_u${UU}.pt
  [ -e $H/${NAME}_u${UU}.done ] && continue
  until [ -e $C ] || [ -e $D/one_${NAME}.done ]; do sleep 30; done
  [ -e $C ] || break
  sleep 10
  for K in 1 3 2 5; do
    F1SIM_HOLD=$K CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 3000 python3 $T/drive_record.py $C $D/rec/hold${K}_${NAME}_u${UU}.npz 512 40 > $H/${NAME}_u${UU}_h$K.log 2>&1
  done
  (cd $D && python3 $T/hold_table.py 0.1s=rec/hold1_${NAME}_u${UU}.npz 0.2s=rec/hold2_${NAME}_u${UU}.npz 0.3s=rec/hold3_${NAME}_u${UU}.npz 0.5s=rec/hold5_${NAME}_u${UU}.npz > $H/${NAME}_u${UU}.txt 2>&1)
  python3 -c "import os, sys; [os.remove(f) for f in sys.argv[1:] if os.path.exists(f)]" $D/rec/hold2_${NAME}_u${UU}.npz $D/rec/hold5_${NAME}_u${UU}.npz
  echo done > $H/${NAME}_u${UU}.done
done
