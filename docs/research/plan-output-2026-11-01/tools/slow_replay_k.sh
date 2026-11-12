#!/bin/bash
# slow_replay.sh의 배율을 바꾼 판 (11-13 08:40): 속도 배율 k로 1 s 동안 -> 닿는 비율과 잃는 거리가 k에 따라 어떻게 바뀌나(0.8과 함께 두 점).
# 인자: run u k GPU(0=4070, 1=5060)
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
cd /home/shchon11/F1tenth_E2E/f1sim || exit 1
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
run=$1; u=$2; k=$3; gpu=$4; tag=slow${k/./}
for seed in 20261301 20261302 20261303; do
  out=$R/out/replay/${tag}_${run}_u${u}_s${seed}
  [ -e $out.npz ] && continue
  if [ "$gpu" = 1 ]; then W="bash $R/tools/slot5060_all.sh"; else W=""; fi
  F1SIM_EXAM_SEED=$seed BOX_REPLAY_ARMS="2,0,0,0,0,2" BOX_REPLAY_SLOW="$k,10" CUDA_VISIBLE_DEVICES=$gpu \
    $W timeout 3600 python3 -u -W ignore $R/tools/box_replay.py /home/shchon11/f1sim_runs/$run/plan_u$u.pt $out.npz 64 90 3.0 uniform > $out.log 2>&1
  echo "rc $?" >> $out.log
done
python3 -W ignore $R/tools/box_replay_table.py $R/out/replay/${tag}_${run}_u${u} $R/out/replay/${tag}_${run}_u${u}_s*.npz > $R/out/replay/${tag}_${run}_u${u}.txt 2>&1
echo done > $R/out/replay/${tag}_${run}_u${u}.done
