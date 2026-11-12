#!/bin/bash
# 다가가는 속도가 지렛대인가 (11-13 08:20). 앞 실험(nudge_replay.sh): 상자에 세게 닿는 것은 왼쪽·오른쪽을 못 정해서가 아니라, 늘 달리는 줄 위의
# 늦게 보이는 상자에 마지막 1 s에 반응하는데 그 반응이 아슬아슬해서(그대로 복제도 1-3 s 전이면 48 %만 닿음).
# 물음: 그 1-3 s 전부터 1 s 동안만 plan 속도를 0.8배로 하면 닿는 비율이 얼마나 떨어지고, 그 대가로 거리를 얼마나 잃는가.
#   크게 떨어지면(48 % -> 15 % 아래) 다가가는 속도가 지렛대 -> 지금 보상(세게 닿으면 약 5 s 손해)이 그 감속을 살 만큼 비싼지가 다음 물음.
#   거의 안 떨어지면 속도가 아니라 상자를 늦게 반영하는 것 자체(보는 쪽)가 원인.
# plan25y u150, 시험 상자 은행 uniform, 64묶음 x (원본 + 그대로 2 + 느리게 2) x 90 s, 3 s마다 복제, 시드 3개. 4070(학습이 없을 때).
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
cd /home/shchon11/F1tenth_E2E/f1sim || exit 1
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
run=${1:-plan25y}; u=${2:-150}
for seed in 20261301 20261302 20261303; do
  out=$R/out/replay/slow_${run}_u${u}_s${seed}
  [ -e $out.npz ] && continue
  F1SIM_EXAM_SEED=$seed BOX_REPLAY_ARMS="2,0,0,0,0,2" BOX_REPLAY_SLOW="0.8,10" CUDA_VISIBLE_DEVICES=0 \
    timeout 3600 python3 -u -W ignore $R/tools/box_replay.py /home/shchon11/f1sim_runs/$run/plan_u$u.pt $out.npz 64 90 3.0 uniform > $out.log 2>&1
  echo "rc $?" >> $out.log
done
python3 -W ignore $R/tools/box_replay_table.py $R/out/replay/slow_${run}_u${u} $R/out/replay/slow_${run}_u${u}_s*.npz > $R/out/replay/slow_${run}_u${u}.txt 2>&1
echo done > $R/out/replay/slow_${run}_u${u}.done
