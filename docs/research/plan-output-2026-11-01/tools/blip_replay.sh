#!/bin/bash
# 탐색 잡음이 실제로 해 보는 크기의 조향 변화가 상자 앞에서 값이 있는가 (11-13 밤, 구조 점검).
# 근거: nudge_value.py (plan25y u150, 복사 직후 조향 2도를 0.3 s 동안 상자 반대쪽으로): 상자 1.5-6 m 앞에서 세게 닿음 7-8 % -> 1.4-4 %,
#   진행 +0.05~+0.12 s, 지금 보상으로 +0.24~+0.36 s 이득. 그런데 policy의 첫 0.1 s는 2.5 m 안에서만 반응함.
#   학습의 탐색은 결정(0.1 s)마다 따로 뽑는 잡음이고 첫 0.1 s 조향의 표준편차는 0.26(u150) = 5.5 m/s에서 약 1.1도.
# 물음: 한 결정(0.1 s)만 2도 틀어 준 것(탐색 잡음 약 2배 크기 한 번)도 값이 있는가.
#   있음(0.3 s짜리의 1/3쯤, 2.5-6 m에서 +0.08~+0.12 s) -> 탐색이 해 보는 것에 신호가 있음: 학습이 그 신호를 못 따라가는 까닭을 봐야 함.
#   없음(0 근처, 닿는 비율도 그대로) -> 한 결정짜리 잡음은 다음 결정들이 지워 버림: 0.1 s마다 따로 뽑는 탐색으로는 일찍 비키는 것을 찾을 수 없음.
# 방법: box_replay.py, 3 s마다 복사, 그대로 2 / 왼쪽 2 / 오른쪽 2, 복사 직후 한 결정만 조향 +-2도. 인자: GPU번호(0 = 4070, 1 = 5060) 시드...
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
cd /home/shchon11/F1tenth_E2E/f1sim || exit 1
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
gpu=$1; shift
for seed in "$@"; do
  out=$R/out/replay/blip_plan25y_u150_s${seed}
  [ -e $out.npz ] && continue
  if [ $gpu = 1 ]; then
    F1SIM_EXAM_SEED=$seed BOX_REPLAY_ARMS="2,0,0,2,2" BOX_REPLAY_NUDGE="2.0,1" CUDA_VISIBLE_DEVICES=1 bash $R/tools/slot5060.sh \
      timeout 3600 python3 -u -W ignore $R/tools/box_replay.py /home/shchon11/f1sim_runs/plan25y/plan_u150.pt $out.npz 64 90 3.0 uniform > $out.log 2>&1
  else
    F1SIM_EXAM_SEED=$seed BOX_REPLAY_ARMS="2,0,0,2,2" BOX_REPLAY_NUDGE="2.0,1" CUDA_VISIBLE_DEVICES=0 \
      timeout 3600 python3 -u -W ignore $R/tools/box_replay.py /home/shchon11/f1sim_runs/plan25y/plan_u150.pt $out.npz 64 90 3.0 uniform > $out.log 2>&1
  fi
  echo "rc $?" >> $out.log
done
touch $R/out/replay/blip_plan25y_u150.gpu$gpu.$1.done
