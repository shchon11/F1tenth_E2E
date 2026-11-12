#!/bin/bash
# 상자에 세게 닿는 원인 확인 (11-13, 사용자: "플랜이 와리가리 하다가 중간으로 탕 가서 ... 무슨 똥딴지 같은 소리야?").
# 물음: 경로를 하나만 내는 policy(plan25y u150)가 상자에 세게 닿을 때, (가) 왼쪽·오른쪽을 못 정해 가운데로 가는 것인가,
#       (나) 상자를 늦게 알아채는 것인가.
# 방법: 달리는 차를 3 s마다 복사(box_replay.py): 그대로 2대 / 복사 직후 0.3 s만 조향 +2도(왼쪽) 2대 / -2도(오른쪽) 2대, 그 뒤는 policy 그대로.
# 판정: 원본이 세게 닿은 통과에서, 남은 시간 0.5-1.5 s(약 2-6 m 앞)에 틀어 준 복사본이
#   (가) 양쪽 다 대부분 안 닿고 틀어 준 쪽으로 계속 감 -> 쪽을 못 정한 것이 원인
#   (나) 틀어 줘도 원래 길로 돌아와 그대로 닿음 -> 상자를 늦게 알아채는 것이 원인
#   한쪽만 피함 -> 피할 쪽이 정해진 자리(한쪽이 막힘)이고, policy가 그쪽을 늦게 고른 것.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
cd /home/shchon11/F1tenth_E2E/f1sim || exit 1
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
run=${1:-plan25y}; u=${2:-150}
for seed in 20261301 20261302 20261303; do
  out=$R/out/replay/nudge_${run}_u${u}_s${seed}
  [ -e $out.npz ] && continue
  F1SIM_EXAM_SEED=$seed BOX_REPLAY_ARMS="2,0,0,2,2" BOX_REPLAY_NUDGE="2.0,3" CUDA_VISIBLE_DEVICES=1 bash $R/tools/slot5060.sh \
    timeout 3600 python3 -u -W ignore $R/tools/box_replay.py /home/shchon11/f1sim_runs/$run/plan_u$u.pt $out.npz 64 90 3.0 uniform > $out.log 2>&1
  echo "rc $?" >> $out.log
done
python3 -W ignore $R/tools/box_replay_table.py $R/out/replay/nudge_${run}_u${u} $R/out/replay/nudge_${run}_u${u}_s*.npz > $R/out/replay/nudge_${run}_u${u}.txt 2>&1
echo done > $R/out/replay/nudge_${run}_u${u}.done
