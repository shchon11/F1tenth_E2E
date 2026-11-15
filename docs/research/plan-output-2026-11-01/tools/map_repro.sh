#!/bin/bash
# (11-15 20:50, 사용자: "map 잘 만든거 맞아..? 이렇게 구리진 않았는데") 예전 MAP 시험(11-14 17:00)을 지금 코드로 같은 시드에 다시: 같은 숫자면 MAP 정상
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam; O=$R/out/maprepro; mkdir -p $O
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
for r in plan25y plan26i; do
  N=$O/exam1_${r}_u150_map_uniform_n256_now.npz
  [ -e $N ] || CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 3600 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/$r/plan_u150.pt $N map 256 uniform > ${N%.npz}.log 2>&1
done
cd $T
for r in plan25y plan26i; do
  echo "== $r u150 MAP: 11-14 17:00 기록 기준, 지금 코드로 다시"
  python3 -W ignore grad_step_table.py $E/exam1_${r}_u150_map_uniform_n256.npz "지금 코드=$O/exam1_${r}_u150_map_uniform_n256_now.npz" 2>&1 | tail -4
done
