#!/bin/bash
# plan26a(닿는 값 2배)가 상자에 박은 뒤 빠져나오지 못하고 붙어 있는가 (11-13 12:00): 학습 중 상자 닿음이 plan25y의 2배(2-7 /분 vs 1-2),
# u50 영상에서 상자에 코를 박고 10 s 넘게 섬. 같은 시험(exam1 ideal, 상자 uniform, 256세션 x 90 s)으로 plan26a u50과 plan25y u50을 기록해
# exam_table의 '세게 닿은 뒤 5 s 안에 서 버린 비율, 선 뒤 다시 2 m/s까지 걸린 시간'을 견줌.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
for run in plan26a plan25y; do
  O=$R/out/exam/exam1_${run}_u50_ideal_uniform_n256
  [ -e $O.npz ] && continue
  CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 5400 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/$run/plan_u50.pt $O.npz ideal 256 uniform > $O.log 2>&1
  echo "rc $?" >> $O.log
done
python3 -W ignore $T/exam_table.py $R/out/exam/exam1_plan26a_u50_ideal_uniform_n256.npz $R/out/exam/exam1_plan25y_u50_ideal_uniform_n256.npz > $R/out/analysis/exam_pair_u50.txt 2>&1
echo done > $R/out/analysis/exam_pair_u50.done
