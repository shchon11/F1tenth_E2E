#!/bin/bash
# 판정용 시험 기록: checkpoint RUN/plan_uU가 나오면 exam1(ideal, 상자 uniform, 256세션 x 90 s) 기록 -> 점수표(세게 닿음, 서 버림·다시 달리기,
# 깨끗한 세션) + 상자 앞 쪽 정하기(side_flip). 인자: RUN U
RUN=$1; U=$2
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools
C=/home/shchon11/f1sim_runs/$RUN/plan_u$U.pt
O=$R/out/exam/exam1_${RUN}_u${U}_ideal_uniform_n256
until [ -e $C ] || [ -e $R/out/one_$RUN.done ]; do sleep 30; done
[ -e $C ] || exit 1
sleep 20
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
if [ ! -e $O.npz ]; then
  CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 5400 python3 -u -W ignore $T/exam_record.py $C $O.npz ideal 256 uniform > $O.log 2>&1
  echo "rc $?" >> $O.log
fi
{ python3 -W ignore $T/exam_table.py $O.npz; python3 -W ignore $T/side_flip.py $O.npz; } > $R/out/analysis/${RUN}_u${U}_exam.txt 2>&1
echo done > $R/out/analysis/${RUN}_u${U}_exam.done
