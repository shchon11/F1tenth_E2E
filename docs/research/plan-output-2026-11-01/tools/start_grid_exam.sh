#!/bin/bash
# 대회 규칙의 출발 자리에서 출발 직후 차끼리 닿는지 보는 기록 (11-12, 출발 구간).
# 근거: 규칙(RoboRacer 2025)은 1대1을 같은 출발선에 30 cm 떨어져 나란히 세움 = 학습·시험의 '나란히' 배치. plan을 그대로 실행하는 시험
#   기록(plan25r u300, 시드 3개)에서 나란히 출발 56번 가운데 14번(25 %)이 2 s 안에 닿고 6번은 세게 닿음. 다만 그 기록은 두 차의 출발
#   속도를 따로따로 0~3 m/s에서 뽑음(규칙은 둘 다 선 채로).
# 질문: 둘 다 선 채로 나란히 출발하면 출발 직후 접촉이 얼마나 되는가, 좁히는 쪽은 누구인가.
# 바꾸는 것: 배치를 모두 나란히(상대 차 종류·속도 배율은 학습 세계 그대로), 출발 속도 0 (F1SIM_EXAM_START_REST=1) / 견줄 기록은 0~3 m/s 그대로.
# 반증: (1) 선 채로 출발해도 2 s 안 접촉이 0~3 m/s 추첨 기록보다 줄지 않는다 -> 출발 속도 추첨은 원인이 아님(두 차가 한 줄로 모이는 것이 원인).
#       (2) 1 s 안에 닿은 것의 2/3 넘게 상대 차만 좁혔다 -> 우리 policy가 아니라 옆 차를 안 보는 상대 차 쪽 문제.
# 인자 없음. 세션 20 s(출발만 봄), 세션 128개, 상자 2개 은행(치우지 않음), plan을 그대로 실행.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
cd $R/tools || exit 1
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1
for run in plan25x plan25r; do
  for rest in 1 0; do
    out=$R/out/exam/grid1_${run}_u300_ideal_max2_n256_rest${rest}
    [ -e $out.npz ] && continue
    F1SIM_EXAM_START_REST=$rest F1SIM_EXAM_OPP_SLOT='{"spawn": "alongside"}' nice -n 5 bash slot5060_all.sh timeout 1500 \
      python3 -u -W ignore exam_record.py /home/shchon11/f1sim_runs/$run/plan_u300.pt $out.npz ideal 256 max2 20 50 > $out.log 2>&1
    echo "rc $?" >> $out.log
  done
done
nice -n 8 python3 -W ignore start_abreast.py $R/out/exam/grid1_*_rest*.npz > $R/out/start/grid1_table.txt 2>&1
echo done > $R/out/start/grid1.done
