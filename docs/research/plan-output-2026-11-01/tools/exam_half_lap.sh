#!/bin/bash
# 대회 방식 1대1 시험 기록 (11-12 23:05): 사용자가 나간 대회는 1대1을 반 바퀴 떨어진 반대쪽에서 선 채로 출발(나란히 출발 없음).
# 질문: 그 방식(반 바퀴 떨어져 선 채로 출발, 상자 2개 고정 뒤 50 s에 치움, 90 s)에서 상대 차와 몇 번 만나고, 몇 번 앞지르고,
#       얼마나 닿는가 -> plan25y u300을 견줄 기준 줄(plan25x u300).
# 견줌: 같은 policy의 exam1 기록(출발 배치가 앞·뒤 2.5~6 m / 나란히를 1/3씩)과 차 접촉을 견줌.
# 반증: 반 바퀴 출발에서 세션당 상대 차에 세게 닿는 수가 exam1과 같으면, 시험의 1대1 줄은 출발 배치와 상관없음(두 기록이 같은 것을 잼).
#       훨씬 적으면 exam1의 차 접촉은 대회에 없는 가까운 출발에서 온 몫이 크다는 뜻 -> 1대1 줄은 반 바퀴 출발로 바꿈.
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
cd $R/tools || exit 1
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=1
run=${1:-plan25x}; u=${2:-300}
out=$R/out/exam/exam1_${run}_u${u}_ideal_max2_n256_half
[ -e $out.npz ] && exit 0
F1SIM_EXAM_HALF_LAP=1 F1SIM_EXAM_START_REST=1 F1SIM_EXAM_OPP_SLOT='{"spawn": "ahead"}' nice -n 5 bash slot5060_all.sh timeout 2400 \
  python3 -u -W ignore exam_record.py /home/shchon11/f1sim_runs/$run/plan_u$u.pt $out.npz ideal 256 max2 90 50 > $out.log 2>&1
echo "rc $?" >> $out.log
touch $out.done
