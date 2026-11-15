#!/bin/bash
# (11-16) 앞이 막혀 선 차를 복제해 '그대로' vs '조향 0으로 이어서 후진'(box_replay.py 갈래 V) -> stall_reverse_table.py
# 인자: checkpoint 출력이름 [후진 "속도,결정수" 0.8,8] [묶음 48] [초 150] [주기 4]
CK=$1; N=$2; REV=${3:-0.8,8}; G=${4:-48}; SECS=${5:-150}; PER=${6:-4}
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim F1SIM_RACELINE_CACHE_ONLY=1
cd $R/tools
BOX_REPLAY_ARMS="2,0,0,0,0,0,2" BOX_REPLAY_REV=$REV CUDA_VISIBLE_DEVICES=1 bash slot5060.sh timeout 3600 python3 -u box_replay.py $CK \
  $R/out/stallrev/$N.npz $G $SECS $PER uniform > $R/out/stallrev/$N.log 2>&1
echo "rc $?" >> $R/out/stallrev/$N.log
touch $R/out/stallrev/$N.done
