#!/bin/bash
# 포팅본 혼자 주행 한 번 (5060, 두 자리 다 잡음; 지금 학습의 평가·분석·영상이 돌거나 기다리면 들어가지 않음)
# 인자: 이름  세계(A=빈 트랙, B=상자 2개)  포팅본 설정 JSON  MAP 덮어쓰기 JSON  환경 수  초  [바닥 점: gate|oracle|off]
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
cd /home/shchon11/F1tenth_E2E/f1sim; source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
O=$R/out/aurorus; L=$R/out/.slot5060
A='{"warps": 0, "prop_fraction": 0.0}'
B='{"warps": 0, "prop_fraction": 1.0, "prop_bank": "/home/shchon11/f1sim_runs/prop_banks_exam1", "prop_count": "max2", "removal_p": 0.0}'
W="$A"; [ "$2" = "B" ] && W="$B"
FL=${7:-gate}
while [ "$(ps -eo cmd | grep -c '[s]lot5060.sh timeout')" -gt 0 ] || [ "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)" -lt 3500000 ]; do sleep 20; done
exec 8>$L.0; flock 8
exec 9>$L.1; flock 9
F1SIM_AUR_CFG="$3" F1SIM_AUR_MAP="$4" F1SIM_LIDAR_MOUNT_X=0.12555 F1SIM_AUR_DRIVE=physics F1SIM_AUR_FLOOR=$FL F1SIM_AUR_FLOOR_TOL=0.03 CUDA_VISIBLE_DEVICES=1 timeout 1500 python3 -u -W ignore $R/tools/aurorus_solo.py $O/$1.npz $5 $6 "$W" > $O/$1.log 2>&1 8>&- 9>&-
echo "rc $?" >> $O/$1.log
exec 8>&- 9>&-
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 5 python3 -u -W ignore $R/tools/solo_table.py $O/$1.npz > $O/t_$1.txt 2>&1
touch $O/$1.done
