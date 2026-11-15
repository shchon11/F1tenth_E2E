#!/bin/bash
# (11-15 21:40) 튜닝 중 나온 짧은 L1 설정(c002: 30 s 세션에서 벽 3.11 -> 0.23, 평균 속도 4.53 -> 5.07 m/s)을 90 s 시험 두 시드로 확인
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
CFG=$(python3 -c "import json; d=[json.loads(l) for l in open('$R/out/maptune/tune.jsonl') if json.loads(l)['i']==2][0]['cfg']; d.pop('lead'); print(json.dumps(d))")
CK=/home/shchon11/f1sim_runs/avg26s/plan_u900.pt
cd /home/shchon11/F1tenth_E2E/f1sim
A=$E/exam1_avg26s_u900_map_uniform_n256_c002.npz
B=$E/exam1_avg26s_u900_map_uniform_n256_c002_s2.npz
[ -e $A ] || F1SIM_MAP_OVERRIDE="$CFG" F1SIM_EXAM_SPEED_LEAD=0.141 CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 -u -W ignore $T/exam_record.py $CK $A map 256 uniform > ${A%.npz}.log 2>&1 &
[ -e $B ] || F1SIM_EXAM_SEED=20261116 F1SIM_MAP_OVERRIDE="$CFG" F1SIM_EXAM_SPEED_LEAD=0.141 CUDA_VISIBLE_DEVICES=1 timeout 3600 python3 -u -W ignore $T/exam_record.py $CK $B map 256 uniform > ${B%.npz}.log 2>&1 &
# the current setting on seed 2 for the pair (seed 1 exists)
S2=$E/exam1_avg26s_u900_map_uniform_n256_s2.npz
wait
[ -e $S2 ] || F1SIM_EXAM_SEED=20261116 CUDA_VISIBLE_DEVICES=1 timeout 3600 python3 -u -W ignore $T/exam_record.py $CK $S2 map 256 uniform > ${S2%.npz}.log 2>&1
cd $T
python3 -W ignore grad_step_table.py $E/exam1_avg26s_u900_map_uniform_n256.npz "짧은 L1 (c002) 시드1=$A" 2>&1 | tail -2
python3 -W ignore grad_step_table.py $S2 "짧은 L1 (c002) 시드2=$B" 2>&1 | tail -2
