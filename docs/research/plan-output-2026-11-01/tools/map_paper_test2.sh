#!/bin/bash
# (11-15 21:00) MAP을 논문대로: 각도는 진행 방향 기준, race_stack이 덧붙인 손 보정(조향 x1.28, 가감속 중 x0.89/x0.81, 고속 축소) 없음,
# a_lat과 조향 표의 속도는 실제 속도(논문 식 2의 v_t), 앞점은 차에서 L1 거리(논문 L1 유도법), 표는 보간. L1(0.35, 0.47 v)과 지연 보상은 그대로.
# B = 논문대로, C = B + 속도 명령을 plan 0.25 s 뒤 속도로(race_stack speed_lookahead 0.25 s와 같은 목적: 모터 지연 0.2 s)
# 판정: 벽 세게/세션(MAP 지금 9.1, 이상 실행 0.05)과 plan 대비 0.4 s 뒤 바깥 밀림(map_lag.py, 지금 3-5 m/s^2에서 +13.7 cm)
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
PAPER='{"eta_ref": "velocity", "v_steer_gain_max": 1.0, "acc_scaler_for_steer": 1.0, "dec_scaler_for_steer": 1.0, "downscale_factor": 0.0, "steer_speed": "measured", "l1_point": "euclid", "lut_interp": "linear"}'
CK=/home/shchon11/f1sim_runs/avg26s/plan_u900.pt
B=$E/exam1_avg26s_u900_map_uniform_n256_paper.npz
C=$E/exam1_avg26s_u900_map_uniform_n256_paperlead.npz
# the 4070 (no training on it; the Korea 2026 bank beside it is CPU-bound), one at a time for RAM (14 GB)
cd /home/shchon11/F1tenth_E2E/f1sim
[ -e $B ] || F1SIM_MAP_OVERRIDE="$PAPER" CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 -u -W ignore $T/exam_record.py $CK $B map 256 uniform > ${B%.npz}.log 2>&1
[ -e $C ] || F1SIM_MAP_OVERRIDE="$PAPER" F1SIM_EXAM_SPEED_LEAD=0.25 CUDA_VISIBLE_DEVICES=0 timeout 3600 python3 -u -W ignore $T/exam_record.py $CK $C map 256 uniform > ${C%.npz}.log 2>&1
cd $T
python3 -W ignore grad_step_table.py $E/exam1_avg26s_u900_ideal_uniform_n256.npz "MAP 지금=$E/exam1_avg26s_u900_map_uniform_n256.npz" "MAP 논문대로=$B" "논문대로+속도 0.25 s 앞=$C" 2>&1 | tail -5
python3 map_lag.py $E/exam1_avg26s_u900_map_uniform_n256.npz $B $C 2>&1 | grep -E "==|전체|옆가속|감속 >3 \+"
