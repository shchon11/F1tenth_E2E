#!/bin/bash
# critic 값까지 남긴 기록(exam_value_run.sh)에서, 닿는 상자를 critic이 닿기 몇 초 전부터 값에 치는지 봄. CPU만.
cd /home/shchon11/F1tenth_E2E/f1sim && source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PROP_BANK=/home/shchon11/f1sim_runs/prop_banks_box2 F1SIM_PROP_COUNT=uniform F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01
O=$R/out/netaudit/value_plan25y_u125
RECS=(); for sd in "$@"; do RECS+=($R/out/exam/examv_plan25y_u125_ideal_uniform_n256_s$sd.npz); done
NET_HITS_VALUE=1 python3 -W ignore $R/tools/net_audit_hits.py /home/shchon11/f1sim_runs/plan25y/plan_u125.pt $O "${RECS[@]}" > $O.log 2>&1
echo "rc $?" >> $O.log; touch $O.done
cd $R/tools && python3 -W ignore net_audit_value_probe.py $O.value.npz $O.probe "plan25y u125: 세게 닿는 상자를 critic과 actor가 언제부터 다르게 다루나 (시험 기록 ${#RECS[@]}개)" > $O.probe.log 2>&1
echo "rc $?" >> $O.probe.log; touch $O.probe.done
