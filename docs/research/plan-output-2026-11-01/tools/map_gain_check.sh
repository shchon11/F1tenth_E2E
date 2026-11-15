#!/bin/bash
# (11-16 02:30) 조향 배율 1.0 기본값 확인: 90 s exam 두 시드, 지금 기록(1.47)과 같은 세션끼리 + 미끄럼각
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
CK=/home/shchon11/f1sim_runs/avg26s/plan_u900.pt
cd /home/shchon11/F1tenth_E2E/f1sim
A=$E/exam1_avg26s_u900_map_uniform_n256_gain100.npz
[ -e $A ] || CUDA_VISIBLE_DEVICES=0 timeout 2400 python3 -u -W ignore $T/exam_record.py $CK $A map 256 uniform > ${A%.npz}.log 2>&1
cd $T
python3 -W ignore grad_step_table.py $E/exam1_avg26s_u900_map_uniform_n256_newdefault.npz "조향 배율 1.0=$A" 2>&1 | tail -2
python3 - <<PY
import numpy as np
for nm, f in (("배율 1.47 (newdefault)", "$E/exam1_avg26s_u900_map_uniform_n256_newdefault.npz"), ("배율 1.0", "$A")):
    z = np.load(f); c = list(z["cols"]); n = z["cnt"][-1]; st = z["st"][:-1]; vx = st[:, :, 3]
    s = f"{nm}: 벽 {n[:, c.index('hard_wall')].mean():.2f} 상자 {n[:, c.index('hard_prop')].mean():.2f} 차 {n[:, c.index('hard_car')].mean():.2f} 속도 {vx.mean():.2f}"
    if "vy" in z.files:
        m = vx > 2.0; b = np.degrees(np.abs(np.arctan2(z["vy"][m], vx[m]))); s += f" 미끄럼 90 % {np.percentile(b, 90):.1f}도"
    print(s)
PY
