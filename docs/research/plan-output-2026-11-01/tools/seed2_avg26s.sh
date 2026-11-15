#!/bin/bash
# (11-15 20:40) 추천 policy(avg26s/plan_u900.pt) 둘째 시드 시험 -> plan26p u250과 두 시드로 비교
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; O=$R/out/exam_s2; E=$R/out/exam
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
N=$O/exam1s2_avg26s_u900.npz
[ -e $N ] || F1SIM_EXAM_SEED=20261116 CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/avg26s/plan_u900.pt $N ideal 256 uniform > ${N%.npz}.log 2>&1
cd $T
EXAM_ROWS=$O/boxrows_avg26s_u900.npy python3 -W ignore exam_boxes.py $N > $O/boxes_avg26s_u900.txt 2>&1
echo "시드 1:"; python3 -W ignore grad_step_table.py $E/exam1_plan26p_u250_ideal_uniform_n256.npz "추천 평균=$E/exam1_avg26s_u900_ideal_uniform_n256.npz" 2>&1 | tail -1
echo "시드 2:"; python3 -W ignore grad_step_table.py $O/exam1s2_plan26p_u250.npz "추천 평균=$N" 2>&1 | tail -1
python3 - $R/out/edge1114/boxrows_plan26p_u250.npy $O/boxrows_plan26p_u250.npy $R/out/edge1114/boxrows_avg26s_u900.npy $O/boxrows_avg26s_u900.npy <<'PY'
import sys, numpy as np
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
for name, fs in (("plan26p u250", sys.argv[1:3]), ("추천 평균", sys.argv[3:5])):
    r = np.concatenate([np.load(f) for f in fs]); on = r[:, IGAP] < 0.40
    fast = on & (r[:, IVIN] >= 6.5) & (r[:, ISEE] < 1.0)
    print(f"{name} (두 시드): 빠르고 늦게 보인 칸 세게 {100 * (r[fast, IHARD] > 0).mean():.1f} % ({int((r[fast, IHARD] > 0).sum())}/{int(fast.sum())}) · 상자 통과당 세게 {100 * (r[on, IHARD] > 0).mean():.1f} %")
PY
