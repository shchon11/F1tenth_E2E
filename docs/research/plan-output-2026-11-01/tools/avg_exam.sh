#!/bin/bash
# (11-15 10:20) plan26r u500·u550·u600 가중치 평균(avg26r/plan_u550.pt)의 시험: checkpoint마다 흔들리는 상자 충돌이 평균으로 줄어드나
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; E=$R/out/exam; X=$R/out/edge1114
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1 F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
N=$E/exam1_avg26r_u550_ideal_uniform_n256.npz
[ -e $N ] || CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/avg26r/plan_u550.pt $N ideal 256 uniform > ${N%.npz}.log 2>&1
cd $T
python3 -W ignore grad_step_table.py $E/exam1_plan26p_u250_ideal_uniform_n256.npz "plan26r u500=$E/exam1_plan26r_u500_ideal_uniform_n256.npz" "plan26r u550=$E/exam1_plan26r_u550_ideal_uniform_n256.npz" "plan26r u600=$E/exam1_plan26r_u600_ideal_uniform_n256.npz" "평균(u500·550·600)=$N" 2>&1 | tail -5
EXAM_ROWS=$X/boxrows_avg26r_u550.npy python3 -W ignore exam_boxes.py $N > $X/boxes_avg26r_u550.txt 2>&1
python3 - <<'PY'
import numpy as np
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
for nm in ("plan26r_u500", "plan26r_u550", "plan26r_u600", "avg26r_u550"):
    r = np.load(f"/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out/edge1114/boxrows_{nm}.npy"); on = r[:, IGAP] < 0.40
    fast = on & (r[:, IVIN] >= 6.5) & (r[:, ISEE] < 1.0)
    k, n = int((r[fast, IHARD] > 0).sum()), int(fast.sum()); k2, n2 = int((r[on, IHARD] > 0).sum()), int(on.sum())
    print(f"{nm}: 빠르고 늦게 보인 칸 {100 * k / n:.1f} % ({k}/{n}) · 상자 통과당 세게 {100 * k2 / n2:.1f} % ({k2}/{n2})")
PY
