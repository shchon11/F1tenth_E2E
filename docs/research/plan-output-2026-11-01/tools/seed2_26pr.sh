#!/bin/bash
# (11-15 04:08) plan26r u250 판정의 '빠르고 늦게 보인 상자' 칸 표본이 70개 정도라(구간 겹침) 둘째 시드로 시험 한 번씩 더(5060, 하나씩)
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; O=$R/out/exam_s2; mkdir -p $O
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
for r in plan26p plan26r; do
  N=$O/exam1s2_${r}_u250.npz
  [ -e $N ] || F1SIM_EXAM_SEED=20261116 CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/$r/plan_u250.pt $N ideal 256 uniform > ${N%.npz}.log 2>&1
  cd $T; EXAM_ROWS=$O/boxrows_${r}_u250.npy python3 -W ignore exam_boxes.py $N > $O/boxes_${r}_u250.txt 2>&1; cd /home/shchon11/F1tenth_E2E/f1sim
done
cd $T
python3 -W ignore grad_step_table.py $O/exam1s2_plan26p_u250.npz "plan26r u250 (시드2)=$O/exam1s2_plan26r_u250.npz" 2>&1 | tail -3
python3 - $R/out/edge1114/boxrows_plan26p_u250.npy $O/boxrows_plan26p_u250.npy $R/out/edge1114/boxrows_plan26r_u250.npy $O/boxrows_plan26r_u250.npy <<'PY'
import sys, numpy as np
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
def wil(k, n):
    p = k / n; z = 1.96; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.1f} % [{100 * max(0, c - h):.0f}-{100 * min(1, c + h):.0f}] ({k}/{n})"
for name, fs in (("plan26p u250", sys.argv[1:3]), ("plan26r u250", sys.argv[3:5])):
    r = np.concatenate([np.load(f) for f in fs]); on = r[:, IGAP] < 0.40
    fast = on & (r[:, IVIN] >= 6.5) & (r[:, ISEE] < 1.0)
    print(f"{name} (두 시드): 빠르고 늦게 보인 칸 세게 {wil(int((r[fast, IHARD] > 0).sum()), int(fast.sum()))} · 상자 세게 전체 {wil(int((r[on, IHARD] > 0).sum()), int(on.sum()))}")
PY
