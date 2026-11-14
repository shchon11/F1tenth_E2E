#!/bin/bash
# (11-15 08:20) plan26r u500 둘째 시드 시험
R=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01; T=$R/tools; O=$R/out/exam_s2; mkdir -p $O
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
export PYTHONPATH=/home/shchon11/F1tenth_E2E/f1sim:/home/shchon11/F1tenth_E2E/f1sim_ros WANDB_MODE=disabled F1SIM_RACELINE_CACHE_ONLY=1
export F1SIM_PLAN_LAT_MARGIN=0.95 F1SIM_LIDAR_MOUNT_X=0.12555
cd /home/shchon11/F1tenth_E2E/f1sim
until [ -e /home/shchon11/f1sim_runs/plan26r/plan_u500.pt ]; do sleep 20; done; sleep 20
for r in plan26r; do
  N=$O/exam1s2_${r}_u500.npz
  [ -e $N ] || F1SIM_EXAM_SEED=20261116 CUDA_VISIBLE_DEVICES=1 bash $T/slot5060.sh timeout 2400 python3 -u -W ignore $T/exam_record.py /home/shchon11/f1sim_runs/$r/plan_u500.pt $N ideal 256 uniform > ${N%.npz}.log 2>&1
  cd $T; EXAM_ROWS=$O/boxrows_${r}_u500.npy python3 -W ignore exam_boxes.py $N > $O/boxes_${r}_u500.txt 2>&1; cd /home/shchon11/F1tenth_E2E/f1sim
done
cd $T
cd $T
until [ -e $R/out/analysis/plan26r_u500_exam.done ]; do sleep 30; done
EXAM_ROWS=$R/out/edge1114/boxrows_plan26r_u500.npy python3 -W ignore exam_boxes.py $R/out/exam/exam1_plan26r_u500_ideal_uniform_n256.npz > $R/out/edge1114/boxes_plan26r_u500.txt 2>&1
for s in 1 2; do
  if [ $s = 1 ]; then A=$R/out/exam/exam1_plan26p_u250_ideal_uniform_n256.npz; B=$R/out/exam/exam1_plan26r_u500_ideal_uniform_n256.npz; else A=$O/exam1s2_plan26p_u250.npz; B=$O/exam1s2_plan26r_u500.npz; fi
  python3 -W ignore grad_step_table.py $A "plan26r u300 vs plan26p u250 (시드$s)=$B" 2>&1 | tail -1
done
python3 - $R/out/edge1114/boxrows_plan26r_u500.npy $O/boxrows_plan26r_u500.npy <<'PY'
import sys, numpy as np
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
def wil(k, n):
    p = k / n; z = 1.96; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.1f} % [{100 * max(0, c - h):.0f}-{100 * min(1, c + h):.0f}] ({k}/{n})"
r = np.concatenate([np.load(f) for f in sys.argv[1:]]); on = r[:, IGAP] < 0.40
fast = on & (r[:, IVIN] >= 6.5) & (r[:, ISEE] < 1.0)
print(f"plan26r u300 (두 시드): 빠르고 늦게 보인 칸 세게 {wil(int((r[fast, IHARD] > 0).sum()), int(fast.sum()))} · 상자 세게 전체 {wil(int((r[on, IHARD] > 0).sum()), int(on.sum()))}")
PY
