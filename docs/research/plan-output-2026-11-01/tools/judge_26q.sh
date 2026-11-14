#!/bin/bash
# (11-15 02:00) plan26q u250 판정: plan26p u250과 같은 세션끼리 + '6.5 m/s 넘게 · 1 s 안쪽에 보인 상자' 칸 세게 닿는 비율
O=/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/out; E=$O/exam; X=$O/edge1114
U=${1:-250}
until [ -e $O/analysis/plan26p_u${U}_exam.done ] && [ -e $O/analysis/plan26q_u${U}_exam.done ]; do sleep 30; done
source /home/shchon11/F1tenth_E2E/.venv/bin/activate
cd /home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools
python3 -W ignore grad_step_table.py $E/exam1_plan26p_u${U}_ideal_uniform_n256.npz "plan26q u${U}=$E/exam1_plan26q_u${U}_ideal_uniform_n256.npz" 2>&1 | tail -4
for r in plan26p plan26q; do
  [ -e $X/boxrows_${r}_u${U}.npy ] || EXAM_ROWS=$X/boxrows_${r}_u${U}.npy nice -n 5 python3 -W ignore exam_boxes.py $E/exam1_${r}_u${U}_ideal_uniform_n256.npz > $X/boxes_${r}_u${U}.txt 2>&1
done
python3 - $X/boxrows_plan26p_u${U}.npy $X/boxrows_plan26q_u${U}.npy <<'PY'
import sys, numpy as np
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
def wil(k, n):
    if n == 0: return "-"
    p = k / n; z = 1.96; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * p:.1f} % [{100 * max(0, c - h):.0f}-{100 * min(1, c + h):.0f}] ({k}/{n})"
for f in sys.argv[1:]:
    r = np.load(f); on = r[:, IGAP] < 0.40
    fast = on & (r[:, IVIN] >= 6.5) & (r[:, ISEE] < 1.0); other = on & ~fast
    print(f"{f.split('/')[-1]}: 길 위 통과 {on.sum()} · 세게 {int((r[on, IHARD] > 0).sum())} | 빠르고 늦게 보인 칸 세게 {wil(int((r[fast, IHARD] > 0).sum()), int(fast.sum()))} · 비킨 폭 중앙 {100 * np.median(r[fast, ISHIFT]):.0f} cm | 나머지 세게 {wil(int((r[other, IHARD] > 0).sum()), int(other.sum()))}")
PY
