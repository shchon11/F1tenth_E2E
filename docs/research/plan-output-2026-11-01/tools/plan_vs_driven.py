"""plan vs the path actually driven, from an exam record (11-14; exam_record.py keeps every decision's 10 plan points in the
world and the car's pose): for each plan point (0.1..1.0 s) the distance to where the car was that much later, over decisions with
no restart in between. Exam = ideal execution, the policy's mean actions (no exploration noise), so this is the 11-13 measure.

    python3 plan_vs_driven.py LABEL=EXAM.npz [...]
"""
import sys
import numpy as np

for a in sys.argv[1:]:
    lab, fn = a.split("=")
    z = np.load(fn)
    pl, st, rs = z["pl"].astype(np.float64), z["st"].astype(np.float64), z["rs"]
    T, B = rs.shape
    cs = np.concatenate([np.zeros((1, B)), np.cumsum(rs, 0)], 0)          # restarts up to decision t-1
    row = []
    for k in range(1, 11):
        if k >= T:
            break
        ok = (cs[k:T] - cs[:T - k]) == 0                                    # no restart in decisions t .. t+k-1
        d = np.linalg.norm(pl[:T - k, :, k - 1] - st[k:T, :, 0:2], axis=-1)
        row.append(np.median(d[ok]) * 100)
    print(f"{lab:<34} plan 점별(0.1..1.0 s) 실제로 간 곳과의 거리 중앙 [cm]: " + " ".join(f"{x:5.1f}" for x in row))
