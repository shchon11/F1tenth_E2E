"""느린 상대 차를 지나칠 때 두 차체 사이에 남긴 틈 (11-14 밤). car_encounters.py의 만남 행(ENC_ROWS)마다 만남 동안 두 차체 사각형
사이 거리의 최솟값(결정 0.1 s 사이는 위치·요를 고르게 10등분). 표: 앞지름으로 끝난 만남의 틈 분포(안 닿음 / 가볍게 / 세게),
틈이 가장 좁았던 순간의 우리 속도·상대 속도, 그 순간 상대 차 양옆 벽까지 거리(지나갈 자리가 있었나).
인자: 기록npz 만남행npy [기록npz 만남행npy ...]"""
import sys
import numpy as np
from rec_lib import Rec
from box_clearance import poly_gap

SUB = 10
for fn, rf in zip(sys.argv[1::2], sys.argv[2::2]):
    R = Rec(fn); dt = R.dt; st = R.st.astype(float); op = R.op.astype(float)
    rows = np.load(rf)
    q = np.array([[R.hl, R.hw], [R.hl, -R.hw], [-R.hl, -R.hw], [-R.hl, R.hw]])
    u = np.linspace(0, 1, SUB + 1)[:, None]

    def bodies(x, s, b):
        dy = (x[s + 1, b, 2] - x[s, b, 2] + np.pi) % (2 * np.pi) - np.pi
        yaw = x[s, b, 2] + u[:, 0] * dy
        p = x[s, b, :2] + u * (x[s + 1, b, :2] - x[s, b, :2])
        c, sn = np.cos(yaw), np.sin(yaw)
        return np.stack([p[:, 0, None] + q[:, 0] * c[:, None] - q[:, 1] * sn[:, None], p[:, 1, None] + q[:, 0] * sn[:, None] + q[:, 1] * c[:, None]], -1)

    out = []
    for r in rows:
        b, a, t = int(r[0]), int(r[1]), int(r[2])
        best, at = np.inf, a
        for s in range(a, min(t + 1, R.T - 1)):
            if np.linalg.norm(st[s + 1, b, :2] - st[s, b, :2]) > 2.0 or np.linalg.norm(op[s + 1, b, :2] - op[s, b, :2]) > 2.0:
                continue
            if np.linalg.norm(st[s, b, :2] - op[s, b, :2]) > 2.5:
                continue
            P, Q = bodies(st, s, b), bodies(op, s, b)
            g = min(poly_gap(P[i:i + 1], Q[i])[0] for i in range(SUB + 1))
            if g < best:
                best, at = g, s
        # room beside the other car at that moment: wall distance from its centre, minus half its width
        wd = R.wall_dist(int(R.tid[at, b]), op[at, b, None, :2])[0] - R.hw
        out.append((best, abs(st[at, b, 3]), abs(op[at, b, 3]), wd))
    o = np.array(out); hard, light = rows[:, 4] > 0, (rows[:, 4] == 0) & (rows[:, 5] > 0)
    clean = ~hard & ~light & (rows[:, 6] == 0); passed = rows[:, 3] == 0
    print(f"== {fn.split('/')[-1]}: 만남 {len(rows)}건, 그중 앞지름으로 끝남 {int(passed.sum())}건")
    m = passed & clean & np.isfinite(o[:, 0])
    g = o[m, 0]
    print(f"  안 닿고 앞지른 {int(m.sum())}건의 가장 좁은 틈: 중앙 {100 * np.median(g):.0f} cm · 아래 10 % {100 * np.percentile(g, 10):.0f} cm · 아래 25 % {100 * np.percentile(g, 25):.0f} cm · 위 25 % {100 * np.percentile(g, 75):.0f} cm")
    print("    틈 구간별 건수: " + " · ".join(f"{int(lo * 100)}-{int(hi * 100)} cm {int(((g >= lo) & (g < hi)).sum())}" for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, 9))))
    for name, mm in (("세게 닿은", hard), ("가볍게만 닿은", light), ("안 닿고 앞지른", m)):
        if mm.sum():
            print(f"  {name} 만남 {int(mm.sum())}건: 가장 가까운 순간 우리 {o[mm, 1].mean():.1f} · 상대 {o[mm, 2].mean():.1f} m/s (차이 {np.mean(o[mm, 1] - o[mm, 2]):+.1f}) | 그 순간 상대 차 옆면에서 가장 가까운 벽까지 중앙 {100 * np.median(o[mm, 3]):.0f} cm")
    np.save(rf.replace(".npy", "_gap.npy"), o)
