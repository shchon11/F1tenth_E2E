"""(11-16) 상자 세션에서 차가 서서 다시 안 가는 것: frozen exam 기록에서 2 s 넘게 선(|v| < 0.3 m/s) 구간마다 시작 시각, 길이, 끝까지
섰나, 시작할 때 가장 가까운 상자까지 거리(차 중심 - 상자 꼭짓점), 상자를 치운 뒤(rm_t)에도 섰나, 서 있는 동안 plan이 얼마나 가자고 했나
(plan 1.0 s 점까지 거리), 그 동안 접촉. 인자: 기록npz [기록npz ...]"""
import sys
import numpy as np
from rec_lib import Rec

for fn in sys.argv[1:]:
    R = Rec(fn); st = R.st.astype(float); pl = R.z["pl"].astype(float); dt = R.dt
    rm_t = float(R.z["rm_t"]) if "rm_t" in R.z.files else 50.0
    kind = R.z["kind"]; kinds = R.kinds
    rows = []
    for b in range(R.B):
        v = st[:-1, b, 3]
        still = np.abs(v) < 0.3
        t = 0
        while t < len(still):
            if still[t]:
                t0 = t
                while t < len(still) and still[t]:
                    t += 1
                if (t - t0) * dt >= 2.0 and t0 * dt > 1.0:
                    poly = R.boxes(t0, b)
                    dmin = min((np.min(np.linalg.norm(p - st[t0, b, :2], axis=1)) for p in poly), default=np.inf) if poly else np.inf
                    plan_go = np.median(np.linalg.norm(pl[t0:t, b, -1] - st[t0:t, b, :2], axis=1))
                    rows.append(dict(b=b, k=kinds[int(kind[0, b])] if kinds else "", t0=t0 * dt, dur=(t - t0) * dt, end=t >= len(still),
                                     box=dmin, after_rm=(t0 * dt < rm_t <= t * dt) or t0 * dt >= rm_t, go=plan_go,
                                     hit=float(R.d[t0:t, b, 3:6].sum())))
            else:
                t += 1
    print(f"== {fn.split('/')[-1]}: 2 s 넘게 선 구간 {len(rows)}개 (세션 {R.B}개)")
    if not rows:
        continue
    a = rows
    print(f"  끝까지 섬 {sum(r['end'] for r in a)}개 · 상자 치운 뒤에도 섬 {sum(r['after_rm'] for r in a)}개 · 시작할 때 가장 가까운 상자 중앙 "
          f"{np.median([r['box'] for r in a]):.2f} m · 서 있는 동안 plan 1.0 s 점까지 중앙 {np.median([r['go'] for r in a]):.2f} m · 접촉 있던 것 {sum(r['hit'] > 0 for r in a)}개")
    for r in sorted(a, key=lambda r: -r['dur'])[:8]:
        print(f"    세션 {r['b']:3d} {r['k']:<14} {r['t0']:5.1f} s부터 {r['dur']:5.1f} s{' (끝까지)' if r['end'] else ''} | 상자 {r['box']:.2f} m | "
              f"plan 1 s 점 {r['go']:.2f} m | 치운 뒤에도 {'예' if r['after_rm'] else '아니오'} | 접촉 {r['hit']:.0f}")
