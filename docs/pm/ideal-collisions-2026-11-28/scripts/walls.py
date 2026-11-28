"""이상 실행의 세게 닿은 벽 사건: 가까운 상자(차체-상자 여유), 0.5 s 전 계획의 벽 여유(차 반폭 뺌), 속도, 같은 만남에서 상자에도 닿았나.
인자: events.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_lib import Rec, body_poly, signed_clear, plan_poses
E = json.load(open(sys.argv[1]))
for lab, r in E.items():
    R = Rec(r["file"]); st = R.st.astype(float)
    print(f"== {lab}")
    for t, b in zip(*np.nonzero(R.d[:, :, 3] > 0)):
        polys = R.boxes(t, b)
        body = body_poly(*st[t, b, :3], R.hl, R.hw)[None]
        bc = min([signed_clear(body, P)[0] for P in polys]) if polys else np.inf
        tp = max(0, t - 5)
        pp = plan_poses(st[tp, b, :3], R.pl[tp, b].astype(float))
        corners = body_poly(pp[:, 0], pp[:, 1], pp[:, 2], R.hl, R.hw).reshape(-1, 2)
        wc = float(R.wall_dist(0, corners).min())
        hb = int(R.d[max(0, t - 20):t + 20, b, 4].sum())
        enc = [e for e in r["enc"] if e["b"] == b and e["t0"] - 5 <= t <= e["t1"] + 5]
        print(f"  세션 {b:3d} {t * R.dt:5.1f} s · 상자까지 {bc:5.2f} m · 0.5 s 전 계획 차체 모서리의 벽 거리 최소 {wc:.2f} m · 속도 {abs(st[t, b, 3]):.1f} · "
              f"±2 s 안 상자 세게 {hb} · 상자 통과 중 {bool(enc)} (그 통과 여유 {enc[0]['clear'] if enc else float('nan'):.2f})")
