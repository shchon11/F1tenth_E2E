"""(11-15) 이상 실행에서 policy가 벽에서 얼마나 떨어져 달리나: 매 0.1 s 차체(네 모서리 + 네 변 가운데)에서 가장 가까운 벽까지.
MAP의 plan 대비 옆 오차(map_lag.py, 10-90 % 폭 ±10-20 cm)와 견줌. 인자: 기록npz [기록npz ...]"""
import sys
import numpy as np
from rec_lib import Rec

for fn in sys.argv[1:]:
    R = Rec(fn); st = R.st.astype(float)
    q = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1], [1, 0], [0, 1], [-1, 0], [0, -1]], float) * [R.hl, R.hw]
    out, near_tip = [], []
    for t in range(0, R.T, 2):
        for b in range(R.B):
            x, y, yaw = st[t, b, :3]
            c, s_ = np.cos(yaw), np.sin(yaw)
            pts = np.stack([x + q[:, 0] * c - q[:, 1] * s_, y + q[:, 0] * s_ + q[:, 1] * c], 1)
            out.append(R.wall_dist(int(R.tid[t, b]), pts).min())
            near_tip.append(np.hypot(x - 3.9, y + 3.3) < 1.5)
    a = np.array(out); m = np.array(near_tip)
    print(f"== {fn.split('/')[-1]}: 차체-벽 거리, 매 0.2 s {len(a)}개")
    for nm, sel in (("전체", np.ones(len(a), bool)), ("오른쪽 아래 섬 끝 1.5 m 안", m)):
        v = a[sel]
        print(f"  {nm:<18} <5 cm {np.mean(v < 0.05) * 100:5.1f} % | <10 cm {np.mean(v < 0.10) * 100:5.1f} % | <20 cm {np.mean(v < 0.20) * 100:5.1f} % | 중앙 {np.median(v) * 100:.0f} cm | 하위 5 % {np.percentile(v, 5) * 100:.1f} cm")
