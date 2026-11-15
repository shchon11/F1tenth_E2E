"""(11-15) 벽 충돌이 몰린 자리에서 plan이 벽과 얼마나 떨어져 있었나: 차가 그 자리 1.5 m 안에 있을 때 낸 plan 10점의 (벽 거리 - 차 반폭) 최소.
이상 실행과 MAP 설정별로 같은 자리를 비교 -> MAP에서 policy가 더 바짝 붙는 plan을 내는지(학습 때 못 본 상태) 아니면 같은 plan을 MAP이 못 따라가는지.
인자: x,y 이름=기록npz ..."""
import sys
import numpy as np
from rec_lib import Rec

cx, cy = (float(v) for v in sys.argv[1].split(","))
print(f"자리 ({cx}, {cy}) 1.5 m 안에서 낸 plan의 벽 여유 (plan 점 중심 - 차 반폭)")
for a in sys.argv[2:]:
    nm, fn = a.split("=", 1)
    R = Rec(fn); st = R.st.astype(float); pl = R.z["pl"].astype(float)
    room, room_car, hits = [], [], 0
    for t in range(R.T):
        near = np.hypot(st[t, :, 0] - cx, st[t, :, 1] - cy) < 1.5
        for b in np.nonzero(near & ~R.rs[t])[0]:
            k = int(R.tid[t, b])
            room.append(R.wall_dist(k, pl[t, b]).min() - R.hw)
            room_car.append(R.wall_dist(k, st[t, b:b + 1, :2])[0] - R.hw)
            hits += int(R.d[t, b, 3] > 0)
    r = np.array(room) * 100; rc = np.array(room_car) * 100
    print(f"  {nm:<14} 결정 {len(r):6d} | plan 여유 하위 5 % {np.percentile(r, 5):5.1f} cm, 10 % {np.percentile(r, 10):5.1f}, 중앙 {np.median(r):5.1f}"
          f" | plan 여유 <10 cm {np.mean(r < 10) * 100:4.1f} % | 차 중심 여유 하위 5 % {np.percentile(rc, 5):5.1f} cm | 벽 세게 {hits}")
