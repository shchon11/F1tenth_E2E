"""(11-27) Who moved into whom, at our at-fault car contacts (exam4). The env's rule (plan_reward.at_fault ->
env.car_contact_charge) charges this car whenever the other car is ahead along the lane OR less than a car length behind --
so while we are alongside during a pass, a contact is ours even if the other car turns into us. For each at-fault contact
decision: over the 0.3 s before, each car's velocity component along the line between the two centres, toward the other
(our approach / its approach), and the same split for the lateral part only (perpendicular to our heading). 'It came to us'
= its lateral approach > ours + 0.1 m/s. By kind (car_contact_kinds.py's: side by side |dx| <= 0.3 m / rear-end / other).
usage: who_moved_in.py EXAM4.npz [...]"""
import sys

import numpy as np

for fn in sys.argv[1:]:
    z = np.load(fn); dt = float(z["dt"]); f = z["cfault"].astype(float); T, B = f.shape
    st, op = z["st"][:T + 1].astype(float), z["op"][:T + 1].astype(float)
    k3 = max(1, int(round(0.3 / dt)))
    rows = []
    for t, b in np.argwhere(f > 0):
        if t < k3:
            continue
        x, y, yw = st[t, b, :3]; ox, oy = op[t, b, :2]
        dx = (ox - x) * np.cos(yw) + (oy - y) * np.sin(yw)
        kind = "side" if abs(dx) <= 0.3 else ("rear" if dx > 0.3 else "other")
        d = np.array([ox - x, oy - y]); n = d / max(np.linalg.norm(d), 1e-6)
        vu = (st[t, b, :2] - st[t - k3, b, :2]) / (k3 * dt); vo = (op[t, b, :2] - op[t - k3, b, :2]) / (k3 * dt)
        app_u, app_o = float(vu @ n), float(-vo @ n)                      # toward the other car
        nl = np.array([-np.sin(yw), np.cos(yw)]) * np.sign(n @ np.array([-np.sin(yw), np.cos(yw)]) or 1.0)   # our lateral, toward it
        lat_u, lat_o = float(vu @ nl), float(-vo @ nl)
        rows.append((kind, app_u, app_o, lat_u, lat_o))
    print(f"== {fn.split('/')[-1]}: 우리 책임 차 접촉 결정 {len(rows)}")
    for kd, nm in (("side", "옆으로 나란히"), ("rear", "앞차 뒤"), ("other", "상대가 뒤")):
        r = np.array([x[1:] for x in rows if x[0] == kd], float)
        if not len(r):
            continue
        came = r[:, 3] > r[:, 2] + 0.1
        print(f"  {nm:<8} {len(r):4d} | 서로 다가간 속도 중앙값: 우리 {np.median(r[:, 0]):.2f} · 상대 {np.median(r[:, 1]):.2f} m/s | "
              f"옆 방향: 우리 {np.median(r[:, 2]):.2f} · 상대 {np.median(r[:, 3]):.2f} m/s | 상대가 옆으로 더 들어온 것 {100 * came.mean():.0f} %")
