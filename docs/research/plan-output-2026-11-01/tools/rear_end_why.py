"""(11-27) Why do 'closing rear-ends' (car_contact_kinds.py: the other car > 0.3 m ahead along our heading, we had NOT been
2 s behind it) happen -- 63-70 % of our at-fault car contacts in exam4? For each such at-fault decision, the 2 s before:
  other car braking   its speed fell by >= 0.8 m/s over the last 1 s (it braked into a corner / for something)
  we were pulling out the lateral offset of the other car in our frame grew by >= 0.15 m over the last 1 s (we were going
                      round it) -- the contact is the pass's first step
  we braked late      our speed over the last 1 s fell by < 0.3 m/s while closing (no real braking before the contact)
  gap / closing speed / time to contact 1 s before (gap / closing speed)
and where on the lap they happen (10 bins).  usage: rear_end_why.py EXAM4.npz [...]"""
import sys

import numpy as np

for fn in sys.argv[1:]:
    z = np.load(fn); dt = float(z["dt"]); f = z["cfault"].astype(float); T, B = f.shape
    st, op = z["st"][:T + 1].astype(float), z["op"][:T + 1].astype(float); L = float(z["lapL"][0])
    pace = z["c_pace"][:B].astype(float)
    n1, n2 = int(1.0 / dt), int(2.0 / dt)
    rows = []
    for t, b in np.argwhere(f > 0):
        if t < n2:
            continue
        x, y, yw, v = st[t, b]; ox, oy = op[t, b, :2]
        dx = (ox - x) * np.cos(yw) + (oy - y) * np.sin(yw)
        sx = (op[t - n2:t, b, 0] - st[t - n2:t, b, 0]) * np.cos(st[t - n2:t, b, 2]) + (op[t - n2:t, b, 1] - st[t - n2:t, b, 1]) * np.sin(st[t - n2:t, b, 2])
        if not (dx > 0.3 and not ((sx > 0) & (sx < 1.5)).all()):
            continue                                                       # closing rear-ends only
        def lat(k):
            X, Y, W = st[k, b, :3]
            return -(op[k, b, 0] - X) * np.sin(W) + (op[k, b, 1] - Y) * np.cos(W)
        def lon(k):
            X, Y, W = st[k, b, :3]
            return (op[k, b, 0] - X) * np.cos(W) + (op[k, b, 1] - Y) * np.sin(W)
        obrake = op[t - n1, b, 3] - op[t, b, 3]
        ubrake = st[t - n1, b, 3] - st[t, b, 3]
        pull = abs(lat(t)) - abs(lat(t - n1))
        gap1 = lon(t - n1) - 0.58; clos1 = st[t - n1, b, 3] - op[t - n1, b, 3]
        rows.append((obrake >= 0.8, pull >= 0.15, ubrake < 0.3, gap1, clos1, gap1 / max(clos1, 0.05), abs(lat(t)), pace[b],
                     np.mod(z["rem"][t, b], L) / L, obrake, ubrake))
    a = np.array(rows, float); n = len(a)
    print(f"== {fn.split('/')[-1]}: 다가가다 앞차를 박음 (우리 책임) {n}건")
    if not n:
        continue
    print(f"  상대가 1 s 동안 0.8 m/s 넘게 감속   {100 * a[:, 0].mean():3.0f} %   (감속량 중앙값 {np.median(a[:, 9]):.2f} m/s)")
    print(f"  우리가 옆으로 빠지던 중(옆 간격 +0.15 m) {100 * a[:, 1].mean():3.0f} %")
    print(f"  우리가 거의 감속 안 함(1 s 동안 < 0.3 m/s) {100 * a[:, 2].mean():3.0f} %   (우리 감속 중앙값 {np.median(a[:, 10]):.2f} m/s)")
    print(f"  1 s 전: 앞뒤 틈 {np.median(a[:, 3]):.2f} m · 다가가는 속도 {np.median(a[:, 4]):.2f} m/s · 닿기까지 {np.median(a[:, 5]):.2f} s | 닿을 때 옆 어긋남 {np.median(a[:, 6]):.2f} m")
    both = (a[:, 0] > 0) & (a[:, 2] > 0); print(f"  상대 감속 + 우리 감속 안 함 {100 * both.mean():.0f} % | 옆으로 빠지던 중 + 우리 감속 안 함 {100 * ((a[:, 1] > 0) & (a[:, 2] > 0)).mean():.0f} %")
    h, _ = np.histogram(a[:, 8], bins=10, range=(0, 1)); print("  트랙 위치(남은 거리/한 바퀴, 10칸):", " ".join(str(int(v)) for v in h))
    print("  상대 페이스: 0.6배 %d · 0.8배 %d · 그 위 %d" % ((a[:, 7] < 0.7).sum(), ((a[:, 7] >= 0.7) & (a[:, 7] < 0.9)).sum(), (a[:, 7] >= 0.9).sum()))
