"""(11-27) What the at-fault car contacts of an exam4 record are: every decision the reward charged a car contact to us
(cfault > 0), by the scene at that moment.
  side by side   the other car's centre less than 0.3 m ahead or behind ours along our heading (|dx| <= 0.3 m)
  rear-end       the other car ahead (dx > 0.3 m); split by whether we had been behind it within 1.5 m for >= 2 s before
                 (following / tailgating) or came up on it (closing)
  other          the other car behind us (dx < -0.3 m) but the contact charged to us anyway
and per kind: our speed, the other car's speed, the closing speed along the line between the centres, whether it repeats an
at-fault contact of the same session within 2 s, whether we were stopped (< 0.5 m/s), the opponent pace of the session.
usage: car_contact_kinds.py EXAM4.npz [...]"""
import sys

import numpy as np

for fn in sys.argv[1:]:
    z = np.load(fn)
    dt = float(z["dt"]); f = z["cfault"].astype(float); T, B = f.shape
    st, op = z["st"][:T].astype(float), z["op"][:T].astype(float)
    pace = z["c_pace"][:B].astype(float)
    ev = np.argwhere(f > 0)
    rows = []
    for t, b in ev:
        x, y, yaw, v = st[t, b]; ox, oy, oyaw, ov = op[t, b]
        c, s = np.cos(yaw), np.sin(yaw)
        dx = (ox - x) * c + (oy - y) * s; dy = -(ox - x) * s + (oy - y) * c
        dist = max(np.hypot(ox - x, oy - y), 1e-6)
        ux, uy = (ox - x) / dist, (oy - y) / dist
        close = v * (np.cos(yaw) * ux + np.sin(yaw) * uy) - ov * (np.cos(oyaw) * ux + np.sin(oyaw) * uy)
        t0 = max(0, t - int(2.0 / dt))
        sx = (op[t0:t, b, 0] - st[t0:t, b, 0]) * np.cos(st[t0:t, b, 2]) + (op[t0:t, b, 1] - st[t0:t, b, 1]) * np.sin(st[t0:t, b, 2])
        foll = t - t0 >= int(2.0 / dt) - 1 and bool(((sx > 0) & (sx < 1.5)).all())
        rep = bool((f[t0:t, b] > 0).any())
        kind = "side" if abs(dx) <= 0.3 else ("rear_follow" if (dx > 0.3 and foll) else ("rear_close" if dx > 0.3 else "other"))
        rows.append((kind, v, ov, close, rep, v < 0.5, pace[b], abs(dy), f[t, b]))
    n = len(rows); nb = B
    print(f"== {fn.split('/')[-1]}: 우리 책임 차 접촉 {sum(r[8] for r in rows):.0f}건 (결정 {n}개) / 세션 {nb}개")
    names = {"side": "옆으로 나란히 (앞뒤 0.3 m 안)", "rear_follow": "앞차 뒤에 2 s 넘게 붙어 가다 박음", "rear_close": "다가가다 앞차를 박음",
             "other": "상대가 뒤인데 우리 책임"}
    for k in ("rear_follow", "rear_close", "side", "other"):
        r = [x for x in rows if x[0] == k]
        if not r:
            print(f"  {names[k]:<26} 0"); continue
        a = np.array([x[1:8] for x in r], float)
        p = a[:, 5]
        print(f"  {names[k]:<26} {len(r):4d} ({100 * len(r) / n:3.0f} %) | 우리 {np.median(a[:, 0]):.2f} m/s 상대 {np.median(a[:, 1]):.2f} | "
              f"다가간 속도 {np.median(a[:, 2]):.2f} m/s | 2 s 안 반복 {100 * a[:, 3].mean():3.0f} % | 서 있음 {100 * a[:, 4].mean():3.0f} % | "
              f"옆 간격 {np.median(a[:, 6]):.2f} m | 상대 0.6배 {100 * (p < 0.7).mean():3.0f} %")
