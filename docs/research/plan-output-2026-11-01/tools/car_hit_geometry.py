"""차끼리 세게 닿은 순간의 배치: 상대가 우리 차의 앞/옆/뒤 어디에 있었나, 누가 더 빨랐나, 0.5 s 전 상대가 옆으로 얼마나 움직였나.
앞 = 우리 차 좌표로 상대 중심이 앞 0.25 m 넘게(우리가 들이받는 쪽), 뒤 = 뒤 0.25 m 넘게(상대가 우리를 들이받는 쪽), 그 사이 = 옆.
인자: 기록 npz [...]"""
import sys
import numpy as np

for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=False); dt = float(z["dt"]); T = z["cnt"].shape[0]
    st = z["st"].astype(float); op = z["op"].astype(float); on = z["on"][:T]; kinds = [str(k) for k in z["kinds"]]
    kind = z["kind"][:T]
    c = z["cnt"][:, :, 5].astype(float)
    d = np.clip(np.diff(np.concatenate([np.zeros((1, c.shape[1])), c], 0), axis=0), 0, None)
    ts, bs = np.nonzero((d > 0) & on)
    mins = on.sum() * dt / 60
    rows = []
    for t, b in zip(ts, bs):
        if np.linalg.norm(st[t + 1, b, :2] - st[t, b, :2]) > 2.0:
            continue
        t0 = max(t - 1, 0)                                  # just before the contact step
        dxy = op[t0, b, :2] - st[t0, b, :2]; cs, sn = np.cos(st[t0, b, 2]), np.sin(st[t0, b, 2])
        dx, dy = dxy[0] * cs + dxy[1] * sn, -dxy[0] * sn + dxy[1] * cs
        t5 = max(t0 - 5, 0)
        # the opponent's sideways move over the last 0.5 s, in ITS heading frame at t5
        mv = op[t0, b, :2] - op[t5, b, :2]; h = op[t5, b, 2]
        lat = -mv[0] * np.sin(h) + mv[1] * np.cos(h)
        mv2 = st[t0, b, :2] - st[t5, b, :2]; h2 = st[t5, b, 2]
        lat2 = -mv2[0] * np.sin(h2) + mv2[1] * np.cos(h2)
        rows.append((dx, dy, abs(st[t0, b, 3]), abs(op[t0, b, 3]), lat, lat2, int(kind[t, b])))
    r = np.array(rows) if rows else np.zeros((0, 7))
    print(f"== {fn.split('/')[-1]}: 상대 있는 경주 {mins:.0f}분, 차끼리 세게 닿음 {len(r)}건 ({len(r) / mins:.2f}/분)")
    if len(r) == 0:
        continue
    for name, m in (("상대가 앞(우리가 들이받음)", r[:, 0] > 0.25), ("옆", np.abs(r[:, 0]) <= 0.25), ("상대가 뒤(상대가 들이받음)", r[:, 0] < -0.25)):
        if m.sum() == 0:
            print(f"  {name}: 0건"); continue
        q = r[m]
        print(f"  {name}: {m.sum()}건 ({100 * m.mean():.0f} %) | 우리 {q[:, 2].mean():.1f} · 상대 {q[:, 3].mean():.1f} m/s | 상대가 2 m/s 아래 {100 * (q[:, 3] < 2).mean():.0f} % | "
              f"직전 0.5 s 옆 이동(중앙값) 상대 {np.median(np.abs(q[:, 4])):.2f} m · 우리 {np.median(np.abs(q[:, 5])):.2f} m")
    for k in sorted(set(r[:, 6].astype(int))):
        m = r[:, 6] == k
        print(f"    종류 {kinds[k] if 0 <= k < len(kinds) else k}: {m.sum()}건, 그중 상대가 뒤 {int((r[m, 0] < -0.25).sum())} · 옆 {int((np.abs(r[m, 0]) <= 0.25).sum())} · 앞 {int((r[m, 0] > 0.25).sum())}")
