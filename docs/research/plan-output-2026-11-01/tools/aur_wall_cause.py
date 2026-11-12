"""포팅본의 벽 접촉이 '고른 경로가 벽에 가까워서'인지 '경로는 떨어져 있었는데 차가 벗어나서'인지 (기록만으로, 11-12).
접촉 0.3 s 전에 쥐고 있던 회피 경로에서 접촉 지점과 가장 가까운 점의 벽 거리(경로 중심) + 접촉 때 차 중심이 그 경로에서 벗어난 거리.
인자: 기록npz [기록npz ...]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rec_lib as RL

HW = 0.155


def seg_dist(p, P):
    """distance from point p to polyline P (K, 2) and the index of the nearest vertex"""
    a, b = P[:-1], P[1:]; ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
    q = a + t[:, None] * ab
    d = np.hypot(*(p - q).T)
    j = int(d.argmin())
    return float(d[j]), q[j]


for fn in sys.argv[1:]:
    r = RL.Rec(fn); z = r.z; T, B = r.T, r.B
    st = r.st.astype(float); path = z["path"][:T].astype(float); avoid = z["avoid"][:T]; ps = z["pscale"][:T]
    print(f"== {os.path.basename(fn)} ==")
    for sc in [float(x) for x in z["scales"]]:
        rows = []
        n_plain = 0
        for col, name in ((0, "all"),):
            tt, bb = np.nonzero((r.d[..., 0] > 0) & (np.abs(ps - sc) < 1e-3))
            for t, b in zip(tt, bb):
                hard = r.d[t, b, 3] > 0
                P = None
                for back in (3, 2, 4, 1, 5):
                    if t - back >= 0 and avoid[t - back, b] and np.isfinite(path[t - back, b]).all() and not r.rs[t - back:t, b].any():
                        P = path[t - back, b]; break
                if P is None:
                    n_plain += 1; continue
                k = int(r.tid[t, b])
                # the car where the contact began: the state entering decision t (contact is within the next 0.1 s) and leaving it
                best = None
                for s_ in (st[t, b], st[t + 1, b]):
                    e, q = seg_dist(s_[:2], P)
                    if best is None or e < best[0]:
                        best = (e, q)
                e, q = best
                c_near = float(r.wall_dist(k, q[None])[0])                       # path centre's wall distance beside the car
                c_min = float(r.wall_dist(k, P).min())                           # the whole path's smallest
                rows.append((hard, c_near, c_min, e, float(st[t, b, 3])))
        if not rows:
            print(f"{sc:g}배: 회피 중 벽 접촉 없음 (회피 아닐 때 {n_plain}건)"); continue
        a = np.array(rows, float); n = len(a)
        near = a[:, 1]; err = a[:, 3]
        close = near < HW + 0.10                                                  # body edge within 0.10 m of the wall if followed exactly
        far_off = (~close) & (err >= 0.10)
        print(f"{sc:g}배: 회피 경로를 쥔 채 벽 접촉 {n}건(그중 세게 {int(a[:, 0].sum())}) · 회피 아닐 때 {n_plain}건")
        print(f"   접촉 지점 옆 경로 중심의 벽 거리: 중앙값 {np.median(near):.2f} m · 0.255 m 미만 {100 * close.mean():.0f} % · 0.30 m 미만 {100 * (near < 0.30).mean():.0f} % · 0.35 m 미만 {100 * (near < 0.35).mean():.0f} %")
        print(f"   경로 전체에서 가장 가까운 벽 거리: 중앙값 {np.median(a[:, 2]):.2f} m")
        print(f"   접촉 때 차 중심이 경로에서 벗어난 거리: 중앙값 {np.median(err):.2f} m · 0.10 m 이상 {100 * (err >= 0.10).mean():.0f} %")
        print(f"   나눔: 경로가 벽 가까이(0.255 m 미만) {100 * close.mean():.0f} % | 경로는 떨어져 있었고 차가 0.10 m 이상 벗어남 {100 * far_off.mean():.0f} % | 그 밖 {100 * (1 - close.mean() - far_off.mean()):.0f} %")
