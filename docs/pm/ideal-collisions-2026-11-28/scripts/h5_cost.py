"""H5 재료: 안 닿은 통과에서 '비켜 가는 데 든 시간'과 여유의 관계 (정책 자신의 주행에서 잼).
상자마다 기준선 = 치운 뒤 첫 바퀴(reaction.py와 같음). 통과마다 상자 8 m 앞 -> 3 m 뒤 구간을 지나는 데 걸린 시간에서 기준선이 같은 구간에
걸린 시간을 뺀 값 dT, 그 구간 최대 옆 어긋남 |dev|max, 통과 여유. 상자마다(같은 자리) 통과들 사이의 차이로 '여유 1 cm당 시간'을 잼
(상자 고정효과 회귀). 인자: events.json out.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_lib import Rec, dense_traj

E = json.load(open(sys.argv[1]))
res = {}
for lab, r in E.items():
    R = Rec(r["file"]); st = R.st.astype(float); rm_k = r["rm_k"]
    by = {}
    for e in r["enc"]:
        by.setdefault((e["b"], e["j"]), []).append(e)
    rows = []
    for (b, j), L in by.items():
        P = R.boxes(0, b)[j]; c = P.mean(0)
        k0 = rm_k + 50
        d = np.linalg.norm(st[k0:, b, :2] - c[None], axis=1)
        near = np.nonzero(d < 2.0)[0]
        if len(near) == 0:
            continue
        kc = k0 + near[np.argmin(d[near[:20]])]
        tr, tt = dense_traj(st, max(k0, kc - 40), min(R.T, kc + 20), b)
        seg = np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1)
        sref = np.concatenate([[0], np.cumsum(seg)])
        tang = np.gradient(tr[:, :2], axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)

        def proj(pts):
            dd = np.linalg.norm(pts[:, None, :] - tr[None, :, :2], axis=-1)
            i = dd.argmin(1)
            return sref[i] + ((pts - tr[i, :2]) * tang[i]).sum(1), ((pts - tr[i, :2]) * nrm[i]).sum(1), dd.min(1)
        lb = float(proj(c[None])[0][0])
        sr = sref - lb
        if sr.min() > -8 or sr.max() < 3:
            continue
        T_ref = float(np.interp(3.0, sr, tt) - np.interp(-8.0, sr, tt)) * R.dt
        for e in L:
            if e["any"] > 0 or e["v"] < 1.0:
                continue
            trA, ttA = dense_traj(st, max(0, e["t0"] - 30), min(R.T, e["t1"] + 15), b)
            s, lat, dist = proj(trA[:, :2])
            s = s - lb
            ok = dist < 1.5
            s, lat, ttA = s[ok], lat[ok], ttA[ok]
            if len(s) < 5 or s.min() > -8 or s.max() < 3 or np.any(np.diff(s) < -0.5):
                continue
            T_p = float(np.interp(3.0, s, ttA) - np.interp(-8.0, s, ttA)) * R.dt
            win = (s > -8) & (s < 3)
            rows.append(dict(key=f"{b}:{j}", clear=e["clear"], dT=T_p - T_ref, devmax=float(np.abs(lat[win]).max())))
    res[lab] = rows
    dT = np.array([x["dT"] for x in rows]); cl = np.array([x["clear"] for x in rows]); dv = np.array([x["devmax"] for x in rows])
    keys = np.array([x["key"] for x in rows])
    # within-box (fixed effect) slope of dT on clearance
    xc, yc = [], []
    for k in np.unique(keys):
        m = keys == k
        if m.sum() >= 3:
            xc.append(cl[m] - cl[m].mean()); yc.append(dT[m] - dT[m].mean())
    xc = np.concatenate(xc); yc = np.concatenate(yc)
    slope = float((xc * yc).sum() / (xc * xc).sum())
    # bootstrap over boxes
    uk = np.unique(keys); sl = []
    rng = np.random.default_rng(0)
    grp = {k: np.nonzero(keys == k)[0] for k in uk}
    for _ in range(500):
        pick = rng.choice(uk, len(uk))
        X, Y = [], []
        for k in pick:
            m = grp[k]
            if len(m) >= 3:
                X.append(cl[m] - cl[m].mean()); Y.append(dT[m] - dT[m].mean())
        X = np.concatenate(X); Y = np.concatenate(Y)
        sl.append((X * Y).sum() / (X * X).sum())
    # across boxes: dT vs devmax
    A = np.vstack([dv, np.ones_like(dv)]).T
    co = np.linalg.lstsq(A, dT, rcond=None)[0]
    print(f"== {lab}: 안 닿은 통과 {len(rows)}개 | 비켜 간 시간 dT(상자 8 m 앞->3 m 뒤, 빈 트랙 바퀴 대비) 중앙 {1000 * np.median(dT):.0f} ms, "
          f"p90 {1000 * np.percentile(dT, 90):.0f} ms | 같은 상자 안에서 여유 10 cm 더 = {1000 * slope * 0.1:+.0f} ms "
          f"[{1000 * np.percentile(sl, 2.5) * 0.1:+.0f}, {1000 * np.percentile(sl, 97.5) * 0.1:+.0f}] (여유 변동 sd {100 * xc.std():.1f} cm) | "
          f"상자 사이: 옆 어긋남 10 cm당 {1000 * co[0] * 0.1:+.0f} ms")
json.dump(res, open(sys.argv[2], "w"))
