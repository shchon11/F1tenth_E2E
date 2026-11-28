"""정책이 상자에 '반응'했나: 같은 세션의 상자 치운 뒤(65 s~) 빈 트랙 주행선을 기준선으로, 상자로 다가가는 실제 경로의 옆 어긋남.

상자마다 기준선 = 치운 뒤 그 상자 자리에 가장 가까이 지난 첫 바퀴의 경로(그 순간 3 s 전 ~ 1 s 뒤, 촘촘한 궤적).
  box_lat = 상자 중심이 기준선에서 옆으로 떨어진 거리 (+ = 진행 방향 왼쪽)
  만남마다 dev(s) = 차 중심이 기준선에서 옆으로 떨어진 거리, s = 기준선을 따라 상자 자리까지 남은 거리(음수 = 상자 앞)
  dev0 = 상자 6-8 m 앞(아직 안 보이거나 막 보일 때)의 어긋남, dev_box = 상자 자리(s=0, 못 갔으면 마지막) 어긋남,
  반응 = |dev - dev0|가 처음 5 cm를 넘은 자리(s)와 시각.
또 상자 양옆 벽까지 빈 폭(벽에 붙은 상자인가)을 붙임.
인자: events.json out.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_lib import Rec, Caster, dense_traj, ray_free

E = json.load(open(sys.argv[1]))
res = {}
for lab, r in E.items():
    R = Rec(r["file"]); C = Caster(R); st = R.st.astype(float)
    rm_k = r["rm_k"]
    out = []
    by = {}
    for e in r["enc"]:
        by.setdefault((e["b"], e["j"]), []).append(e)
    hard_at = {}
    for h in r["hard"]:
        hard_at.setdefault((h["b"], h["j"]), []).append(h["t"])
    for (b, j), L in by.items():
        P = R.boxes(0, b)[j]; c = P.mean(0)
        # reference line: the first post-removal lap past this spot
        k0 = rm_k + 50
        d = np.linalg.norm(st[k0:, b, :2] - c[None], axis=1)
        near = np.nonzero(d < 2.0)[0]
        if len(near) == 0:
            continue
        kc = k0 + near[np.argmin(d[near[:20]])] if len(near) else None
        tr, _ = dense_traj(st, max(k0, kc - 30), min(R.T, kc + 10), b)
        seg = np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1)
        sref = np.concatenate([[0], np.cumsum(seg)])
        tang = np.gradient(tr[:, :2], axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)

        def proj(pts):
            dd = np.linalg.norm(pts[:, None, :] - tr[None, :, :2], axis=-1)
            i = dd.argmin(1)
            lat = ((pts - tr[i, :2]) * nrm[i]).sum(1)
            lon = sref[i] + ((pts - tr[i, :2]) * tang[i]).sum(1)
            return lon, lat, dd.min(1)
        lb, latb, _ = proj(c[None])
        lb, latb = float(lb[0]), float(latb[0])
        # wall room either side of the box across the line
        ib = int(np.linalg.norm(tr[:, :2] - c[None], axis=1).argmin())
        n_ = nrm[ib]
        loc = ((P - c[None]) * n_).sum(1)
        room_left = ray_free(C, P[np.argmax(loc)] + 0.01 * n_, n_)
        room_right = ray_free(C, P[np.argmin(loc)] - 0.01 * n_, -n_)
        L = sorted(L, key=lambda e: e["t0"])
        prev_end = 0
        for n_enc, e in enumerate(L):
            hits = [t for t in hard_at.get((b, j), []) if e["t0"] - 1 <= t <= e["t1"] + 1]
            t_end = hits[0] if hits else int(round(e["tmin"]))
            ta = max(prev_end, t_end - 25)
            prev_end = e["t1"]
            trA, ttA = dense_traj(st, ta, t_end, b)
            lon, lat, dist = proj(trA[:, :2])
            s = lon - lb
            ok = (dist < 1.5) & (s > -9) & (s < 1.0)
            if ok.sum() < 5:
                continue
            s, lat, ttA = s[ok], lat[ok], ttA[ok]
            base = (s > -8) & (s < -6)
            dev0 = float(np.median(lat[base])) if base.any() else float(lat[0])
            dl = lat - dev0
            onset = np.nonzero(np.abs(dl) > 0.05)[0]
            on_s = float(s[onset[0]]) if len(onset) else np.nan
            on_t = float(ttA[onset[0]]) if len(onset) else np.nan
            # dev at stations
            def at(x):
                if s.min() > x or s.max() < x:
                    return np.nan
                return float(np.interp(x, s, lat))
            out.append(dict(b=int(b), j=int(j), enc=n_enc, hit=bool(hits), any=e["any"], clear=e["clear"], v=e["v"],
                            box_lat=latb, room_left=float(room_left), room_right=float(room_right), dev0=dev0,
                            dev_m3=at(-3.0), dev_m2=at(-2.0), dev_m1=at(-1.0), dev_m05=at(-0.5), dev_end=float(lat[-1]),
                            s_end=float(s[-1]), on_s=on_s, on_t=on_t, t_end=int(t_end)))
    res[lab] = out
    print(lab, len(out), "approaches", sum(o["hit"] for o in out), "with hard hits")
json.dump(res, open(sys.argv[2], "w"))
