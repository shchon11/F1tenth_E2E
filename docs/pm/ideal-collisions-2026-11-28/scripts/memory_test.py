"""H3: 지난 바퀴에 본 상자를 다음 바퀴에 쓰나 -- 상자가 LiDAR에 보이기 '전' 주행이 바퀴를 거듭하며 달라지나.
상자마다 첫 만남에서 처음 보인 거리 d_vis(차 중심-상자 중심, 빔 2개 이상)를 잼. 늦게 보이는 상자 = d_vis < 5 m.
만남마다 차가 상자에서 d_vis + 1 m 떨어진 순간(아직 안 보일 때)의 속도와, 기준선(치운 뒤 첫 바퀴) 대비 옆 위치를 잼.
첫 만남 / 둘째 이후 / 앞 만남에서 닿은 뒤를 기준선과 비교. 기억을 쓴다면 늦게 보이는 상자 앞에서 둘째 바퀴부터 속도나 옆 위치가 달라져야 함.
인자: events.json out.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_lib import Rec, Caster, dense_traj

E = json.load(open(sys.argv[1]))
res = {}
for lab, r in E.items():
    R = Rec(r["file"]); C = Caster(R); st = R.st.astype(float); rm_k = r["rm_k"]; mx = float(R.z["lidar_mount_x"])
    by = {}
    for e in r["enc"]:
        by.setdefault((e["b"], e["j"]), []).append(e)
    rows = []
    for (b, j), L in by.items():
        L = sorted(L, key=lambda e: e["t0"])
        polys = R.boxes(0, b); c = polys[j].mean(0)
        # first visible distance on the first encounter
        e0 = L[0]
        tc = int(round(e0["tmin"]))
        dvis = None
        for t in range(max(0, tc - 40), tc + 1):
            n, _ = C.beams_on(st[t, b, :3], polys, j, mx)
            if n >= 2:
                dvis = float(np.linalg.norm(st[t, b, :2] - c)); break
        if dvis is None:
            continue
        dq = dvis + 1.0
        # reference: post-removal lap
        k0 = rm_k + 50
        d = np.linalg.norm(st[k0:, b, :2] - c[None], axis=1)
        cand = np.nonzero((d[1:] < dq) & (d[:-1] >= dq))[0]
        if len(cand) == 0:
            continue
        kr = k0 + cand[0] + 1
        tr, _ = dense_traj(st, max(k0, kr - 20), min(R.T, kr + 30), b)
        tang = np.gradient(tr[:, :2], axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        v_ref = abs(st[kr, b, 3])
        def lat_of(p):
            i = np.linalg.norm(tr[:, :2] - p[None], axis=1).argmin()
            return float(((p - tr[i, :2]) * nrm[i]).sum()), float(np.linalg.norm(tr[i, :2] - p))
        lat_ref, _ = lat_of(st[kr, b, :2])
        prev_hit = False
        for n_enc, e in enumerate(L):
            lo = e["t0"] - 40 if n_enc == 0 else max(L[n_enc - 1]["t1"], e["t0"] - 40)
            dd = np.linalg.norm(st[lo:e["t0"] + 20, b, :2] - c[None], axis=1)
            cc = np.nonzero((dd[1:] < dq) & (dd[:-1] >= dq))[0]
            if len(cc) == 0:
                prev_hit = prev_hit or e["any"] > 0
                continue
            k = lo + cc[0] + 1
            la, dist = lat_of(st[k, b, :2])
            if dist < 1.0:
                rows.append(dict(b=b, j=j, enc=n_enc, dvis=dvis, dv=float(abs(st[k, b, 3]) - v_ref), dlat=la - lat_ref,
                                 after_hit=prev_hit, hit=e["hard"] > 0, any=e["any"] > 0))
            prev_hit = prev_hit or e["any"] > 0
    res[lab] = rows
    A = [x for x in rows if x["dvis"] < 5.0]
    nbox = len(set((x["b"], x["j"]) for x in A))
    print(f"== {lab}: 늦게 보이는 상자(첫 만남에서 5 m 안에서 처음 보임) {nbox}개 / 전체 {len(set((x['b'], x['j']) for x in rows))}개")
    for nm, sel in (("첫 만남", [x for x in A if x["enc"] == 0]), ("둘째 이후(안 닿은 뒤)", [x for x in A if x["enc"] > 0 and not x["after_hit"]]),
                    ("앞 만남에서 닿은 뒤", [x for x in A if x["after_hit"]])):
        if not sel:
            continue
        dv = np.array([x["dv"] for x in sel]); dl = np.array([x["dlat"] for x in sel])
        print(f"   {nm:<18} n {len(sel):4d} | 안 보일 때 속도 - 빈 트랙 바퀴: 중앙 {np.median(dv):+.2f} m/s (p10 {np.percentile(dv, 10):+.2f}) | "
              f"옆 위치 차 |.| 중앙 {100 * np.median(np.abs(dl)):.1f} cm (p90 {100 * np.percentile(np.abs(dl), 90):.1f}) | 이 만남에서 세게 {sum(x['hit'] for x in sel)}")
json.dump(res, open(sys.argv[2], "w"))
