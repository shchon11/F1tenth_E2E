"""계획이 상자를 어느 쪽으로 지나가려 했나, 결정마다 (C의 이상 실행 기록, 만남 전부).
만남마다 가장 가까운 순간(또는 세게 닿은 결정) 1.5 s 전부터: 그 결정의 계획(차 + 10점) 가운데 상자 중심에 가장 가까운 자리에서 상자 중심의
옆 위치(계획 진행 방향 기준, + = 상자가 계획 왼쪽), 계획 차체의 상자 여유. 계획이 상자 자리에 닿지 않으면(가장 가까운 점이 계획 끝) 빼고 셈.
  쪽 바꿈 = 옆 위치 부호가 바뀐 횟수(|옆| > 5 cm인 결정만), 관통 = 계획 차체 여유 <= 0인 결정 비율.
인자: events.json out.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_lib import Rec, body_poly, signed_clear, plan_poses

E = json.load(open(sys.argv[1]))
res = {}
for lab, r in E.items():
    R = Rec(r["file"]); st = R.st.astype(float)
    hard_at = {}
    for h in r["hard"]:
        hard_at.setdefault((h["b"], h["j"]), []).append(h["t"])
    rows = []
    for e in r["enc"]:
        b, j = e["b"], e["j"]
        hits = [t for t in hard_at.get((b, j), []) if e["t0"] - 1 <= t <= e["t1"] + 1]
        te = hits[0] if hits else int(round(e["tmin"]))
        if te < 3 or e["v"] < 1.0 and not hits:
            continue
        P = R.boxes(te, b)[j]; c = P.mean(0)
        side, thr = [], []
        for t in range(max(0, te - 15), te):
            pp = plan_poses(st[t, b, :3], R.pl[t, b].astype(float))
            i = int(np.linalg.norm(pp[:, :2] - c[None], axis=1).argmin())
            if i >= len(pp) - 1:
                continue                                              # the plan does not reach the box
            hx, hy = np.cos(pp[i, 2]), np.sin(pp[i, 2])
            off = hx * (c[1] - pp[i, 1]) - hy * (c[0] - pp[i, 0])
            cl = float(signed_clear(body_poly(pp[:, 0], pp[:, 1], pp[:, 2], R.hl, R.hw), P).min())
            side.append(off); thr.append(cl <= 0.0)
        if len(side) < 3:
            continue
        sd = np.array(side)
        sg = np.sign(sd[np.abs(sd) > 0.05])
        nsw = int((np.diff(sg) != 0).sum()) if len(sg) > 1 else 0
        rows.append(dict(b=b, j=j, hit=bool(hits), any=e["any"] > 0, nsw=nsw, thr=float(np.mean(thr)), thr_last=bool(thr[-1]),
                         n=len(side), first_thr=bool(thr[0])))
    res[lab] = rows
    H = [x for x in rows if x["hit"]]; Cn = [x for x in rows if not x["any"]]
    def s(S):
        return (f"n {len(S):4d} · 쪽 바꿈 1번 이상 {100 * np.mean([x['nsw'] > 0 for x in S]):4.1f} % · 2번 이상 {100 * np.mean([x['nsw'] > 1 for x in S]):4.1f} %"
                f" · 계획 관통 결정 비율 평균 {100 * np.mean([x['thr'] for x in S]):4.1f} % · 계획이 상자에 처음 닿을 때부터 관통 {100 * np.mean([x['first_thr'] for x in S]):4.1f} %")
    print(f"== {lab}\n  세게 닿은 만남: {s(H)}\n  안 닿은 통과:   {s(Cn)}")
    sw = [x for x in rows if x["nsw"] > 0]; ns = [x for x in rows if x["nsw"] == 0]
    print(f"  쪽 바꿈이 있던 만남의 세게 닿음 {sum(x['hit'] for x in sw)}/{len(sw)} ({100 * np.mean([x['hit'] for x in sw]):.1f} %), 없던 만남 {sum(x['hit'] for x in ns)}/{len(ns)} ({100 * np.mean([x['hit'] for x in ns]):.2f} %)")
json.dump(res, open(sys.argv[2], "w"))
