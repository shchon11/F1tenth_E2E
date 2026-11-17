"""(11-17, 사용자: "장애물은 구조적으로 좀 쳐 안박을 수준이 되어야하는거 아니야? 왜자꾸 쳐 박는 곳에서 계속 쳐 박지")
frozen exam(같은 128세션, 같은 상자 배치) 여러 checkpoint에서 상자·벽에 세게 닿은 건을 세션·상자별로 맞춰 봄:
- 상자: 닿은 상자(닿는 순간 차에 가장 가까운 상자) -- 같은 세션의 같은 상자를 다른 checkpoint도 박았나
- 닿기 1.0 s / 0.5 s 전 plan(10점, 차 모양)이 그 상자를 지나가나 = 박기로 '계획'했나, 그때 거리·속도
- 그 상자가 처음 보인 때(차 앞 ±60도, 6 m 안, 벽에 가리지 않음은 못 봄 -> 대신 처음 4 m 안에 들어온 때)부터 닿을 때까지 몇 초
인자: 기록npz [...] (같은 시험의 checkpoint들, 시간 순)"""
import sys
import numpy as np
sys.path.insert(0, ".")
from rec_lib import Rec

CAR_L, CAR_W = 0.50, 0.30
res = {}
for fn in sys.argv[1:]:
    R = Rec(fn); z = R.z
    st, pl, sid, pose = R.st.astype(float), z["pl"].astype(float), z["sid"], z["pose"].astype(float)
    T, B = R.rs.shape
    half = {int(k[3:]): 0.5 * np.ptp(z[k], 0) for k in z.files if k.startswith("fp_")}
    ev = []
    for t, b in zip(*np.nonzero(R.d[..., 4] > 0)):
        live = np.nonzero(sid[t, b] >= 0)[0]
        if not len(live):
            continue
        dd = np.hypot(pose[t, b, live, 0] - st[t + 1, b, 0], pose[t, b, live, 1] - st[t + 1, b, 1])
        j = live[dd.argmin()]
        bx, by, byaw = pose[t, b, j]
        hx, hy = half.get(int(sid[t, b, j]), np.array([0.15, 0.15]))

        def plan_hits(tt):
            if tt < 0 or R.rs[tt, b]:
                return np.nan
            p = pl[tt, b] - [bx, by]
            c, s = np.cos(-byaw), np.sin(-byaw)
            lx, ly = p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c
            return float(((np.abs(lx) < hx + 0.5 * CAR_L) & (np.abs(ly) < hy + 0.5 * CAR_W)).any())
        n10, n05 = int(round(1.0 / R.dt)), int(round(0.5 / R.dt))
        d_all = np.hypot(pose[:t + 1, b, j, 0] - st[:t + 1, b, 0], pose[:t + 1, b, j, 1] - st[:t + 1, b, 1])
        near = np.nonzero(d_all > 4.0)[0]
        t_in = (t - near[-1]) * R.dt if len(near) else np.nan
        ev.append(dict(b=int(b), t=int(t), box=(int(b), int(j)), p10=plan_hits(t - n10), p05=plan_hits(t - n05),
                       v10=float(st[max(0, t - n10), b, 3]), v=float(st[t, b, 3]), t_in=t_in))
    res[fn] = ev
names = [f.split("exam1_")[-1].split("_ideal")[0] for f in sys.argv[1:]]
for fn, nm in zip(sys.argv[1:], names):
    ev = res[fn]
    if not ev:
        print(f"{nm}: 상자 세게 0건"); continue
    p10 = np.array([e["p10"] for e in ev]); p05 = np.array([e["p05"] for e in ev])
    boxes = {e["box"] for e in ev}
    before = set().union(*[{e["box"] for e in res[f]} for f in sys.argv[1:sys.argv[1:].index(fn) + 1] if f != fn]) if sys.argv[1:].index(fn) else set()
    rep = len(boxes & before)
    print(f"{nm}: 상자 세게 {len(ev)}건 · 상자 {len(boxes)}개 (그중 앞 checkpoint들도 박은 상자 {rep}개) | "
          f"1.0 s 전 plan이 그 상자를 지나감 {100 * np.nanmean(p10):.0f} % · 0.5 s 전 {100 * np.nanmean(p05):.0f} % | "
          f"1.0 s 전 속도 중앙 {np.median([e['v10'] for e in ev]):.1f} m/s | 4 m 안에 들어와서 닿기까지 중앙 {np.nanmedian([e['t_in'] for e in ev]):.2f} s")
