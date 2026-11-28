"""H5 (보상상 박는 게 합리적인가)와 H6 (끼였을 때 빠져나오나)의 기록 측정. CPU만.
H5: 세게 닿은 사건마다 시간 손실 = 그 자리 5 m 앞 -> 20 m 뒤 구간을 지나는 데 걸린 시간 - 같은 세션 안 닿은 바퀴들의 같은 구간 중앙값
    (트랙 따라 남은 거리 rem으로 구간을 잼). 통과당 세게 닿을 확률(여유 구간별)과 함께 '여유를 넓히기'의 손익분기를 계산.
H6: 멈춤(5 s 넘게 트랙 따라 1 m 못 감)마다 그동안 속도 부호(후진했나), 계획 점이 차 뒤로 가나(후진 계획), 얼마나 멈췄나, 어떻게 풀렸나.
인자: events.json"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ev_lib import Rec, signed_clear, body_poly

E = json.load(open(sys.argv[1]))
for lab, r in E.items():
    R = Rec(r["file"]); st = R.st.astype(float); dt = R.dt
    L = float(R.z["lapL"][0])
    rem = R.z["rem"].astype(float)
    for t in range(1, rem.shape[0]):
        bad = ~np.isfinite(rem[t]); rem[t, bad] = rem[t - 1, bad]
    dd = np.mod(rem[:-1] - rem[1:] + 0.5 * L, L) - 0.5 * L
    P = np.concatenate([np.zeros((1, R.B)), np.cumsum(dd, 0)], 0)          # progress [m]
    s_on_lap = np.mod(L - rem, L)                                           # place along the lap
    losses = []
    for h in r["hard"]:
        b, t = h["b"], h["t"]
        if t < 10:
            continue
        p0 = P[t, b] - 5.0; p1 = P[t, b] + 20.0
        def cross(Pb, x):
            i = np.nonzero(Pb >= x)[0]
            return i[0] if len(i) else None
        a = cross(P[:, b], p0); z = cross(P[:, b], p1)
        if a is None:
            continue
        T_hit = ((z if z is not None else R.T) - a) * dt
        # the same stretch on other laps of the same session (laps not touching a box there)
        place0 = s_on_lap[a, b]
        ref = []
        k = 0
        while True:
            # next time the car is at place0 on a later/earlier lap
            cand = np.nonzero(np.abs(np.mod(s_on_lap[:, b] - place0 + 0.5 * L, L) - 0.5 * L) < 0.3)[0]
            break
        laps = []
        last = -100
        for c in cand:
            if c - last > 30:
                laps.append(c); last = c
        for c in laps:
            if abs(c - a) < 30:
                continue
            zz = cross(P[c:, b] - P[c, b], 25.0)
            if zz is None:
                continue
            if R.d[c:c + zz, b, 4].sum() > 0 or R.d[c:c + zz, b, 1].sum() > 0:
                continue
            ref.append(zz * dt)
        if ref:
            losses.append(dict(b=b, t=t, loss=T_hit - float(np.median(ref)), stuck=z is None, v_hit=h["v_hit"]))
    Ls = np.array([x["loss"] for x in losses])
    print(f"== {lab}: 세게 닿은 사건 {len(losses)}건의 시간 손실 (5 m 앞 -> 20 m 뒤 구간, 같은 세션의 안 닿은 바퀴 대비) "
          f"중앙 {np.median(Ls):.2f} s, 평균 {Ls.mean():.2f} s, 3 s 넘음 {np.mean(Ls > 3) * 100:.0f} %, 끝까지 못 감 {sum(x['stuck'] for x in losses)}")
    # passes and hard rate per pass
    enc = [e for e in r["enc"] if e["v"] > 1.0 or e["hard"] > 0]
    nh = sum(e["hard"] > 0 for e in enc)
    print(f"   통과(만남) {len(enc)}개 중 세게 닿은 만남 {nh} ({100 * nh / len(enc):.2f} %)")
    # ---- H6 stalls
    n = 50
    stalls = []
    for b in range(R.B):
        t = 10
        while t < R.T - n:
            if P[t + n, b] - P[t, b] < 1.0:
                t1 = t + n
                while t1 < R.T and P[t1, b] - P[t1 - n, b] < 1.0:
                    t1 += 1
                v = st[t:t1, b, 3]
                # plan points behind the car (in the car frame) = a reverse plan
                rel = R.pl[t:t1, b].astype(float) - st[t:t1, b, None, :2]
                c, s_ = np.cos(st[t:t1, b, 2])[:, None], np.sin(st[t:t1, b, 2])[:, None]
                lon = rel[..., 0] * c + rel[..., 1] * s_
                polys = R.boxes(t + 5, b)
                body = body_poly(*st[t + 5, b, :3], R.hl, R.hw)[None]
                on_box = bool(polys) and min(signed_clear(body, p)[0] for p in polys) < 0.05
                stalls.append(dict(b=b, t0=t, dur=(t1 - t) * dt, rev_frac=float((v < -0.05).mean()), vmin=float(v.min()),
                                   plan_back_frac=float((lon[:, -1] < -0.05).mean()), plan_len_med=float(np.median(np.abs(lon[:, -1]))),
                                   on_box=on_box, ended=t1 < R.T, vmax=float(np.abs(v).max()),
                                   n_hard=int(R.d[t:t1, b, 3:5].sum()), n_any=int(R.d[t:t1, b, 0:2].sum())))
                t = t1 + 1
            else:
                t += 1
    print(f"   멈춤(5 s 넘게 1 m 못 감) {len(stalls)}건:")
    for x in stalls:
        print(f"     세션 {x['b']:3d} {x['t0'] * dt:5.1f} s부터 {x['dur']:5.1f} s · 상자에 붙음 {x['on_box']} · 후진한 결정 {100 * x['rev_frac']:3.0f} % (최저 {x['vmin']:.2f} m/s)"
              f" · 1 s 뒤 계획점이 차 뒤 {100 * x['plan_back_frac']:3.0f} % · 계획 길이 중앙 {x['plan_len_med']:.2f} m · 최고 속도 {x['vmax']:.2f} · 그동안 접촉 {x['n_any']} (세게 {x['n_hard']}) · 풀림 {x['ended']}")
    neg = (st[:-1, :, 3] < -0.05)
    print(f"   기록 전체에서 후진한 결정 {neg.sum()}개 ({100 * neg.mean():.2f} %), 후진한 세션 {neg.any(0).sum()}개")
