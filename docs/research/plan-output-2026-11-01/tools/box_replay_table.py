"""box_replay.py 기록의 표와 장면 (box_replay_table.py, 11-12).
물음: 상자를 세게 치는 통과는 (H0) 복제한 순간의 상태·기억이 이미 정해 놓은 것인가, (H1) 같은 상태에서 센서 잡음만으로 갈리는 것인가.
단위 = 원본 차의 상자 통과 하나(상자가 앞에 있다가 뒤로 넘어가거나, 그 전에 그 상자에 처음 닿음). 그 일이 든 주기(복제 직후 ~ 다음 복제 직전)
안에서 같은 묶음의 복제들이 같은 상자를 어떻게 지났는지를 봄. '남은 시간' = 복제한 순간부터 원본이 그 상자를 지나거나 처음 닿을 때까지.
  H0이면 남은 시간이 길어도(1 s 이상) 원본이 세게 친 통과에서 복제도 대부분(80 % 이상) 세게 침.
  H1이면 남은 시간이 길 때 복제가 세게 치는 비율이 전체 평균 가까이로 내려가고, 남은 시간이 0에 가까울수록 올라감.
복제 갈래: A 그대로(센서 잡음만 다름) / B 기억을 지난 바퀴 같은 자리 것으로 / C 기억을 0으로. 원본의 그 주기 안 결과도 A와 같은 잣대로 셈.
옆 = 상자를 지나는 순간 차에서 본 상자의 옆 자리(+ 왼쪽). 참 위치는 이 분석에서만 씀.
인자: 출력머리말 기록npz [기록npz ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, plt, C_BOX

NEAR, FAR, AFTER = 1.5, 3.0, 10          # [m] a pass: box within NEAR at the crossing; a new pass needs FAR in between; decisions after
BINS = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.9)]
ARM = {0: "원본", 1: "A 그대로", 2: "B 지난 바퀴 기억", 3: "C 기억 0", 4: "L 왼쪽으로 틂", 5: "R 오른쪽으로 틂", 6: "S 잠깐 느리게"}
COL = {0: "#0b0b0b", 1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#8a5cd0", 5: "#d0a020", 6: "#1baf7a"}
C_HIT, MUTED = "#d03b3b", "#898781"
rng = np.random.default_rng(0)


def los(r, k, p, q):
    u = np.linspace(0.0, 1.0, 24)[:, None]
    return bool((r.wall_dist(k, p[None] * (1 - u) + q[None] * u) > 0.0).all())


def analyse(fn):
    r = Rec(fn); z = r.z
    st = r.st.astype(np.float64); T = r.T
    inc = z["inc"]; grp = z["grp"].astype(int); arm = z["arm"].astype(int); PER = int(z["period"])
    div = z["div"]; bvalid = z["bvalid"]; rs = r.rs.astype(bool)
    G = int(grp.max()) + 1
    enc, wins = [], []
    n_pass = n_drop = 0
    for g in range(G):
        rows = np.nonzero(grp == g)[0]
        assert arm[rows[0]] == 0, (g, rows, arm[rows])
        s = rows[0]; k = int(r.tid[0, s]); K = len(rows)
        cuts = [0] + [int(t) + 1 for t in np.nonzero(rs[:, s])[0]] + [T]
        has = np.zeros(len(div), bool)
        for e0, e1 in zip(cuts[:-1], cuts[1:]):
            if e1 - e0 < PER:
                continue
            live = np.nonzero(r.sid[e0, s] >= 0)[0]
            if len(live) == 0:
                continue
            P = r.pose[e0, s, live, :2].astype(float)
            X = st[e0:e1 + 1, rows]
            d = P[None, None] - X[:, :, None, :2]
            c, sn = np.cos(X[..., 2])[..., None], np.sin(X[..., 2])[..., None]
            fwd = c * d[..., 0] + sn * d[..., 1]; lat = -sn * d[..., 0] + c * d[..., 1]; dist = np.hypot(d[..., 0], d[..., 1])
            near = dist.argmin(2)
            I = inc[e0:e1, rows]
            seg = (fwd[:-1] > 0) & (fwd[1:] <= 0) & (np.minimum(dist[:-1], dist[1:]) < NEAR)

            def lat_at(tau, col, b):
                f0, f1 = fwd[tau, col, b], fwd[tau + 1, col, b]
                u = f0 / max(f0 - f1, 1e-9)
                return float(lat[tau, col, b] + u * (lat[tau + 1, col, b] - lat[tau, col, b])), float(u)

            for b in range(len(live)):
                xs = [int(t) for t in np.nonzero(seg[:, 0, b])[0] if los(r, k, X[t, 0, :2], P[b])]
                ct = np.nonzero((I[:, 0, 1] + I[:, 0, 4] > 0) & (near[:-1, 0] == b))[0]
                hd = np.nonzero((I[:, 0, 4] > 0) & (near[:-1, 0] == b))[0]
                # passes of the source at this box
                plist, start = [], 0
                marks = xs + ([int(ct[ct > (xs[-1] + AFTER if xs else -1)][0])] if (ct > (xs[-1] + AFTER if xs else -1)).any() else [])
                for j, tc in enumerate(marks):
                    crossed = j < len(xs)
                    end = tc + AFTER if crossed else e1 - e0
                    c_in = ct[(ct >= start) & (ct <= end)]; h_in = hd[(hd >= start) & (hd <= end)]
                    if crossed:
                        l0, u = lat_at(tc, 0, b)
                        td = min(float(c_in[0]), tc + u) if len(c_in) else tc + u
                    else:
                        l0, td = float(lat[tc, 0, b]), float(tc)
                    if plist and dist[plist[-1]["end"]:int(td) + 1, 0, b].max(initial=0.0) < FAR:
                        plist[-1]["hard"] |= len(h_in) > 0; plist[-1]["touch"] |= len(c_in) > 0     # same visit (backed out, crossed again)
                        plist[-1]["end"] = min(end, e1 - e0); start = end + 1
                        continue
                    plist.append(dict(td=td, tc=tc if crossed else None, lat0=l0, hard=len(h_in) > 0, touch=len(c_in) > 0,
                                      end=min(end, e1 - e0), n=len(plist) + 1))
                    start = end + 1
                for j, p in enumerate(plist):
                    n_pass += 1
                    ta = e0 + p["td"]; w = int(ta) // PER; t0 = w * PER; tend = min(t0 + PER, T)
                    lead = (ta - t0) * r.dt
                    if t0 < e0 or tend > e1 or lead >= BINS[-1][1]:
                        n_drop += 1
                        continue
                    has[w] = True
                    prev = plist[j - 1] if j else None
                    out = []
                    for col in range(K):
                        row = rows[col]; a = int(arm[row])
                        if rs[t0:tend, row].any() or (a == 2 and not bvalid[w, g]):
                            continue
                        lo, hi = t0 - e0, tend - e0
                        sg = np.nonzero(seg[lo:hi - 1, col, b])[0]
                        cc = (I[lo:hi, col, 1] + I[lo:hi, col, 4] > 0) & (near[lo:hi, col] == b)
                        hh = (I[lo:hi, col, 4] > 0) & (near[lo:hi, col] == b)
                        l0 = None
                        if len(sg):
                            l0 = lat_at(lo + int(sg[0]), col, b)[0]
                        elif cc.any():
                            l0 = float(lat[lo + int(np.nonzero(cc)[0][0]), col, b])
                        tv = lo + int(sg[0]) if len(sg) else (lo + int(np.nonzero(cc)[0][0]) if cc.any() else None)
                        out.append(dict(row=int(row), arm=a, crossed=len(sg) > 0, touch=bool(cc.any()), hard=bool(hh.any()), lat0=l0,
                                        v=float(X[tv, col, 3]) if tv is not None else None,
                                        t_hit=(t0 + int(np.nonzero(hh)[0][0])) if hh.any() else None))
                    enc.append(dict(r=r, fn=os.path.basename(fn), g=g, s=int(s), k=k, box=P[b], n=p["n"], t0=t0, tend=tend, lead=lead,
                                    td=ta, hard=bool(p["hard"]), touch=bool(p["touch"]), lat0=p["lat0"], crossed=p["tc"] is not None,
                                    prev_lat=prev["lat0"] if prev else None, prev_hard=bool(prev["hard"]) if prev else None,
                                    v0=float(st[t0, s, 3]), out=out, wall=float(r.wall_dist(k, P[b][None])[0]),
                                    bid=(os.path.basename(fn), g, round(float(P[b][0]), 2), round(float(P[b][1]), 2))))
        for w in range(len(div) - 1):                         # where a window ended: the gap measured just before the next overwrite
            for row in rows[1:]:
                if np.isfinite(div[w + 1, row]) and not (arm[row] == 2 and not bvalid[w, g]):
                    wins.append((int(arm[row]), bool(has[w]), float(div[w + 1, row])))
    return enc, wins, n_pass, n_drop


def pct(a, b):
    return f"{100.0 * a / b:5.1f} % ({a}/{b})" if b else "   -  (0/0)"


def boot(E, f):
    """ratio of sums over passes; 95 % Wilson interval with the number of PASSES as the sample size (replicates of one pass are not
    independent, so this is the cautious count); f(e) -> (numerator, denominator)"""
    v = np.array([f(e) for e in E], float).reshape(-1, 2)
    v = v[v[:, 1] > 0]
    if len(v) == 0:
        return None
    p, n, zq = v[:, 0].sum() / v[:, 1].sum(), len(v), 1.96
    mid = (p + zq * zq / (2 * n)) / (1 + zq * zq / n)
    half = zq * np.sqrt(p * (1 - p) / n + zq * zq / (4 * n * n)) / (1 + zq * zq / n)
    return p, max(mid - half, 0.0), min(mid + half, 1.0), int(v[:, 0].sum()), int(v[:, 1].sum())


def fmt(bt):
    return "   -" if bt is None else f"{100 * bt[0]:5.1f} % [{100 * bt[1]:.0f}-{100 * bt[2]:.0f}] ({bt[3]}/{bt[4]})"


def res(e, a):
    """resolved replicates of arm a in this encounter (crossed the box or touched it inside the window)"""
    return [o for o in e["out"] if o["arm"] == a and (o["crossed"] or o["touch"])]


def kind(e):
    return "세게 닿음" if e["hard"] else ("약하게 닿음" if e["touch"] else "안 닿음")


def side(x):
    return 0 if x is None else (1 if x > 0 else -1)


out_prefix, files = sys.argv[1], sys.argv[2:]
_present = sorted({int(a) for f in files for a in np.unique(np.load(f)["arm"])})     # the arms the records hold (11-13: + L, R)
ARMS0 = tuple(_present); ARMS1 = tuple(a for a in _present if a != 0)
E, W = [], []
n_pass = n_drop = 0
for fn in files:
    e_, w_, a_, b_ = analyse(fn)
    E += e_; W += w_; n_pass += a_; n_drop += b_
print(f"[복제 주행 표] 기록 {len(files)}개 ({', '.join(os.path.basename(f) for f in files)})")
print(f"원본의 상자 통과 {n_pass}번 가운데 주기 안에서 볼 수 있는 것 {len(E)}번 (주기 끝 0.1 s에 걸리거나 재시작이 낀 것 {n_drop}번 제외)")
nh, nt = sum(e["hard"] for e in E), sum(e["touch"] and not e["hard"] for e in E)
print(f"원본: 세게 닿음 {pct(nh, len(E))} · 약하게만 닿음 {pct(nt, len(E))}")
for a in ARMS0:
    bt = boot(E, lambda e: (sum(o["hard"] for o in res(e, a)), len(res(e, a))))
    print(f"  {ARM[a]:<12} 주기 안 결과: 세게 닿음 {fmt(bt)} · 판정 안 된 복제(주기 안에 상자에 못 감) "
          f"{sum(len([o for o in e['out'] if o['arm'] == a]) - len(res(e, a)) for e in E)}")

print("\n(1) 센서 잡음만 다른 복제(A)가 원본과 같은 결과를 내는가 - 남은 시간별")
print(f"{'남은 시간':<11}{'원본 결과':<10}{'통과':>5}  {'복제가 세게 닿음':<28}{'복제가 닿음(약하게 포함)':<28}{'원본과 같은 쪽':<24}{'옆 자리 차이 중앙 / 90 % [m]'}")
for lo, hi in BINS + [(1.0, 2.9), (0.0, 2.9)]:
    for kd in ("세게 닿음", "약하게 닿음", "안 닿음"):
        S = [e for e in E if lo <= e["lead"] < hi and kind(e) == kd]
        if not S:
            continue
        bh = boot(S, lambda e: (sum(o["hard"] for o in res(e, 1)), len(res(e, 1))))
        btc = boot(S, lambda e: (sum(o["touch"] for o in res(e, 1)), len(res(e, 1))))
        bs = boot(S, lambda e: (sum(side(o["lat0"]) == side(e["lat0"]) for o in res(e, 1) if o["lat0"] is not None),
                                sum(o["lat0"] is not None for o in res(e, 1))))
        dl = np.array([abs(o["lat0"] - e["lat0"]) for e in S if e["crossed"] for o in res(e, 1) if o["crossed"]])
        tag = f"{lo:.1f}-{hi:.1f} s" + (" *" if (lo, hi) in ((1.0, 2.9), (0.0, 2.9)) else "")
        print(f"{tag:<13}{kd:<10}{len(S):>5}  {fmt(bh):<30}{fmt(btc):<30}{fmt(bs):<26}"
              + (f"{np.median(dl):.2f} / {np.percentile(dl, 90):.2f} ({len(dl)})" if len(dl) else "-"))
print("  * = 구간을 합친 줄. [ ] = 95 % 구간(한 통과의 복제들은 서로 독립이 아니므로 표본 수 = 통과 수로 잡음).")

print("\n(2) 기억을 바꾼 복제 - 원본이 상자까지 1.0 s 넘게 남았을 때 복제한 통과만")
print(f"{'원본 결과':<10}{'갈래':<16}{'세게 닿음':<28}{'닿음(약하게 포함)':<28}{'원본과 같은 쪽':<26}{'옆 자리 차이 중앙 / 90 % [m]'}")
for kd in ("세게 닿음", "약하게 닿음", "안 닿음", "모두"):
    S = [e for e in E if e["lead"] >= 1.0 and (kd == "모두" or kind(e) == kd)]
    for a in ARMS0:
        if not S:
            continue
        bh = boot(S, lambda e: (sum(o["hard"] for o in res(e, a)), len(res(e, a))))
        btc = boot(S, lambda e: (sum(o["touch"] for o in res(e, a)), len(res(e, a))))
        bs = boot(S, lambda e: (sum(side(o["lat0"]) == side(e["lat0"]) for o in res(e, a) if o["lat0"] is not None),
                                sum(o["lat0"] is not None for o in res(e, a))))
        dl = np.array([abs(o["lat0"] - e["lat0"]) for e in S if e["crossed"] for o in res(e, a) if o["crossed"]])
        print(f"{kd:<10}{ARM[a]:<16}{fmt(bh):<30}{fmt(btc):<30}{fmt(bs):<28}"
              + (f"{np.median(dl):.2f} / {np.percentile(dl, 90):.2f} ({len(dl)})" if len(dl) else "-"))
print("  원본 줄 = 원본 자신을 그 주기 안에서만 본 것(잣대 확인: 원본과 같은 쪽 100 %, 차이 0이어야 함).")

print("\n(3) 지난번에 그 상자를 지난 쪽과 견줌 - 지난 통과가 있고 1.0 s 넘게 남았을 때 복제한 통과만")
print(f"{'갈래':<16}{'지난번과 다른 쪽으로 감':<28}{'같은 쪽으로 간 통과의 세게 닿음':<34}{'다른 쪽으로 간 통과의 세게 닿음'}")
S = [e for e in E if e["lead"] >= 1.0 and e["prev_lat"] is not None]
for a in ARMS0:
    O = [(o, e) for e in S for o in res(e, a) if o["lat0"] is not None]
    oth = [(o, e) for o, e in O if side(o["lat0"]) != side(e["prev_lat"])]
    sam = [(o, e) for o, e in O if side(o["lat0"]) == side(e["prev_lat"])]
    print(f"{ARM[a]:<16}{pct(len(oth), len(O)):<30}{pct(sum(o['hard'] for o, _ in sam), len(sam)):<36}{pct(sum(o['hard'] for o, _ in oth), len(oth))}")
for lab, cond in (("지난번에 세게 닿은 상자", True), ("지난번에 안 닿거나 약하게 닿은 상자", False)):
    S2 = [e for e in S if e["prev_hard"] == cond]
    print(f"  {lab}: 통과 {len(S2)}번 · " + " · ".join(
        f"{ARM[a]} 세게 {fmt(boot(S2, lambda e: (sum(o['hard'] for o in res(e, a)), len(res(e, a)))))}" for a in ARMS0))

print("\n(4) 한 통과 안에서 원본 + A 복제가 모두 같은 쪽으로 갔는가 (셋 이상 판정된 통과)")
for lo, hi in BINS:
    S = [e for e in E if lo <= e["lead"] < hi]
    U = [[side(o["lat0"]) for o in res(e, 0) + res(e, 1) if o["lat0"] is not None] for e in S]
    U = [(u, e) for u, e in zip(U, S) if len(u) >= 3]
    un = [(u, e) for u, e in U if len(set(u)) == 1]
    sp = [(u, e) for u, e in U if len(set(u)) > 1]
    hu = sum(o["hard"] for _, e in un for o in res(e, 0) + res(e, 1)); nu = sum(len(res(e, 0) + res(e, 1)) for _, e in un)
    hs = sum(o["hard"] for _, e in sp for o in res(e, 0) + res(e, 1)); ns = sum(len(res(e, 0) + res(e, 1)) for _, e in sp)
    print(f"  {lo:.1f}-{hi:.1f} s: 모두 같은 쪽 {pct(len(un), len(U))} · 그 안의 세게 닿음 {pct(hu, nu)} | 쪽이 갈린 통과의 세게 닿음 {pct(hs, ns)}")

print("\n(5) 주기 끝(3 s 뒤)에 복제가 원본에서 벌어진 거리 [m]: 중앙 / 90 % (복제-주기 수)")
for a in ARMS1:
    row = []
    for hb, lab in ((True, "상자 통과가 든 주기"), (False, "상자 통과가 없는 주기")):
        v = np.array([x[2] for x in W if x[0] == a and x[1] == hb])
        row.append(f"{lab} {np.median(v):.2f} / {np.percentile(v, 90):.2f} ({len(v)})" if len(v) else f"{lab} -")
    print(f"  {ARM[a]:<16}" + " · ".join(row))

print("\n(6) 상자가 벽에서 떨어진 거리별 (상자 가운데에서 가장 가까운 벽까지)")
print(f"{'벽까지 [m]':<12}{'통과':>6}  {'원본 세게 닿음':<24}{'A 복제 세게 닿음':<24}{'원본이 세게 친 통과(1.0 s 넘게 남음)에서 A 복제 세게'}")
for lo, hi in ((0.0, 0.45), (0.45, 0.8), (0.8, 9.0)):
    S = [e for e in E if lo <= e["wall"] < hi]
    Sh = [e for e in S if e["hard"] and e["lead"] >= 1.0]
    print(f"{lo:.2f}-{hi:.2f}   {len(S):>6}  {pct(sum(e['hard'] for e in S), len(S)):<26}"
          f"{pct(sum(o['hard'] for e in S for o in res(e, 1)), sum(len(res(e, 1)) for e in S)):<26}"
          f"{pct(sum(o['hard'] for e in Sh for o in res(e, 1)), sum(len(res(e, 1)) for e in Sh))} (통과 {len(Sh)})")

print("\n(7) 세게 닿음이 몇 개 상자에 몰려 있는가 (원본 + A 복제의 세게 닿음을 상자별로)")
BX = {}
for e in E:
    d = BX.setdefault(e["bid"], dict(n=0, sh=0, an=0, ah=0, wall=e["wall"]))
    d["n"] += 1; d["sh"] += e["hard"]; d["an"] += len(res(e, 1)); d["ah"] += sum(o["hard"] for o in res(e, 1))
tot = sum(d["sh"] + d["ah"] for d in BX.values())
top = sorted(BX.items(), key=lambda kv: -(kv[1]["sh"] + kv[1]["ah"]))
print(f"  지나간 상자 {len(BX)}개 가운데 한 번이라도 세게 닿은 상자 {sum(d['sh'] + d['ah'] > 0 for d in BX.values())}개 · 세게 닿음 모두 {tot}건")
acc = 0
for (fn_, g_, x_, y_), d in top[:6]:
    acc += d["sh"] + d["ah"]
    print(f"  {fn_.split('_')[-1][:-4]} 묶음 {g_:>2} ({x_:6.2f}, {y_:6.2f}) 벽까지 {d['wall']:.2f} m: 원본 세게 {pct(d['sh'], d['n'])} · A 복제 세게 {pct(d['ah'], d['an'])}"
          f" · 여기까지 누적 {100.0 * acc / max(tot, 1):.0f} %")

print("\n(8) 원본이 세게 친 통과에서 A 복제가 원본과 같은 길로 갔는가 (상자 옆을 지날 때 옆 자리 차이 0.15 m 기준)")
for lo, hi in ((0.5, 1.0), (1.0, 2.9)):
    O = [(o, e) for e in E if e["hard"] and lo <= e["lead"] < hi and e["lat0"] is not None for o in res(e, 1) if o["lat0"] is not None]
    sm = [o for o, e in O if abs(o["lat0"] - e["lat0"]) < 0.15]; df = [o for o, e in O if abs(o["lat0"] - e["lat0"]) >= 0.15]
    print(f"  남은 시간 {lo:.1f}-{hi:.1f} s: 같은 길 {pct(len(sm), len(O))} -> 그 가운데 세게 닿음 {pct(sum(o['hard'] for o in sm), len(sm))}"
          f" | 다른 길로 간 복제의 세게 닿음 {pct(sum(o['hard'] for o in df), len(df))}")

print("\n(9) 상자 옆을 지나는 순간: 복제의 속도 - 원본의 속도 [m/s], 상자와의 옆 간격(가운데끼리) [m] - 1.0 s 넘게 남았을 때 복제한 통과")
src_gap = np.array([abs(e["lat0"]) for e in E if e["crossed"] and e["lead"] >= 1.0])
print(f"  원본: 옆 간격 중앙 {np.median(src_gap):.2f} · 하위 10 % {np.percentile(src_gap, 10):.2f} · 하위 2 % {np.percentile(src_gap, 2):.2f} ({len(src_gap)})")
for a in ARMS1:
    O = [(o, e) for e in E if e["crossed"] and e["lead"] >= 1.0 for o in res(e, a) if o["crossed"] and o["v"] is not None]
    if not O:
        continue
    vs = {id(e): float(e["r"].st[int(e["td"]), e["s"], 3]) for _, e in O}
    dv = np.array([o["v"] - vs[id(e)] for o, e in O]); gp = np.array([abs(o["lat0"]) for o, _ in O])
    print(f"  {ARM[a]:<14} 속도 차 중앙 {np.median(dv):+.2f} · 10 % {np.percentile(dv, 10):+.2f} · 90 % {np.percentile(dv, 90):+.2f} | "
          f"옆 간격 중앙 {np.median(gp):.2f} · 하위 10 % {np.percentile(gp, 10):.2f} · 하위 2 % {np.percentile(gp, 2):.2f} ({len(O)})")

# ---- 그림 1: 남은 시간별 재현 비율 ----
fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6), sharey=True)
base = boot(E, lambda e: (sum(o["hard"] for o in res(e, 1)), len(res(e, 1))))
for ax, kd, ttl in ((axs[0], "세게 닿음", "원본이 세게 닿은 통과"), (axs[1], "안 닿음", "원본이 안 닿은 통과")):
    labs = []
    for i, (lo, hi) in enumerate(BINS):
        S = [e for e in E if lo <= e["lead"] < hi and kind(e) == kd]
        bh = boot(S, lambda e: (sum(o["hard"] for o in res(e, 1)), len(res(e, 1))))
        labs.append(f"{lo:.1f}~{hi:.1f}\n통과 {len(S)}\n복제 {bh[4] if bh else 0}")
        if bh is None:
            continue
        ax.vlines([i], [100 * bh[1]], [100 * bh[2]], color=COL[1], lw=2, zorder=2)
        ax.plot([i], [100 * bh[0]], "o", color=COL[1], ms=8, mec="white", mew=2, zorder=3)
        ax.annotate(f"{100 * bh[0]:.0f} %", (i, 100 * bh[0]), xytext=(9, 3), textcoords="offset points", fontsize=9.5, color="#0b0b0b")
    if base:
        ax.axhline(100 * base[0], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(range(len(BINS))); ax.set_xticklabels(labs, fontsize=8.8)
    ax.set_xlim(-0.5, len(BINS) - 0.5); ax.set_ylim(-4, 104)
    ax.set_title(ttl, fontsize=11, loc="left"); ax.set_xlabel("복제한 순간에 상자까지 남은 시간 [s]")
    ax.grid(axis="y", color="#e6e4dd", lw=0.8); ax.set_axisbelow(True)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
axs[0].set_ylabel("센서 잡음만 다른 복제가 세게 닿은 비율 [%]")
fig.suptitle(f"같은 순간에서 복제한 차가 같은 상자를 세게 치는가 - {E[0]['r'].name if E else ''}, 기록 {len(files)}개\n"
             f"세로 막대 = 95 % 구간(표본 수 = 통과 수) · 점선 = 복제 전체 평균 {100 * base[0] if base else 0:.1f} %", fontsize=10.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(out_prefix + "_lead.png", dpi=130); plt.close(fig)


# ---- 그림 2, 3: 장면 (통과마다 넓은 그림 + 상자 앞 확대) ----
def scenes(S, fn, title):
    if not S:
        return
    S = S[:8]
    nr = int(np.ceil(len(S) / 2))
    fig, axs = plt.subplots(nr, 4, figsize=(17.5, 4.75 * nr), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for i, e in enumerate(S):
        r, s, t0, tend = e["r"], e["s"], e["t0"], e["tend"]
        a0 = max(t0 - 6, 0); a1 = min(tend + 6, r.T)
        for ax, zoom in ((axs[i // 2, 2 * (i % 2)], False), (axs[i // 2, 2 * (i % 2) + 1], True)):
            ax.axis("on")
            r.draw_map(ax, e["k"])
            for q in r.boxes(t0, s):
                ax.fill(q[:, 0], q[:, 1], color=C_BOX, zorder=2)
            ax.plot(r.st[a0:a1 + 1, s, 0], r.st[a0:a1 + 1, s, 1], "-", color=COL[0], lw=2.0, zorder=5)
            for o in e["out"]:
                if o["arm"] == 0:
                    continue
                p = r.st[t0:tend, o["row"], :2]
                ax.plot(p[:, 0], p[:, 1], "-", color=COL[o["arm"]], lw=1.4, zorder=4)
                if not zoom:
                    ax.plot(p[-1, 0], p[-1, 1], "o", color=COL[o["arm"]], ms=4, mec="white", mew=0.8, zorder=6)
            if zoom:                                            # each car where it was closest to the box (or where it hit hard)
                for o in e["out"]:
                    row = o["row"]
                    tt = o["t_hit"] if o["t_hit"] is not None else t0 + int(np.hypot(*(r.st[t0:tend, row, :2] - e["box"]).T).argmin())
                    q = r.body(*r.st[tt, row, :3])
                    ax.fill(q[:, 0], q[:, 1], facecolor="none", edgecolor=C_HIT if o["t_hit"] is not None else COL[o["arm"]],
                            lw=1.8 if o["t_hit"] is not None else 1.0, zorder=7 if o["t_hit"] is not None else 6)
                ax.set_xlim(e["box"][0] - 1.1, e["box"][0] + 1.1); ax.set_ylim(e["box"][1] - 1.1, e["box"][1] + 1.1)
                ax.set_title("상자 앞 확대 (2.2 m 칸): 가장 가까웠던 순간의 차, 빨강 = 세게 닿은 차", fontsize=8.4, loc="left")
            else:
                for o in e["out"]:
                    if o["t_hit"] is not None:
                        p = r.st[o["t_hit"], o["row"], :2]
                        ax.plot(p[0], p[1], "x", color=C_HIT, ms=9, mew=2.2, zorder=8)
                q = r.body(*r.st[t0, s, :3])
                ax.fill(q[:, 0], q[:, 1], facecolor="none", edgecolor=COL[0], lw=1.2, zorder=7)
                cx, cy = 0.5 * (e["box"][0] + r.st[t0, s, 0]), 0.5 * (e["box"][1] + r.st[t0, s, 1])
                h = max(2.6, 0.5 * np.hypot(e["box"][0] - r.st[t0, s, 0], e["box"][1] - r.st[t0, s, 1]) + 1.6)
                ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy - h, cy + h)
                txt = []
                for a in ARMS1:
                    oo = [o for o in e["out"] if o["arm"] == a]
                    if oo:
                        txt.append(f"{ARM[a][0]} 세게 {sum(o['hard'] for o in oo)}/{len(oo)}")
                ax.set_title(f"{e['fn'].split('_')[-1][:-4]} 묶음 {e['g']} · {t0 * r.dt:.0f} s · {e['n']}번째 통과 · 상자까지 {e['lead']:.1f} s · {e['v0']:.1f} m/s\n"
                             f"원본 {kind(e)} | " + " · ".join(txt), fontsize=8.4, loc="left")
    hs = [plt.Line2D([], [], color=COL[a], lw=2, label=ARM[a] + (" (네모 = 복제한 순간의 차)" if a == 0 else "")) for a in ARMS0]
    hs.append(plt.Line2D([], [], color=C_HIT, marker="x", ls="", ms=8, mew=2, label="세게 닿은 순간의 차 자리"))
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, fontsize=9.5)
    fig.suptitle(title, fontsize=12, x=0.01, y=1 - 0.06 / (4.75 * nr), ha="left", va="top")
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.75 / (4.75 * nr), bottom=0.5 / (4.75 * nr), hspace=0.16, wspace=0.04)
    fig.savefig(fn, dpi=100); plt.close(fig)


hardE = sorted([e for e in E if e["hard"]], key=lambda e: -e["lead"])
scenes(hardE, out_prefix + "_scenes_hard.png", "원본이 상자를 세게 친 통과와 그 주기의 복제들 (남은 시간이 긴 것부터)")
flipE = sorted([e for e in E if not e["touch"] and e["lead"] >= 0.5 and any(o["hard"] for o in e["out"] if o["arm"] == 1)], key=lambda e: -e["lead"])
scenes(flipE, out_prefix + "_scenes_clonehit.png", "원본은 안 닿았는데 센서 잡음만 다른 복제(A)가 세게 친 통과 (남은 시간이 긴 것부터)")
memE = sorted([e for e in E if not e["touch"] and e["lead"] >= 1.0 and not any(o["hard"] for o in e["out"] if o["arm"] == 1)
               and any(o["hard"] for o in e["out"] if o["arm"] in (2, 3))], key=lambda e: -e["lead"])
scenes(memE, out_prefix + "_scenes_memhit.png", "원본과 A 복제는 안 닿았는데 기억을 바꾼 복제(B, C)만 세게 친 통과")
print(f"\n그림: {out_prefix}_lead.png · _scenes_hard.png ({len(hardE)}) · _scenes_clonehit.png ({len(flipE)}) · _scenes_memhit.png ({len(memE)})")
