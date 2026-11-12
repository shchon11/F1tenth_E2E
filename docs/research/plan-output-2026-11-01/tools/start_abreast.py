"""나란히 출발한 세션에서 출발 직후 차끼리 닿는가, 누가 옆으로 좁혔는가 (start_abreast.py, 11-12).
대회 규칙(RoboRacer 2025)의 1대1 출발 = 같은 출발선에 30 cm 떨어져 나란히 -> 학습·시험의 '나란히' 배치가 바로 그 자리.
시험 기록(exam_record.py; F1SIM_EXAM_START_REST / F1SIM_EXAM_OPP_SLOT '{"spawn": "alongside"}' 포함)을 읽음.
표: 기록마다 나란히 출발 수 · 2 s / 3 s 안에 차 접촉(세게) · 닿은 출발에서 닿기까지 두 차가 옆으로 서로에게 다가간 거리
    (출발 때 우리 차 방향 기준; 굽은 길에서는 길의 굽음이 섞이므로 1 s 안에 닿은 것만 셈) · 상대 종류별.
인자: 기록npz [기록npz ...]"""
import os, sys, collections
import numpy as np

W2, W3 = 2.0, 3.0                                                     # [s] windows after the start
SHORT = 1.0                                                           # [s] "who closed" only for contacts this early


def abreast(z):
    st = z["st"].astype(float); op = z["op"].astype(float); on = z["on"].astype(bool); c = z["cnt"].astype(float)
    dt = 0.1
    yaw = st[0, :, 2]; cy, sy = np.cos(yaw), np.sin(yaw)

    def lat(p):                                                       # sideways in our car's start frame (left +)
        return -(p[..., 0] - st[0, :, 0]) * sy + (p[..., 1] - st[0, :, 1]) * cy

    al0 = (op[0, :, 0] - st[0, :, 0]) * cy + (op[0, :, 1] - st[0, :, 1]) * sy
    la0 = lat(op[0])
    rows = np.where(on[0] & (np.abs(al0) < 0.6) & (np.abs(la0) < 1.2))[0]
    kinds = [str(k) for k in z["kinds"]]
    out = []
    k3 = int(round(W3 / dt))
    for b in rows:
        d = c[1:k3 + 1, b, 2] - c[0, b, 2]
        hit = np.where(d > 0)[0]
        tc = (int(hit[0]) + 1) * dt if len(hit) else None
        dh = c[1:k3 + 1, b, 5] - c[0, b, 5]
        hh = np.where(dh > 0)[0]
        th = (int(hh[0]) + 1) * dt if len(hh) else None
        k = int(round((tc if tc else SHORT) / dt))
        sg = np.sign(la0[b])
        out.append(dict(b=int(b), tc=tc, th=th, me=float(lat(st[k])[b] * sg), op=float(-(lat(op[k])[b] - la0[b]) * sg),
                        v0=float(st[0, b, 3]), ov0=float(op[0, b, 3]), kind=kinds[int(z["kind"][0, b])],
                        scale=float(z["scale"][0, b]), gap=float(abs(la0[b]))))
    return out


for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=True)
    rr = abreast(z)
    n = len(rr)
    name = os.path.basename(fn).replace(".npz", "")
    rest = bool(z["start_rest"]) if "start_rest" in z.files else False
    print(f"== {name}: 나란히 출발 {n} · 출발 속도 {'둘 다 0' if rest else '차마다 0~3 m/s'} · 두 차 중심 옆 거리 중앙 "
          f"{np.median([r['gap'] for r in rr]) if n else float('nan'):.2f} m ==")
    if not n:
        continue
    a2 = [r for r in rr if r["tc"] is not None and r["tc"] <= W2]
    a3 = [r for r in rr if r["tc"] is not None]
    h2 = [r for r in rr if r["th"] is not None and r["th"] <= W2]
    h3 = [r for r in rr if r["th"] is not None]
    print(f"  차 접촉: 2 s 안 {len(a2)} ({100 * len(a2) / n:.0f} %) · 그중 세게 {len(h2)} ({100 * len(h2) / n:.0f} %) | "
          f"3 s 안 {len(a3)} ({100 * len(a3) / n:.0f} %) · 세게 {len(h3)} ({100 * len(h3) / n:.0f} %)")
    early = [r for r in a3 if r["tc"] <= SHORT]
    if early:
        opp = sum(r["op"] > 0.15 and r["me"] < 0.15 for r in early)
        me = sum(r["me"] > 0.15 and r["op"] < 0.15 for r in early)
        print(f"  {SHORT:g} s 안에 닿은 {len(early)}건에서 0.15 m 넘게 옆으로 좁힌 쪽: 상대 차만 {opp} · 우리 차만 {me} · 둘 다/그 밖 {len(early) - opp - me} "
              f"| 좁힌 거리 중앙값 우리 {np.median([r['me'] for r in early]):+.2f} m · 상대 {np.median([r['op'] for r in early]):+.2f} m")
    kc = collections.Counter(r["kind"] for r in rr); kh = collections.Counter(r["kind"] for r in a3)
    print("  상대 종류별 3 s 안 접촉/출발: " + " · ".join(f"{k} {kh.get(k, 0)}/{v}" for k, v in sorted(kc.items())))
    for r in a3[:10]:
        print(f"    차 {r['b']:3d}: {r['tc']:.1f} s{' (세게 ' + format(r['th'], '.1f') + ' s)' if r['th'] is not None else ''} · 좁힌 거리 우리 "
              f"{r['me']:+.2f} / 상대 {r['op']:+.2f} m · 출발 속도 우리 {r['v0']:.1f} / 상대 {r['ov0']:.1f} m/s · {r['kind']} x{r['scale']:.2f}")
