"""상자 통과를 '기억이 도움이 될 수 있는 상황'별로 나눠 봄 (11-12 사용자: 사각지대 상자, 라인을 크게 바꿔야 하는 상자, 급하게 감속·조향하는
통과, 지난 바퀴에 닿은 상자를 다음 바퀴에 또, 상대 차). 전체 평균이 아니라 드문 상황 안에서의 비율과, 같은 차가 같은 상자를 다시 지날 때
달라지나(기억을 쓰면 달라져야 함)를 봄. box_response.py 기록을 읽음 (길을 막은 상자 = 지나가기 0.1~1.2 s 전에 상자 없는 plan이 그 상자
중심 0.40 m 안을 지남). 둘째 인자(drive_record.py 기록)가 있으면 상대 차와 닿은 건에서 상대 차가 스캔 시야 안에 있었나도 봄.
인자: box_response npz [drive_record npz]"""
import os, sys
import numpy as np

D = np.load(sys.argv[1])
st, sid, pose, cnt, rs = D["st"], D["sid"], D["pose"], D["cnt"], D["rs"]
seen = D["seen"] >= 2                       # the box is in the scan: two beams or more end on it
d1, d5, cfhit = D["d1"], D["d5"], D["cfhit"]
dt = float(D["dt"]); T, B, P = sid.shape
W = 30
dprop = np.diff(np.concatenate([np.zeros((1, B, 6), np.float32), cnt], 0), axis=0).clip(min=0.0)
live = sid >= 0
xy = st[:, :, :2].astype(float)
dist = np.where(live, np.linalg.norm(pose[..., :2].astype(float) - xy[:, :, None], axis=3), 1e9)
near_j = dist.argmin(2)
V = np.abs(st[:, :, 3]).astype(float)
jump = np.concatenate([np.zeros((1, B), bool), rs[:-1]], 0)          # the state at t is the first of a new episode
step = np.concatenate([np.zeros((1, B)), np.linalg.norm(np.diff(xy, axis=0), axis=2)], 0); step[jump] = 0.0
ARC = np.cumsum(step, 0)
dy = np.diff(st[:, :, 2].astype(float), axis=0); dy = (dy + np.pi) % (2 * np.pi) - np.pi
ALAT = V * np.concatenate([np.zeros((1, B)), np.abs(dy) / dt], 0); ALAT[jump] = 0.0
IB, IJ, IEP, IN, IT, IWAY, ITOUCH, IHARD, ITAU, IVIN, IVMIN, IDROP, IAL, IPT, ISHIFT, IO1, IBX, IBY = range(18)
rows = []
for b in range(B):
    if not live[:, b].any():
        continue
    start = np.concatenate([[0], np.nonzero(rs[:, b])[0] + 1])
    v = V[:, b]; s = ARC[:, b]
    for j in range(P):
        if not live[:, b, j].any():
            continue
        d = dist[:, b, j]; n_pass = 0; last_ep = -1; t = 1; s_last = -1e9; i_last = -1
        while t < T - 1:
            if d[t] < 1.2:
                lo, hi = max(t - 10, 0), min(t + 10, T - 1)
                if d[t] <= d[lo: hi + 1].min():
                    ep0 = int(start[start <= t].max())
                    if ep0 != last_ep:
                        n_pass = 0; last_ep = ep0; s_last = -1e9; i_last = -1
                    # one visit per lap (11-12): a car that touches a box, backs out and comes again within 15 m of driving is still
                    # on the same visit -- its contacts go to that visit's row, it is not "the next pass"
                    if s[t] - s_last < 15.0:
                        if i_last >= 0:
                            w0, w1 = max(t - 8, 0), min(t + 5, T - 1)
                            mine = near_j[w0: w1 + 1, b] == j
                            rows[i_last][ITOUCH] = max(rows[i_last][ITOUCH], float((dprop[w0: w1 + 1, b, 1] * mine).sum() > 0.5))
                            rows[i_last][IHARD] = max(rows[i_last][IHARD], float((dprop[w0: w1 + 1, b, 4] * mine).sum() > 0.5))
                        s_last = s[t]; t = hi + 1; continue
                    s_last = s[t]; i_last = -1
                    nx = start[start > t]; ep1 = int(nx.min()) if len(nx) else T
                    if t - ep0 >= W:
                        w0, w1 = max(t - 8, 0), min(t + 5, T - 1)
                        mine = near_j[w0: w1 + 1, b] == j
                        touch = float((dprop[w0: w1 + 1, b, 1] * mine).sum() > 0.5); hard = float((dprop[w0: w1 + 1, b, 4] * mine).sum() > 0.5)
                        way = float(cfhit[t - 12: t, b, j].any())
                        sn = seen[t - W: t + 1, b, j]
                        u = W
                        while u > 0 and (sn[u - 1] or (u > 1 and sn[u - 2])):
                            u -= 1
                        hit_k = np.nonzero(dprop[w0: w1 + 1, b, 1] * mine > 0.5)[0]
                        ke = w0 + int(hit_k[0]) if len(hit_k) else t          # braking up to the contact, not the speed lost IN it (11-12:
                        seg = v[ke - 10: ke + 1]                              # that made 43 % of "hurried" passes look like hard hits)
                        drop = float((np.maximum.accumulate(seg) - seg).max())
                        pt = np.nan
                        if s[t] - s[ep0] >= 5.0 and s[ep1 - 1] - s[t] >= 2.0:
                            tt = np.arange(ep0, ep1)
                            pt = float(np.interp(s[t] + 2.0, s[ep0:ep1], tt) - np.interp(s[t] - 5.0, s[ep0:ep1], tt)) * dt
                        x = d1[t - W: t + 1, b]; o1 = np.nan
                        for k in range(max(u, 0), W - 1):
                            if (x[k: k + 3] > 0.02).all():
                                o1 = (k - W) * dt; break
                        rows.append([b, j, ep0, n_pass, t, way, touch, hard, (u - W) * dt, v[t - 20], v[t - 15: min(t + 6, T)].min(), drop,
                                     ALAT[t - 10: min(t + 4, T), b].max(), pt, float(d5[t - 15: t + 1, b].max()), o1,
                                     float(pose[t, b, j, 0]), float(pose[t, b, j, 1])])
                        i_last = len(rows) - 1
                    n_pass += 1; t = hi + 1; continue
            t += 1
R = np.array(rows, float)
if os.environ.get("EDGE_ROWS"):
    np.save(os.environ["EDGE_ROWS"], R)                    # for edge_scenes.py
way = R[:, IWAY] > 0.5; late = -R[:, ITAU]


def wil(k, n):
    if n == 0:
        return "–"
    p = k / n; z = 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{p * 100:.1f} % ({int(k)}/{int(n)}, 95 % {max(c - h, 0) * 100:.1f}~{(c + h) * 100:.1f})"


def med(x):
    x = x[~np.isnan(x)]
    return float(np.median(x)) if len(x) else float("nan")


def row(m, label):
    r = R[m]; n = len(r)
    if n == 0:
        print(f"  {label} | 0건"); return
    print(f"  {label} | {n}건 | 세게 닿음 {wil(r[:, IHARD].sum(), n)} | 닿음 전체 {r[:, ITOUCH].mean() * 100:.1f} % | 2 s 전 속도 {med(r[:, IVIN]):.2f} m/s | "
          f"가장 느린 속도(닿아서 멈춘 것 포함) {med(r[:, IVMIN]):.2f} m/s | 닿기 전 1 s 안에 1 m/s 넘게 감속 {(r[:, IDROP] >= 1.0).mean() * 100:.0f} % | 횡가속도 최대 {med(r[:, IAL]):.1f} m/s2 | "
          f"지나는 시간(5 m 전~2 m 뒤) {med(r[:, IPT]):.2f} s | 차가 실제로 비키기 시작 {med(r[:, IO1]):+.2f} s (끝까지 안 비킴 {np.isnan(r[:, IO1]).mean() * 100:.0f} %) | "
          f"처음 보인 때 {med(r[:, ITAU]):+.1f} s | 두 번째 이후 통과 {(r[:, IN] >= 1).mean() * 100:.0f} %")


def boot(x):
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return f"건수 부족({len(x)}쌍)"
    bs = np.random.default_rng(0).choice(x, (2000, len(x))).mean(1)
    return f"{x.mean():+.2f} (95 % {np.percentile(bs, 2.5):+.2f}~{np.percentile(bs, 97.5):+.2f}, {len(x)}쌍)"


print(f"{D['name']}: 학습 차 {B}대 x {T * dt:.0f} s, 다시 출발 {int(rs.sum())}번 | 상자 1.2 m 안으로 지나간 건 {len(R)} (출발 3 s 안은 뺌) | 길을 막은 상자 {int(way.sum())} "
      f"({way.mean() * 100:.0f} %) | 세게 닿음: 길을 막은 건 {int(R[way, IHARD].sum())}, 아닌 건 {int(R[~way, IHARD].sum())}")
print("\n[1] 길을 막은 상자, 처음 스캔에 보인 때별 (늦게 보임 = 사각지대)")
for lo_, hi_, lab in ((0, 0.5, "0.5 s 안"), (0.5, 0.9, "0.5~0.9 s"), (0.9, 1.5, "0.9~1.5 s"), (1.5, 2.95, "1.5~3 s"), (2.95, 9, "3 s 넘게 전부터")):
    row(way & (late >= lo_) & (late < hi_), lab)
print("\n[2] 길을 막고 늦게(0.9 s 안) 보인 상자: 그 차가 그 상자를 처음 지날 때 / 이미 지난 적 있을 때 (기억을 쓰면 뒤쪽이 나아야 함)")
row(way & (late < 0.9) & (R[:, IN] < 0.5), "처음 지남"); row(way & (late < 0.9) & (R[:, IN] >= 0.5), "두 번째 이후")
print("\n[3] 길을 막은 상자, 그 차가 그 상자를 몇 번째 지나나")
for n_ in range(3):
    row(way & (R[:, IN] == n_), f"{n_ + 1}번째")
row(way & (R[:, IN] >= 3), "4번째 이후")
key = {(int(r[IB]), int(r[IJ]), int(r[IEP]), int(r[IN])): i for i, r in enumerate(R)}
pairs = [(i, key[(k[0], k[1], k[2], k[3] + 1)]) for k, i in key.items() if (k[0], k[1], k[2], k[3] + 1) in key]
A_ = np.array([p[0] for p in pairs]); B_ = np.array([p[1] for p in pairs])
moved = np.linalg.norm(R[A_][:, [IBX, IBY]] - R[B_][:, [IBX, IBY]], axis=1)
print(f"\n[4] 같은 차가 같은 상자를 이어서 두 번 지난 쌍 {len(pairs)}: 앞 통과의 결과별로 다음 통과에서 닿는 비율 (지난 바퀴에 닿은 상자를 또 닿나)")
ph, ptc = R[A_, IHARD] > 0.5, R[A_, ITOUCH] > 0.5
for m, lab in ((ph, "앞에서 세게 닿음"), (ptc & ~ph, "앞에서 약하게 닿음"), (~ptc & way[A_], "앞에서 안 닿음, 길을 막은 상자"), (~ptc & ~way[A_], "앞에서 안 닿음, 길을 안 막은 상자")):
    n = int(m.sum())
    if n == 0:
        print(f"  {lab} | 0쌍"); continue
    print(f"  {lab} | {n}쌍 | 다음에 세게 닿음 {wil(R[B_[m], IHARD].sum(), n)} | 다음에 닿음 전체 {wil(R[B_[m], ITOUCH].sum(), n)} | 다음에도 길을 막음 {way[B_[m]].mean() * 100:.0f} % | "
          f"그 사이 상자가 옮겨진 거리 중앙 {np.median(moved[m]):.2f} m")
print("\n[5] 같은 차·같은 상자, 이어진 두 통과가 모두 길을 막은 경우: 다음 통과 - 앞 통과 (기억을 쓰면 더 일찍 비키고 덜 감속해야 함)")
for m, lab in ((way[A_] & way[B_], "모든 쌍"), (way[A_] & way[B_] & (R[A_, IN] < 0.5), "첫 번째 -> 두 번째"), (way[A_] & way[B_] & (late[B_] < 0.9), "다음 통과에서 늦게(0.9 s 안) 보인 쌍")):
    print(f"  {lab} | 비키기 시작(s, 음수 = 더 일찍) {boot(R[B_[m], IO1] - R[A_[m], IO1])} | 가장 느린 속도(m/s) {boot(R[B_[m], IVMIN] - R[A_[m], IVMIN])} | "
          f"지나는 시간(s) {boot(R[B_[m], IPT] - R[A_[m], IPT])} | 2 s 전 속도(m/s) {boot(R[B_[m], IVIN] - R[A_[m], IVIN])}")
print("\n[6] 길을 막은 상자, 차가 실제로 비키기 시작한 때별 (일찍 라인을 잡으면 얼마나 다른가; 상관일 뿐)")
o = R[:, IO1]
for m, lab in ((o <= -1.0, "1.0 s 넘게 전"), ((o > -1.0) & (o <= -0.5), "1.0~0.5 s 전"), (o > -0.5, "0.5 s 안"), (np.isnan(o), "끝까지 안 비킴")):
    row(way & m, lab)
print("\n[7] 길을 막은 상자, 상자 때문에 plan의 0.5 s 점이 옮겨진 최대 거리별 (라인을 크게 바꿔야 하는 상자)")
sh = R[:, ISHIFT]
for lo_, hi_ in ((0, 0.15), (0.15, 0.4), (0.4, 0.8), (0.8, 99)):
    row(way & (sh >= lo_) & (sh < hi_), f"{lo_:g}~{hi_:g} m")
print("\n[8] 길을 막은 상자, 지나가기(닿았으면 닿기) 전 1 s 안에 1 m/s 넘게 감속했나 (급하게 대응한 통과; 닿으면서 잃은 속도는 뺌)")
row(way & (R[:, IDROP] >= 1.0), "급감속 있음"); row(way & (R[:, IDROP] < 1.0), "없음")
if len(sys.argv) > 2:
    E = np.load(sys.argv[2]); s2, op, on, c2 = E["st"].astype(float), E["op"].astype(float), E["on"], E["cnt"]
    T2, B2 = c2.shape[:2]
    dc = np.diff(np.concatenate([np.zeros((1, B2, 6)), c2], 0), axis=0).clip(min=0.0)
    fov = float(D["fov"]); fov = np.deg2rad(fov) if fov > 7 else fov
    rel = op[:T2, :, :2] - s2[:T2, :, :2]
    bear = np.arctan2(rel[..., 1], rel[..., 0]) - s2[:T2, :, 2]; bear = np.abs((bear + np.pi) % (2 * np.pi) - np.pi)
    gap = np.linalg.norm(rel, axis=2); outv = bear > fov / 2
    near = on[:T2] & (gap < 3.0)
    print(f"\n[9] 상대 차 ({E['name']}, 학습 차 {B2}대 x {T2 * dt:.0f} s): 스캔 시야 {np.rad2deg(fov):.0f} deg | 상대 차가 3 m 안에 있는 시간 중 시야 밖(뒤쪽) {outv[near].mean() * 100:.0f} %")
    for col, lab in ((5, "세게 닿음"), (2, "닿음 전체")):
        ev = np.argwhere(dc[:, :, col] > 0.5); ev = ev[ev[:, 0] >= 10]
        if len(ev) == 0:
            continue
        o0 = np.array([outv[t, b] for t, b in ev]); o5 = np.array([outv[t - 5, b] for t, b in ev]); oa = np.array([outv[t - 10: t + 1, b].any() for t, b in ev])
        b5 = np.rad2deg(np.array([bear[t - 5, b] for t, b in ev]))
        print(f"  {lab} {len(ev)}건 | 닿는 순간 상대 차가 시야 밖 {o0.mean() * 100:.0f} % | 0.5 s 전 시야 밖 {o5.mean() * 100:.0f} % | 지난 1 s 중 한 번이라도 시야 밖 {oa.mean() * 100:.0f} % | "
              f"0.5 s 전 상대 차 방향: 앞(60 deg 안) {(b5 < 60).mean() * 100:.0f} %, 옆(60~{np.rad2deg(fov) / 2:.0f} deg) {((b5 >= 60) & (b5 <= np.rad2deg(fov) / 2)).mean() * 100:.0f} %, 뒤 {(b5 > np.rad2deg(fov) / 2).mean() * 100:.0f} %")
