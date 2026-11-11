"""시험 기록(exam_record.py npz)의 상자 통과를 상황별로 나눠 봄: 전체 평균이 아니라 드문 상황 안에서의 비율.
'길 위의 상자' = 같은 시험의 혼자·빈 트랙 세션들에서 이 policy가 실제로 달린 길(세션마다 가장 가까이 지난 거리의 중앙값)이 상자 중심
0.40 m 안을 지나는 상자. 통과 한 번 = 상자에 가장 가까이 간 순간(같은 상자를 15 m 달리기 전에 다시 건드리면 같은 통과).
통과마다: 길에서 상자까지 거리, 상자 중심이 스캔에 직접 보이기 시작한 때(벽에 안 가리고 시야각·거리 안; 상자 한가운데만 봄),
그 세션에서 그 상자를 몇 번째 지나나, 지난번에 세게 닿았나, 닿기 전 1 s 동안 속도를 얼마나 줄였나, 실제로 길에서 얼마나 비켰나, 닿았나.
표: (가) 길에서의 거리별 (나) 보이기 시작한 때별 (다) 몇 번째 통과인지별 + 같은 상자 첫째·둘째 통과 짝 (라) 지난번에 세게 닿은 뒤
(마) 급하게 줄인 통과 (바) 혼자 / 상대 차 있을 때. 비율 옆 [ ] = 95 % 구간(Wilson).
인자: 기록npz   (EXAM_ROWS=파일 이면 통과별 행을 npy로 저장)"""
import os, sys
import numpy as np
from rec_lib import Rec

R = Rec(sys.argv[1]); z = R.z; T, B, dt = R.T, R.B, R.dt
rm_t = float(z["rm_t"]); k_rm = min(int(round(rm_t / dt)), T)
st = R.st.astype(float); xy = st[:, :, :2]; yaw = st[:, :, 2]; V = np.abs(st[:, :, 3])
sid, pose = R.sid, R.pose.astype(float)
box = (sid[0] >= 0).any(-1); opp = R.on[:T].any(0)
mount = float(z["lidar_mount_x"]) if "lidar_mount_x" in z.files and float(z["lidar_mount_x"]) > 0 else 0.297
FOV, RMAX = np.radians(135.0), 10.0
W = int(round(3.0 / dt)); n10 = int(round(1.0 / dt))
MODE = dict(map="MAP", track="시간 추종기", ideal="plan대로 정확히")[str(z["mode"])]


def wilson(k, n):
    if n == 0:
        return "    -"
    p = k / n; zz = 1.96; den = 1 + zz * zz / n
    c = (p + zz * zz / (2 * n)) / den; h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)) / den
    return f"{100 * p:5.1f} % [{100 * max(c - h, 0):4.1f}-{100 * min(c + h, 1):4.1f}]"


# the path this policy drives without boxes: solo sessions on the empty track, normal driving only (after the first 5 s,
# faster than 2 m/s, not within 3 s after a hard contact)
hard_any = R.d[:, :, 3:6].sum(-1) > 0
ref = np.nonzero(~box & ~opp)[0]
SA, SB, SO = [], [], [0]
for b in ref:
    ok = V[:T, b] > 2.0; ok[:int(round(5.0 / dt))] = False
    for t in np.nonzero(hard_any[:, b])[0]:
        ok[t: t + int(round(3.0 / dt))] = False
    k = np.nonzero(ok[:-1] & ok[1:])[0]
    SA.append(xy[k, b]); SB.append(xy[k + 1, b]); SO.append(SO[-1] + len(k))
keep = [i for i in range(len(ref)) if SO[i + 1] - SO[i] > 100]
SA = [SA[i] for i in keep]; SB = [SB[i] for i in keep]
off = np.concatenate([[0], np.cumsum([len(a) for a in SA])])[:-1]
SA, SB = np.concatenate(SA), np.concatenate(SB); SD = SB - SA; SL = (SD ** 2).sum(1).clip(min=1e-9)
n_ref = len(keep)


def to_free_path(pts):
    """[m] from each point to the path driven without boxes: per reference session the nearest distance, then the median"""
    out = np.zeros(len(pts))
    for i in range(0, len(pts), 32):
        p = pts[i: i + 32, None, :]
        u = (((p - SA[None]) * SD[None]).sum(2) / SL[None]).clip(0.0, 1.0)
        d = np.linalg.norm(p - (SA[None] + u[..., None] * SD[None]), axis=2)             # (n, segments)
        out[i: i + 32] = np.median(np.minimum.reduceat(d, off, axis=1), axis=1)
    return out


def sees(tk, t, b, c):
    """is the box centre c in the scan's direct view from the scanner at decision t: inside the field of view and the range,
    no wall on the way (up to 0.25 m short of the centre)"""
    sx, sy = xy[t, b, 0] + mount * np.cos(yaw[t, b]), xy[t, b, 1] + mount * np.sin(yaw[t, b])
    dx, dy = c[0] - sx, c[1] - sy; dist = np.hypot(dx, dy)
    a = (np.arctan2(dy, dx) - yaw[t, b] + np.pi) % (2 * np.pi) - np.pi
    if dist > RMAX or abs(a) > FOV:
        return False
    n = max(int((dist - 0.25) / 0.05), 1)
    f = np.linspace(0.0, max(dist - 0.25, 0.0) / max(dist, 1e-6), n)
    return bool((R.wall_dist(tk, np.stack([sx + f * dx, sy + f * dy], 1)) > 0.0).all())


step = np.concatenate([np.zeros((1, B)), np.linalg.norm(np.diff(xy, axis=0), axis=2)], 0)
ARC = np.cumsum(step, 0)
live = sid >= 0
P = sid.shape[2]
dist_all = np.where(live, np.linalg.norm(pose[..., :2] - xy[:, :, None], axis=3), 1e9)
near_j = dist_all.argmin(2)
IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
rows = []
for b in np.nonzero(box)[0]:
    tk = int(R.tid[0, b]); s = ARC[:, b]; v = V[:, b]
    for j in range(P):
        if not live[:k_rm, b, j].any():
            continue
        d = dist_all[:k_rm, b, j]; n_pass = 0; s_last = -1e9; i_last = -1; prev_hard = -1.0; t = 1
        while t < k_rm - 1:
            if d[t] < 1.5:
                lo, hi = max(t - 10, 0), min(t + 10, k_rm - 1)
                if d[t] <= d[lo: hi + 1].min():
                    w0, w1 = max(t - 8, 0), min(t + 5, T - 1)
                    mine = near_j[w0: w1 + 1, b] == j
                    touch = float((R.d[w0: w1 + 1, b, 1] * mine).sum() > 0.5); hard = float((R.d[w0: w1 + 1, b, 4] * mine).sum() > 0.5)
                    if s[t] - s_last < 15.0:                     # still the same visit (touched, backed out, came again)
                        if i_last >= 0:
                            rows[i_last][ITOUCH] = max(rows[i_last][ITOUCH], touch); rows[i_last][IHARD] = max(rows[i_last][IHARD], hard)
                            prev_hard = rows[i_last][IHARD]
                        s_last = s[t]; t = hi + 1; continue
                    s_last = s[t]; i_last = -1; n_pass += 1
                    if t >= W:
                        c = pose[t - W, b, j, :2]
                        vis = [sees(tk, k, b, pose[k, b, j, :2]) for k in range(t - W, t + 1)]
                        k0 = W                                    # index of the first decision of the visible stretch that ends at t
                        while k0 > 0 and vis[k0 - 1]:
                            k0 -= 1
                        if not any(vis[-5:]):
                            see, see_d = 0.0, 0.0
                        else:
                            if not vis[W]:
                                k0 = W - 1
                                while k0 > 0 and not vis[k0]:
                                    k0 -= 1
                                while k0 > 0 and vis[k0 - 1]:
                                    k0 -= 1
                            see = (W - k0) * dt; see_d = float(np.linalg.norm(pose[t - W + k0, b, j, :2] - xy[t - W + k0, b]))
                        a0 = np.searchsorted(s[: t + 1], s[t] - 4.0); a1 = min(np.searchsorted(s[: T + 1], s[t] + 2.0), T)
                        t_pass = (a1 - a0) * dt if (s[t] - s[a0] > 3.5 and s[a1] - s[t] > 1.5) else np.nan
                        hit_k = np.nonzero(R.d[w0: w1 + 1, b, 1] * mine > 0.5)[0]
                        ke = max(w0 + int(hit_k[0]), n10) if len(hit_k) else t      # speed up to the contact, not through it
                        rows.append([b, j, t, n_pass, 0.0, see, see_d, touch, hard, prev_hard, v[ke - n10], v[ke - n10] - v[ke - n10: ke + 1].min(),
                                     0.0, t_pass, float(opp[b]), c[0], c[1]])
                        i_last = len(rows) - 1
                    prev_hard = hard
                    t = hi + 1; continue
            t += 1
rows = np.array(rows, float).reshape(-1, 17)
if len(rows) == 0 or n_ref == 0:
    print("상자 통과 없음 또는 빈 트랙 세션 없음"); sys.exit(0)
rows[:, IGAP] = to_free_path(rows[:, [IBX, IBY]])
rows[:, ISHIFT] = to_free_path(xy[rows[:, IT].astype(int), rows[:, IB].astype(int)])
if os.environ.get("EXAM_ROWS"):
    np.save(os.environ["EXAM_ROWS"], rows)

way = rows[:, IGAP] < 0.40
hd, tc = rows[:, IHARD] > 0, rows[:, ITOUCH] > 0
print(f"{z['exam']} · {MODE} · {R.name} · 상자 {z['count']} · 상자 세션 {int(box.sum())}개(상자는 {rm_t:g} s에 치움) · 기준 길 = 혼자·빈 트랙 세션 {n_ref}개")
print(f"상자 통과 {len(rows)}번 (세션 시작 3 s 안은 뺌) | 길 위의 상자(달리던 길이 상자 중심 0.40 m 안) {int(way.sum())}번 = {100 * way.mean():.0f} % | "
      f"세게 닿음 {int(hd.sum())}번 가운데 길 위 {int((hd & way).sum())}번")


def line(name, m):
    n = int(m.sum())
    if n == 0:
        print(f"  {name:<26} {0:5d}번"); return
    r = rows[m]
    print(f"  {name:<26} {n:5d}번 | 세게 닿음 {wilson(int((r[:, IHARD] > 0).sum()), n)} | 닿음(약한 것 포함) {wilson(int((r[:, ITOUCH] > 0).sum()), n)} | "
          f"실제로 비킨 폭(중앙) {np.median(r[:, ISHIFT]):.2f} m | 1 s 전 속도 {np.median(r[:, IVIN]):.1f} m/s | 통과 시간(앞 4 m~뒤 2 m) "
          f"{np.nanmedian(r[:, IPASS]) if np.isfinite(r[:, IPASS]).any() else float('nan'):.2f} s")


print("\n(가) 달리던 길에서 상자 중심까지의 거리별")
for name, lo, hi in (("길 한가운데 (0.15 m 안)", 0.0, 0.15), ("길에 걸침 (0.15-0.40 m)", 0.15, 0.40), ("길 옆 (0.40-0.80 m)", 0.40, 0.80), ("멀리 (0.80 m 밖)", 0.80, 1e9)):
    line(name, (rows[:, IGAP] >= lo) & (rows[:, IGAP] < hi))
print("\n(나) 길 위의 상자: 상자 중심이 스캔에 보이기 시작한 때별 (지나가기 몇 s 전)")
for name, lo, hi in (("0.5 s 안 (거의 안 보임)", -1.0, 0.5), ("0.5-0.9 s 전", 0.5, 0.9), ("0.9-1.5 s 전", 0.9, 1.5), ("1.5-3.0 s 전", 1.5, 2.95), ("3 s 넘게 전부터", 2.95, 1e9)):
    line(name, way & (rows[:, ISEE] >= lo) & (rows[:, ISEE] < hi))
print("\n(다) 길 위의 상자: 그 세션에서 그 상자를 몇 번째 지나나 (기억을 쓰면 뒤로 갈수록 덜 닿아야 함)")
for name, lo, hi in (("첫째", 1, 1), ("둘째", 2, 2), ("셋째", 3, 3), ("넷째 이후", 4, 99)):
    line(name, way & (rows[:, IN] >= lo) & (rows[:, IN] <= hi))
key = {(int(r[IB]), int(r[IJ]), int(r[IN])): i for i, r in enumerate(rows)}
pair = [(i, key[(int(r[IB]), int(r[IJ]), 2)]) for i, r in enumerate(rows) if r[IN] == 1 and way[i] and (int(r[IB]), int(r[IJ]), 2) in key]
if pair:
    a, b_ = np.array(pair).T
    f = lambda col: (np.nanmedian(rows[a, col]), np.nanmedian(rows[b_, col]), np.nanmedian(rows[b_, col] - rows[a, col]))
    print(f"  같은 상자 첫째 -> 둘째 짝 {len(pair)}개: 세게 닿음 {100 * hd[a].mean():.1f} -> {100 * hd[b_].mean():.1f} % · 닿음 {100 * tc[a].mean():.1f} -> "
          f"{100 * tc[b_].mean():.1f} % · 1 s 전 속도 {f(IVIN)[0]:.2f} -> {f(IVIN)[1]:.2f} m/s (짝 차이 중앙 {f(IVIN)[2]:+.2f}) · 통과 시간 "
          f"{f(IPASS)[0]:.2f} -> {f(IPASS)[1]:.2f} s (짝 차이 중앙 {f(IPASS)[2]:+.2f}) · 비킨 폭 {f(ISHIFT)[0]:.2f} -> {f(ISHIFT)[1]:.2f} m")
print("\n(라) 길 위의 상자, 둘째 통과부터: 지난번 통과에서 세게 닿았나")
line("지난번에 세게 닿음", way & (rows[:, IN] >= 2) & (rows[:, IPREV] > 0))
line("지난번에 안 닿거나 약하게", way & (rows[:, IN] >= 2) & (rows[:, IPREV] == 0))
print("\n(마) 길 위의 상자: 닿기 직전까지(안 닿았으면 가장 가까운 순간까지) 1 s 동안 속도를 얼마나 줄였나")
for name, lo, hi in (("거의 안 줄임 (0.5 m/s 안)", -1e9, 0.5), ("0.5-1.5 m/s", 0.5, 1.5), ("1.5 m/s 넘게 (급하게)", 1.5, 1e9)):
    line(name, way & (rows[:, IDROP] >= lo) & (rows[:, IDROP] < hi))
print("\n(바) 길 위의 상자: 혼자 / 상대 차 있는 세션")
line("혼자", way & (rows[:, IOPP] == 0)); line("상대 차 있음", way & (rows[:, IOPP] > 0))
