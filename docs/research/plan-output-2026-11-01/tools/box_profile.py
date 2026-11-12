"""상자 통과 전후의 움직임을 시간축으로 겹쳐 봄 (11-12, 사용자: "희안하네. 보고도 미리 안 피하다니").
box_early_late.py의 '비키기 시작'은 평소 길에서 0.10 m 넘게 벗어난 때라 0.3 m쯤 비키는 통과에서는 움직임의 뒷부분만 잡고, 감속으로
먼저 반응했는지도 안 보임. 여기서는 혼자 달리는 상자 세션의 길 위 상자 통과(닿지 않은 것)마다, 가장 가까워지기 3 s 전부터 0.6 s 뒤까지
0.1 s마다:
 - 옆으로 벗어난 거리: 같은 세션에서 상자를 치운 뒤 달린 길 기준, 상자에서 멀어지는 쪽이 +
 - 속도 차: 치운 뒤 같은 자리에서의 속도 대비
 - plan이 상자 옆에 둔 거리: 그때 낸 plan의 길이 상자 중심에서 떨어진 거리 (plan이 상자를 지나칠 만큼 길 때만)
옆으로 움직이기 시작한 때 = 벗어난 거리가 마지막으로 (최종 비킨 거리의 10 %와 치운 뒤 바퀴들끼리의 차이 90 % 값 중 큰 쪽) 아래였던 때.
감속 시작 = 평소보다 0.3 m/s 넘게 느려지기 시작한 때. 잃은 시간 = 상자 8 m 전 ~ 3 m 뒤 구간, 치운 뒤 같은 구간 대비.
자리 맞춤은 보상이 쓰는 벽 기준 거리장(분석에만 씀).
인자: 기록npz 통과행npy 그림png [장면png]"""
import sys
import numpy as np
from rec_lib import Rec, plt, along_track, C_BOX

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
PRE, POST, BEFORE, AFTER = 30, 6, 8.0, 3.0
R = Rec(sys.argv[1]); z = R.z; T, B, dt = R.T, R.B, R.dt
rows = np.load(sys.argv[2]); fig_out = sys.argv[3]; scene_out = sys.argv[4] if len(sys.argv) > 4 else ""
st = R.st.astype(float); xy = st[:, :, :2]; V = np.abs(st[:, :, 3]); pl = R.pl.astype(float)
k_rm = min(int(round(float(z["rm_t"]) / dt)), T)
hard_any = R.d[:, :, 3:6].sum(-1) > 0
opp = R.on[:T].any(0); pose = R.pose.astype(float)
s, lap, L = along_track(z, T); u = lap * L + s
alat = V[:T] * np.diff(st[:, :, 2], axis=0) / dt                # lateral acceleration [m/s^2]
jerk = np.diff(alat, axis=0) / dt                               # its change [m/s^3]
START = 70                                                      # passes in the first 7 s (standing start) are left out
MODE = dict(map="MAP", track="시간 추종기", ideal="plan대로 정확히")[str(z["mode"])]


def ref_of(b, kk):
    """the session's usual path from the decisions kk (boxes gone): segment starts, vectors, squared lengths"""
    kk = kk[(V[kk, b] > 2.0) & (V[kk + 1, b] > 2.0) & ~R.rs[kk, b]]
    A = xy[kk, b]; D = xy[kk + 1, b] - A; LL = (D ** 2).sum(1)
    ok = LL < 1.5 ** 2
    return kk[ok], A[ok], D[ok], LL[ok].clip(min=1e-9)


def offset(p, ref, b):
    """the points' signed distance from the usual path (left +) and the usual speed at that place"""
    kk, A, D, LL = ref
    w = (((p[:, None] - A[None]) * D[None]).sum(2) / LL[None]).clip(0.0, 1.0)
    q = A[None] + w[..., None] * D[None]
    d = np.linalg.norm(p[:, None] - q, axis=2); i = d.argmin(1); n = np.arange(len(p))
    r = p - q[n, i]; side = D[i, 0] * r[:, 1] - D[i, 1] * r[:, 0]
    return np.sign(side) * d[n, i], V[kk[i], b] * (1 - w[n, i]) + V[kk[i] + 1, b] * w[n, i]


def cross(ub, c, k_lo, k_hi):
    k = np.nonzero(ub[k_lo: k_hi + 1] >= c)[0]
    if len(k) == 0 or k[0] == 0:
        return np.nan
    k = k_lo + int(k[0]); d = ub[k] - ub[k - 1]
    return k - 1 + (c - ub[k - 1]) / d if d > 1e-6 else float(k)


def through(b, c, k_lo, k_hi):
    """[s] through the places [c - BEFORE, c + AFTER], or nan"""
    a, e = cross(u[:, b], c - BEFORE, k_lo, k_hi), cross(u[:, b], c + AFTER, k_lo, k_hi)
    if not (np.isfinite(a) and np.isfinite(e)) or e <= a:
        return np.nan
    i0, i1 = int(np.floor(a)), int(np.ceil(e))
    return (e - a) * dt if not hard_any[i0: i1 + 1, b].any() and V[i0: i1 + 1, b].min() >= 1.0 else np.nan


def seg(b, c0, c1, k_lo, k_hi):
    """over the places [c0, c1]: time [s], the summed squared lateral acceleration and its summed squared change (x dt), or None"""
    a, e = cross(u[:, b], c0, k_lo, k_hi), cross(u[:, b], c1, k_lo, k_hi)
    if not (np.isfinite(a) and np.isfinite(e)) or e <= a:
        return None
    i0, i1 = int(np.floor(a)), int(np.ceil(e))
    if hard_any[i0: i1 + 1, b].any() or V[i0: i1 + 1, b].min() < 1.0 or R.rs[i0: i1 + 1, b].any():
        return None
    return np.array([(e - a) * dt, float((alat[i0: i1, b] ** 2).sum() * dt), float((jerk[i0: max(i1 - 1, i0 + 1), b] ** 2).sum() * dt)])


def extra(b, t, c0, c1):
    """the pass's (time, lateral sums) over the places [c0, c1] around it minus/over the same places once the boxes are gone"""
    w_ = seg(b, u[t, b] + c0, u[t, b] + c1, max(t - 90, 0), k_rm - 1)
    rf = [x for x in (seg(b, u[t, b] + m * L + c0, u[t, b] + m * L + c1, k_rm + 30, T - 2) for m in range(1, 9)) if x is not None]
    if w_ is None or len(rf) < 2:
        return np.full(3, np.nan), np.full(3, np.nan)
    return w_, np.median(np.array(rf), 0)


def plan_gap(k, b, c):
    """how far the plan made at decision k passes from the point c, nan when the plan ends before it"""
    P = np.concatenate([xy[k, b][None], pl[k, b]], 0); A = P[:-1]; D = P[1:] - A; LL = (D ** 2).sum(1).clip(min=1e-9)
    w = ((c[None] - A) * D).sum(1) / LL
    d = np.linalg.norm(c[None] - (A + w.clip(0.0, 1.0)[:, None] * D), axis=1); i = int(d.argmin())
    return np.nan if (i == len(A) - 1 and w[i] >= 1.0) else float(d[i])


# how much two laps without boxes differ from each other: the first lap after the boxes are gone against the later ones
refs, noise = {}, []
for b in range(B):
    if opp[b] or not (R.sid[0, b] >= 0).any():
        continue
    refs[b] = ref_of(b, np.arange(k_rm + 50, T - 1))
    late = ref_of(b, np.arange(k_rm + 130, T - 1))
    if len(late[0]) > 80:
        noise.append(np.abs(offset(xy[k_rm + 50: k_rm + 110, b], late, b)[0]))
noise = np.concatenate(noise); NOISE = float(np.percentile(noise, 90))
print(f"{z['exam']} · {MODE} · {R.name} · 상자 {z['count']}")
print(f"상자 없는 바퀴끼리 길의 차이: 중앙 {100 * np.median(noise):.1f} cm, 90 % {100 * NOISE:.1f} cm (이보다 작은 벗어남은 움직임으로 안 봄)")

W = PRE + POST + 1
prof, dvel, gap, meta, keep, more = [], [], [], [], [], []
skipped = 0
for r in rows:
    b, j, t = int(r[IB]), int(r[IJ]), int(r[IT])
    if opp[b] or r[IGAP] >= 0.40 or b not in refs or len(refs[b][0]) < 100:
        continue
    if r[ITOUCH] > 0.5 or t < START or t + POST >= k_rm:
        skipped += 1; continue
    w = np.arange(t - PRE, t + POST + 1)
    if hard_any[w, b].any() or R.rs[w, b].any() or V[w, b].min() < 1.0:
        skipped += 1; continue
    off, vref = offset(xy[w, b], refs[b], b)
    c = pose[t, b, j, :2]; boff = float(offset(c[None], refs[b], b)[0][0])
    sgn = 1.0 if off[PRE] >= boff else -1.0
    p = sgn * off; dv = V[w, b] - vref
    i_max = PRE - 3 + int(p[PRE - 3: PRE + 4].argmax()); fin = float(p[i_max])
    thr = max(0.10 * fin, NOISE)
    k0 = i_max
    while k0 > 0 and p[k0 - 1] > thr:
        k0 -= 1
    kh = i_max
    while kh > 0 and p[kh - 1] > 0.5 * fin:
        kh -= 1
    k9 = k0 + int(np.argmax(p[k0: i_max + 1] >= 0.9 * fin))
    t_move = (PRE - k0) * dt if fin > thr else np.nan            # s before the closest approach the sideways move began
    dmax = float(-dv[: PRE + 1].min()); iv = int(dv[: PRE + 1].argmin())
    while iv > 0 and -dv[iv - 1] > 0.3:
        iv -= 1
    t_brake = (PRE - iv) * dt if dmax > 0.3 else np.nan
    g = np.array([plan_gap(k, b, c) for k in w[: PRE + 1]] + [np.nan] * POST)
    ref_t = np.array([through(b, u[t, b] + m * L, k_rm + 30, T - 2) for m in range(1, 9)]); ref_t = ref_t[np.isfinite(ref_t)]
    loss = through(b, u[t, b], max(t - 80, 0), min(t + 60, k_rm - 1)) - np.median(ref_t) if len(ref_t) >= 2 else np.nan
    parts = [extra(b, t, c0, c1) for c0, c1 in ((-8.0, 0.0), (0.0, 3.0), (3.0, 10.0), (-8.0, 6.0))]
    d_move = float(np.linalg.norm(xy[t - PRE + k0, b] - c)) if fin > thr else np.nan
    prev = rows[(rows[:, IB] == b) & (rows[:, IJ] != j) & (rows[:, IT] < t)]       # the last other box passed before this one
    du = float(u[t, b] - u[int(prev[:, IT].max()), b]) if len(prev) else np.inf
    prof.append(p); dvel.append(dv); gap.append(g); keep.append((b, j, t, sgn))
    more.append([parts[0][0][0] - parts[0][1][0], parts[1][0][0] - parts[1][1][0], parts[2][0][0] - parts[2][1][0],
                 parts[3][0][1], parts[3][1][1], parts[3][0][2], parts[3][1][2], d_move, du])
    meta.append([fin, r[ISEE], t_move, (PRE - kh) * dt, (k9 - k0) * dt, dmax, t_brake, loss, r[IN], abs(boff), float(np.linalg.norm(xy[t, b] - c)),
                 r[IVIN], float(k0 == 0)])
P, DV, G, M, X = np.array(prof), np.array(dvel), np.array(gap), np.array(meta), np.array(more)
XB, XN, XA, XIA, XIA0, XIJ, XIJ0, XDM, XDU = range(9)
MF, MSEE, MMOVE, MHALF, MDUR, MDMAX, MBRAKE, MLOSS, MN, MBOFF, MCLEAR, MV, MCENS = range(13)
print(f"혼자 달리는 상자 세션의 길 위 상자 통과 가운데 잰 것 {len(M)}번 (닿았거나 창이 모자라 뺀 것 {skipped}번)")
print(f"상자 중심이 이 세션의 평소 길에서 떨어진 거리: 중앙 {np.median(M[:, MBOFF]):.2f} m · 지날 때 차 중심과 상자 중심 거리: 중앙 {np.median(M[:, MCLEAR]):.2f} m")


def q(x, fmt="{:.1f}"):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "-"
    return (fmt + " [" + fmt + "~" + fmt + "]").format(*np.percentile(x, [50, 25, 75]))


GROUPS = (("비킨 거리 0.15 m 안", M[:, MF] < 0.15), ("0.15~0.30 m", (M[:, MF] >= 0.15) & (M[:, MF] < 0.30)),
          ("0.30~0.60 m", (M[:, MF] >= 0.30) & (M[:, MF] < 0.60)), ("0.60 m 넘게", M[:, MF] >= 0.60))
print("(가) 실제 비킨 거리별: 언제 보이고, 언제 옆으로 움직이기 시작하고, 언제 감속하나  (중앙 [25~75 %], 가장 가까워지기 몇 s 전)")
for name, m in GROUPS:
    k = M[m]
    if len(k) == 0:
        print(f"  {name} | 0번"); continue
    lag = k[:, MSEE] - k[:, MMOVE]; seen = k[:, MSEE] > 0
    braked = np.isfinite(k[:, MBRAKE])
    first = np.isfinite(k[:, MBRAKE]) & np.isfinite(k[:, MMOVE]) & (k[:, MBRAKE] > k[:, MMOVE] + 0.05)
    print(f"  {name} | {len(k)}번 | 보인 때 {q(k[seen, MSEE])} s 전 | 옆으로 움직이기 시작 {q(k[:, MMOVE])} s 전 (보고 나서 {q(lag[seen])} s 뒤; 보이기 전부터 "
          f"{int((lag[seen] < -0.05).sum())}번) | 절반 비킨 때 {q(k[:, MHALF])} s 전 | 움직이기 시작해서 90 %까지 {q(k[:, MDUR])} s | "
          f"감속한 통과 {100 * braked.mean():.0f} % (가장 크게 {q(k[braked, MDMAX])} m/s 느림, 감속 시작 {q(k[braked, MBRAKE])} s 전, "
          f"보고 나서 {q((k[:, MSEE] - k[:, MBRAKE])[braked & seen])} s 뒤) | 감속이 옆 움직임보다 먼저 {100 * first.mean():.0f} % | 잃은 시간 {q(k[:, MLOSS], '{:+.2f}')} s")
print("(나) plan이 상자 옆에 둔 거리 (plan이 상자를 지나칠 만큼 길 때만; 중앙 [25~75 %] m, 괄호 = 그런 plan이 있는 통과 수)")
for name, m in GROUPS[1:]:
    if m.sum() < 5:
        continue
    txt = []
    for lead in (12, 10, 9, 8, 7, 6, 5, 4, 3, 2):
        g = G[m, PRE - lead]; txt.append(f"{lead / 10:.1f} s 전 {q(g, '{:.2f}')} ({int(np.isfinite(g).sum())})")
    print(f"  {name}: " + " · ".join(txt) + f" · 실제 {q(M[m, MCLEAR], '{:.2f}')}")
print("(다) 같은 만큼 비킨 통과끼리, 옆으로 일찍 움직이기 시작한 절반과 늦게 움직인 절반  (미리 움직이면 지금 보상에서 이득인가)")
for name, m in GROUPS[1:]:
    k = M[m & np.isfinite(M[:, MMOVE]) & np.isfinite(M[:, MLOSS])]
    if len(k) < 12:
        continue
    cut = np.median(k[:, MMOVE]); e, l = k[k[:, MMOVE] > cut], k[k[:, MMOVE] <= cut]
    cc = np.corrcoef(k[:, MMOVE], k[:, MLOSS])[0, 1]
    print(f"  {name}: 일찍({len(e)}번, {np.median(e[:, MMOVE]):.1f} s 전) 잃은 시간 {np.median(e[:, MLOSS]):+.2f} s · 가장 크게 {np.median(e[:, MDMAX]):.1f} m/s 느림 · 들어가는 속도 "
          f"{np.median(e[:, MV]):.1f} m/s | 늦게({len(l)}번, {np.median(l[:, MMOVE]):.1f} s 전) 잃은 시간 {np.median(l[:, MLOSS]):+.2f} s · {np.median(l[:, MDMAX]):.1f} m/s 느림 · "
          f"들어가는 속도 {np.median(l[:, MV]):.1f} m/s | 움직이기 시작한 때와 잃은 시간의 상관 {cc:+.2f} (음수 = 일찍 움직일수록 덜 잃음)")
print("(라) 잃은 시간이 어디서 나오나: 잃은 시간과의 상관 — 가장 크게 느려진 속도 "
      f"{np.corrcoef(M[np.isfinite(M[:, MLOSS]), MDMAX], M[np.isfinite(M[:, MLOSS]), MLOSS])[0, 1]:+.2f}, 비킨 거리 "
      f"{np.corrcoef(M[np.isfinite(M[:, MLOSS]), MF], M[np.isfinite(M[:, MLOSS]), MLOSS])[0, 1]:+.2f}")

print("(마) 잃은 시간이 어느 구간에서 나오나 (치운 뒤 같은 구간 대비, 중앙 [25~75 %] s) · 옆으로 움직이기 시작할 때 상자까지 거리")
for name, m in GROUPS:
    if m.sum() < 5:
        continue
    print(f"  {name} | 상자 8 m 전 ~ 상자 {q(X[m, XB], '{:+.2f}')} | 상자 ~ 3 m 뒤 {q(X[m, XN], '{:+.2f}')} | 3 ~ 10 m 뒤 {q(X[m, XA], '{:+.2f}')} | "
          f"움직이기 시작할 때 상자까지 {q(X[m, XDM], '{:.1f}')} m")
print("(바) 횡방향 비용을 둔다면 무엇을 세게 되나: 상자 8 m 전 ~ 6 m 뒤 구간에서 더한 값, 상자 있을 때 / 치운 뒤 같은 구간 (중앙 [25~75 %]; 1 = 같음)")
for name, m in GROUPS:
    if m.sum() < 5:
        continue
    ra, rj = X[m, XIA] / X[m, XIA0], X[m, XIJ] / X[m, XIJ0]
    k = np.isfinite(rj) & np.isfinite(M[m, MMOVE])
    txt = ""
    if k.sum() >= 12:
        mv = M[m, MMOVE][k]; cut = np.median(mv)
        txt = (f" | 일찍 움직인 절반의 변화² 비 {np.median(rj[k][mv > cut]):.2f} · 늦게 움직인 절반 {np.median(rj[k][mv <= cut]):.2f} "
               f"(움직이기 시작한 때와의 상관 {np.corrcoef(mv, rj[k])[0, 1]:+.2f}; 음수 = 일찍 움직일수록 비용이 작음)")
    print(f"  {name} | 횡가속도² {q(ra, '{:.2f}')} | 횡가속도 변화² {q(rj, '{:.2f}')} | 치운 뒤보다 큰 통과 {100 * np.nanmean(rj > 1.0):.0f} %" + txt)
print(f"  참고: 치운 뒤 같은 구간의 값 자체 — 횡가속도² 중앙 {np.nanmedian(X[:, XIA0]):.0f} (m/s²)²·s, 횡가속도 변화² 중앙 {np.nanmedian(X[:, XIJ0]):.0f} (m/s³)²·s "
      f"(빈 트랙을 달리는 것만으로 내는 값)")

print("(사) 상자가 보이기 전(가장 가까워지기 2.5~1.6 s 전)에 이미 평소와 다른가: 속도 차와 옆으로 벗어난 거리 (중앙 [25~75 %])")
pre = slice(PRE - 25, PRE - 15)
for name, m in (("앞선 다른 상자가 20 m 넘게 떨어져 있거나 없음", X[:, XDU] > 20.0), ("앞선 다른 상자가 12 m 안", X[:, XDU] < 12.0)):
    for nm2, m2 in (("둘째 통과부터", M[:, MN] >= 2), ("첫째 통과", M[:, MN] == 1)):
        mm = m & m2 & (M[:, MSEE] > 0) & (M[:, MSEE] < 1.6)
        if mm.sum() >= 5:
            print(f"  {name} · {nm2} | {int(mm.sum())}번 | 속도 차 {q(DV[mm][:, pre].mean(1), '{:+.2f}')} m/s | 옆으로 {q(P[mm][:, pre].mean(1), '{:+.3f}')} m")

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#333333", "#888888"
tau = (np.arange(W) - PRE) * dt
sel = [(GROUPS[2][0].replace("비킨 거리 ", ""), GROUPS[2][1], BLUE), (GROUPS[3][0], GROUPS[3][1], ORANGE)]
fig, axs = plt.subplots(1, 4, figsize=(24, 6.0))
for name, m, col in sel:
    if m.sum() < 5:
        continue
    lab = f"{name} 비킨 통과 ({int(m.sum())}번)"
    for ax, X in ((axs[0], P), (axs[1], DV), (axs[2], G)):
        ok = np.isfinite(X[m]).sum(0) >= max(5, 0.5 * m.sum())
        with np.errstate(all="ignore"):
            lo, md, hi = np.nanpercentile(X[m], [25, 50, 75], axis=0)
        ax.fill_between(tau[ok], lo[ok], hi[ok], color=col, alpha=0.18, lw=0)
        ax.plot(tau[ok], md[ok], "-", color=col, lw=2.5, label=lab)
        see = np.median(M[m & (M[:, MSEE] > 0), MSEE])
        ax.axvline(-see, color=col, lw=1.5, ls="--")
    # the same moves counted from the moment the box became visible
    rel, ax = [], axs[3]
    for p, mm in zip(P[m], M[m]):
        if mm[MSEE] <= 0:
            continue
        i0 = PRE - int(round(mm[MSEE] / dt)); row = np.full(26, np.nan)
        for n, i in enumerate(range(i0 - 5, i0 + 21)):
            if 0 <= i <= PRE + 2:
                row[n] = p[i] / max(mm[MF], 1e-6)
        rel.append(row)
    rel = np.array(rel); x = (np.arange(26) - 5) * dt
    ok = np.isfinite(rel).sum(0) >= max(5, 0.5 * len(rel))
    with np.errstate(all="ignore"):
        lo, md, hi = np.nanpercentile(rel, [25, 50, 75], axis=0)
    ax.fill_between(x[ok], lo[ok], hi[ok], color=col, alpha=0.18, lw=0); ax.plot(x[ok], md[ok], "-", color=col, lw=2.5, label=lab)
for ax, yl, title in ((axs[0], "평소 길에서 옆으로 벗어난 거리 (m; 상자에서 멀어지는 쪽 +)", "① 옆으로 언제 움직이나"),
                      (axs[1], "속도 - 평소 같은 자리의 속도 (m/s)", "② 언제 얼마나 느려지나"),
                      (axs[2], "그때 낸 plan의 길이 상자 중심에서 떨어진 거리 (m)", "③ plan은 언제 상자를 비켜 그리나")):
    ax.axvline(0, color=INK, lw=1); ax.axhline(0, color=INK, lw=1)
    ax.set_xlabel("상자에 가장 가까워지기까지 (s; 0 = 상자 옆)   점선 = 상자가 보이기 시작한 때(중앙)"); ax.set_ylabel(yl); ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, frameon=False, loc="upper left"); ax.grid(alpha=0.25); ax.set_xlim(-2.5, 0.6)
axs[2].set_xlim(-1.3, 0.1)
ax = axs[3]; ax.axvline(0, color=INK, lw=1); ax.axhline(0, color=INK, lw=1); ax.axhline(1, color=MUTED, lw=1, ls=":")
ax.set_xlabel("상자가 보이기 시작한 뒤 (s; 0 = 처음 보인 때)"); ax.set_ylabel("벗어난 거리 / 최종 비킨 거리")
ax.set_title("④ 보고 나서 얼마 뒤에 움직이나 (1 = 다 비킴)", fontsize=12); ax.legend(fontsize=10, frameon=False, loc="upper left"); ax.grid(alpha=0.25)
fig.suptitle(f"{z['exam']} · {R.name} · {MODE} · 상자 {z['count']} — 혼자 달리는 세션, 길 위의 상자를 닿지 않고 지난 통과 (선 = 중앙, 띠 = 25~75 %)", fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(fig_out, dpi=70); print("그림", fig_out)

if scene_out:
    big = np.nonzero((M[:, MF] >= 0.30) & (M[:, MSEE] > 0) & np.isfinite(M[:, MMOVE]))[0]
    lag = M[big, MSEE] - M[big, MMOVE]; o = big[np.argsort(lag)]
    pick = [o[-1], o[-2], o[len(o) // 2], o[len(o) // 2 - 1], o[1], o[0]]          # the two slowest to move, two typical, the two quickest
    fig, axs = plt.subplots(2, 3, figsize=(19, 12.5))
    for ax, i in zip(axs.ravel(), pick):
        b, j, t, sgn = keep[i]; mm = M[i]; tk = int(R.tid[0, b])
        R.draw_map(ax, tk)
        kk, A, D, _ = refs[b]; c = pose[t, b, j, :2]
        near = np.linalg.norm(A - c, axis=1) < 9.0
        ax.plot(A[near, 0], A[near, 1], ".", color=BLUE, ms=3, zorder=3, label="상자를 치운 뒤 달린 길")
        w = np.arange(t - PRE, t + POST + 1)
        ax.plot(xy[w, b, 0], xy[w, b, 1], "-o", color=ORANGE, lw=1.8, ms=3.5, zorder=5, label="상자가 있을 때 달린 길 (점 = 0.1 s)")
        for poly in R.boxes(t, b):
            ax.fill(poly[:, 0], poly[:, 1], color=C_BOX, zorder=4)
        k_see = t - int(round(mm[MSEE] / dt)); k_mv = t - int(round(mm[MMOVE] / dt))
        for k, mk, colr, lab in ((k_see, "*", "#111111", "상자가 처음 보인 때의 차 위치"), (k_mv, "^", "#111111", "옆으로 움직이기 시작한 때")):
            ax.plot(*xy[k, b], mk, color=colr, ms=13 if mk == "*" else 9, mec="white", mew=1.0, zorder=7, label=lab)
        for k, colr, lab in ((k_see, "#7b3fbf", "처음 보인 때 낸 plan"), (t - 7, "#1a9850", "0.7 s 전에 낸 plan"), (t - 4, "#c2185b", "0.4 s 전에 낸 plan")):
            if k >= 0:
                q_ = np.concatenate([xy[k, b][None], pl[k, b]], 0)
                ax.plot(q_[:, 0], q_[:, 1], "-", color=colr, lw=2.0, alpha=0.9, zorder=6, label=lab)
        ax.set_xlim(c[0] - 6.0, c[0] + 6.0); ax.set_ylim(c[1] - 6.0, c[1] + 6.0)
        ax.set_title(f"세션 {b} · {t * dt:.1f} s · {int(mm[MN])}번째 통과 · {mm[MV]:.1f} m/s\n보인 때 {mm[MSEE]:.1f} s 전 · 옆으로 움직이기 시작 {mm[MMOVE]:.1f} s 전 · "
                     f"비킨 거리 {mm[MF]:.2f} m · 가장 크게 {mm[MDMAX]:.1f} m/s 느림", fontsize=10)
    axs[0, 0].legend(fontsize=8.5, loc="best", framealpha=0.9)
    fig.suptitle(f"{R.name} · {MODE} — 0.30 m 넘게 비킨 통과 6개: 윗줄 왼쪽 2개 = 보고 나서 가장 늦게 움직인 것, 윗줄 오른쪽·아랫줄 왼쪽 = 보통, 아랫줄 오른쪽 2개 = 가장 빨리 움직인 것",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(scene_out, dpi=70); print("장면", scene_out)
