"""상자로 다가가는 동안 plan이 상자의 어느 쪽으로 지나가려 했나 (11-12). 앞서 본 것(box_clearance.py와 짝 비교): 이미 안 닿고 지나간
상자를 다음 바퀴에 세게 닿은 통과는 들어가는 속도도, 가까운 상대 차도 앞선 통과와 같았는데 옆으로 비킨 폭만 0에 가까웠음.
가설: 닿은 통과에서는 다가가는 동안 plan이 상자의 왼쪽·오른쪽 사이를 오가거나 앞선 통과와 반대쪽을 골라 늦게 정함 (양쪽 사이에서
못 정하면 평균은 한가운데 = 상자).
반증: 닿은 통과의 plan이 안 닿은 통과보다 쪽을 더 자주 바꾸지 않고 앞선 통과와 같은 쪽으로 지나가려 했다면 틀림 -> 그때는 plan이
처음부터 상자를 지나는 길(상자에 반응이 없음)이었는지, 비키는 plan이 늦게 나왔는지를 같은 표에서 봄.
결정마다(0.1 s) plan = 지금 자리 + 10점. plan을 10 cm 간격으로 다시 찍어 그 방향으로 차체 사각형을 놓고 상자 사각형까지의 틈을 잼.
plan의 끝이 상자 자리에 못 미치면 "아직 안 닿음". 쪽 = plan에서 볼 때 상자가 왼쪽(L)인지 오른쪽(R)인지.
표: (가) 같은 상자 연속 두 통과에서 실제로 지나간 쪽이 바뀐 비율 (다음에 세게 닿은 쌍 / 안 닿은 쌍)
    (나) 가장 가까워지기 몇 초 전의 plan: 상자 자리까지 닿은 비율 · 그중 상자에 닿는 plan(틈 5 cm 아래) 비율 · 틈 중앙값 · 끝내 지나간 쪽과 같은 쪽 비율
    (다) 통과마다: 마지막 1.5초에 plan의 쪽이 바뀐 통과 비율 · 비키는 plan(틈 5 cm 넘음)이 끝까지 이어지기 시작한 때
    (라) 세게 닿은 통과 하나하나: 1.5초 전부터 0.1초마다 plan의 쪽 (대문자 = 비키는 plan, 소문자 = 상자에 닿는 plan, . = 아직 안 닿음)
인자: 기록npz 행npy [기록npz 행npy ...]   (BOX_SIDE_FIG=png 이면 첫 기록의 닿은 통과와 그 앞선 통과를 나란히 그림)"""
import os
import sys
import numpy as np
from matplotlib.patches import Polygon
from rec_lib import Rec, plt
import rec_lib as RL

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
FIG = os.environ.get("BOX_SIDE_FIG", "")
NB = 15                                                               # decisions before the closest one
CLEAR = 0.05
TAUS = (1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)


def poly_gap(P, Q):
    """smallest distance between the convex quads P (m, 4, 2) and one quad Q (4, 2); 0 when they overlap (box_clearance.py's)"""
    def pt_seg(p, a, b):
        ab = b - a
        u = np.clip(((p[:, :, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
        return np.linalg.norm(a + u[..., None] * ab - p[:, :, None], axis=-1).min((1, 2))

    def inside(p, poly):
        poly = np.broadcast_to(poly, (p.shape[0], 4, 2))
        a, b = poly, np.roll(poly, -1, 1)
        cr = (b - a)[:, None, :, 0] * (p[:, :, None, 1] - a[:, None, :, 1]) - (b - a)[:, None, :, 1] * (p[:, :, None, 0] - a[:, None, :, 0])
        return ((cr >= 0).all(-1) | (cr <= 0).all(-1)).any(-1)

    Qm = np.broadcast_to(Q, P.shape)
    d1 = pt_seg(P, Q, np.roll(Q, -1, 0))
    d2 = np.stack([pt_seg(Qm[i:i + 1], P[i], np.roll(P[i], -1, 0))[0] for i in range(len(P))])
    d = np.minimum(d1, d2)
    d[inside(P, Q) | inside(Qm, P)] = 0.0
    return d


def plan_vs_box(X, c, Q, q):
    """X (11, 2) the plan from the car on; c (2,) the box centre; Q (4, 2) its quad; q (4, 2) the car's quad in its frame.
    -> reached (the plan gets as far as the box), side (+1 = the box on the plan's left), gap [m] body to box along the plan"""
    seg = X[1:] - X[:-1]; L = np.linalg.norm(seg, axis=1)
    if L.sum() < 0.05:
        return False, 0.0, np.nan
    u = np.clip(((c - X[:-1]) * seg).sum(1) / np.maximum(L * L, 1e-12), 0, 1)
    near = X[:-1] + u[:, None] * seg
    d = np.linalg.norm(near - c, axis=1); i = int(d.argmin())
    last = max(int(np.nonzero(L > 1e-6)[0][-1]), 0)
    if i >= last and u[i] >= 1.0 - 1e-9 and d[i] > 0.3:                # the box lies beyond the plan's end
        return False, 0.0, np.nan
    side = float(np.sign(seg[i, 0] * (c[1] - near[i, 1]) - seg[i, 1] * (c[0] - near[i, 0])))
    s = np.concatenate([[0.0], np.cumsum(L)])
    ss = np.arange(0.0, s[-1] + 1e-9, 0.1)
    p = np.stack([np.interp(ss, s, X[:, 0]), np.interp(ss, s, X[:, 1])], 1)
    j = np.clip(np.searchsorted(s, ss, side="right") - 1, 0, len(seg) - 1)
    yaw = np.arctan2(seg[j, 1], seg[j, 0])
    m = np.linalg.norm(p - c, axis=1) < 1.5
    if not m.any():
        return True, side, 1.0
    p, yaw = p[m], yaw[m]; cy, sy = np.cos(yaw), np.sin(yaw)
    P = np.stack([p[:, 0, None] + q[:, 0] * cy[:, None] - q[:, 1] * sy[:, None],
                  p[:, 1, None] + q[:, 0] * sy[:, None] + q[:, 1] * cy[:, None]], -1)
    return True, side, float(min(poly_gap(P, Q).min(), 1.0))


def pct(k, n):
    return f"{100 * k / n:.0f} % ({k}/{n})" if n else "- (0)"


def one(fn, rows_fn, fig):
    R = Rec(fn); z = R.z; dt = R.dt
    rows = np.load(rows_fn); rows = rows[rows[:, IGAP] < 0.40]
    st = R.st.astype(float); pose = R.pose.astype(float); pl = R.pl.astype(float)
    q = np.array([[R.hl, R.hw], [R.hl, -R.hw], [-R.hl, -R.hw], [-R.hl, R.hw]])
    n = len(rows)
    reach = np.zeros((n, NB), bool); side = np.zeros((n, NB)); gap = np.full((n, NB), np.nan)   # column k = (k + 1) decisions before
    act = np.zeros(n)
    for i, r in enumerate(rows):
        b, j, t = int(r[IB]), int(r[IJ]), int(r[IT])
        k = int(R.sid[t, b, j])
        if k < 0:
            continue
        fp = z[f"fp_{k}"].astype(float)
        bp = pose[t, b, j]; cb, sb = np.cos(bp[2]), np.sin(bp[2])
        Q = np.stack([bp[0] + fp[:, 0] * cb - fp[:, 1] * sb, bp[1] + fp[:, 0] * sb + fp[:, 1] * cb], 1)
        c = bp[:2]
        x, y, yaw = st[t, b, :3]
        act[i] = np.sign(-np.sin(yaw) * (c[0] - x) + np.cos(yaw) * (c[1] - y))
        for kk in range(NB):
            s = t - 1 - kk
            if s < int(R.k0[t, b]) or s >= pl.shape[0]:
                break
            X = np.concatenate([st[s, b, :2][None], pl[s, b]], 0)
            reach[i, kk], side[i, kk], gap[i, kk] = plan_vs_box(X, c, Q, q)
    touch, hard = rows[:, ITOUCH] > 0, rows[:, IHARD] > 0
    clean = ~touch
    MODE = dict(map="MAP", track="시간 추종기", ideal="plan대로 정확히")[str(z["mode"])]
    print(f"== {R.name} · {MODE} ({fn.split('/')[-1]}): 길 위 상자 통과 {n}번 (안 닿음 {int(clean.sum())} · 세게 {int(hard.sum())}) ==")
    key = rows[:, IB] * 1000 + rows[:, IJ]; order = np.lexsort((rows[:, IN], key))
    A = np.array([(a, b) for a, b in zip(order[:-1], order[1:]) if key[a] == key[b] and rows[b, IN] == rows[a, IN] + 1 and clean[a]], int)
    print("(가) 같은 상자 연속 두 통과(앞선 통과는 안 닿음)에서 실제로 지나간 쪽이 바뀜")
    for lab, m in (("다음에 세게 닿은 쌍", hard[A[:, 1]]), ("다음에도 안 닿은 쌍", clean[A[:, 1]])):
        a, b = A[m, 0], A[m, 1]
        print(f"  {lab}: {pct(int((act[a] != act[b]).sum()), int(m.sum()))}")
    chg = act[A[:, 0]] != act[A[:, 1]]
    print("(가2) 앞선 통과(안 닿음)와 같은 쪽으로 갔는가에 따라 이번 통과가 닿은 비율")
    for lab, m in (("반대쪽으로 감", chg), ("같은 쪽으로 감", ~chg)):
        print(f"  {lab}: 세게 {pct(int(hard[A[m, 1]].sum()), int(m.sum()))} · 약하게라도 닿음 {pct(int(touch[A[m, 1]].sum()), int(m.sum()))}")

    def off_prev(a, b, tau):
        """how far the car is, `tau` before this pass's closest point, from the path it drove on the pass before [m]"""
        ra, rb = rows[a], rows[b]; car = int(rb[IB]); ta, tb = int(ra[IT]), int(rb[IT])
        s = tb - int(round(tau / dt))
        a0 = max(int(R.k0[ta, car]), ta - 30)
        if s < int(R.k0[tb, car]) or ta + 6 - a0 < 3:
            return np.nan
        P = st[a0:ta + 6, car, :2]; p = st[s, car, :2]
        ab = P[1:] - P[:-1]
        u = np.clip(((p - P[:-1]) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
        d = np.linalg.norm(P[:-1] + u[:, None] * ab - p, axis=1)
        i = int(d.argmin())
        return np.nan if (i == 0 and u[0] <= 0.0) else float(d[i])     # before the stretch kept of the earlier pass
    print("(마) 앞선 통과가 달린 길에서 이번 통과가 옆으로 벗어나 있던 거리 (중앙 / 90 %; 괄호 = 쌍 수)")
    for lab, m in (("반대쪽으로 간 쌍", chg), ("같은 쪽으로 간 쌍", ~chg), ("세게 닿은 쌍", hard[A[:, 1]]), ("안 닿은 쌍", clean[A[:, 1]])):
        s = f"  {lab}"
        for tau in (2.0, 1.5, 1.0, 0.5):
            d = np.array([off_prev(a, b, tau) for a, b in A[m]]); d = d[np.isfinite(d)]
            s += f" | {tau:.1f} s 전: {100 * np.median(d):.0f} / {100 * np.percentile(d, 90):.0f} cm ({len(d)})" if len(d) else f" | {tau:.1f} s 전: -"
        print(s)
    print("(바) 상자가 빈 트랙에서 달리던 길에서 떨어진 거리별: 세게 닿은 통과 · (쌍 가운데) 반대쪽으로 간 비율")
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.4)):
        m = (rows[:, IGAP] >= lo) & (rows[:, IGAP] < hi); mp = (rows[A[:, 1], IGAP] >= lo) & (rows[A[:, 1], IGAP] < hi)
        print(f"  {100 * lo:.0f}-{100 * hi:.0f} cm: {pct(int(hard[m].sum()), int(m.sum()))} · {pct(int(chg[mp].sum()), int(mp.sum()))}")
    G = (("세게 닿은 통과", hard), ("안 닿은 통과", clean))
    print("(나) 가장 가까워지기 전의 plan | 상자 자리까지 닿음 · 그중 상자에 닿는 plan · 틈 중앙 · 끝내 지나간 쪽과 같은 쪽")
    for tau in TAUS:
        kk = int(round(tau / dt)) - 1
        s = f"  {tau:.1f} s 전"
        for lab, m in G:
            rc = reach[m, kk]; g = gap[m, kk][rc]
            same = (side[m, kk][rc] == act[m][rc])
            s += (f" | {lab}: {pct(int(rc.sum()), int(m.sum()))} · {pct(int((g < CLEAR).sum()), len(g))} · "
                  f"{100 * np.median(g) if len(g) else float('nan'):.0f} cm · {pct(int(same.sum()), len(g))}")
        print(s)
    # per pass: side changes among the decisions whose plan reached the box; when the plan cleared it for good
    flips = np.zeros(n, int); t_clear = np.full(n, np.nan); t_first = np.full(n, np.nan)
    for i in range(n):
        ks = [k for k in range(NB - 1, -1, -1) if reach[i, k]]       # in time order
        sd = [side[i, k] for k in ks if side[i, k] != 0]
        flips[i] = int(sum(1 for a, b in zip(sd[:-1], sd[1:]) if a != b))
        if ks:
            t_first[i] = (ks[0] + 1) * dt
            ok = [gap[i, k] >= CLEAR for k in ks]
            j = len(ok)
            while j > 0 and ok[j - 1]:
                j -= 1
            if j < len(ok):
                t_clear[i] = (ks[j] + 1) * dt                         # clear from here to the end
    print("(다) 통과마다")
    for lab, m in G:
        tc = t_clear[m]
        print(f"  {lab} {int(m.sum())}번: 마지막 1.5초에 plan의 쪽이 바뀜 {pct(int((flips[m] > 0).sum()), int(m.sum()))} · "
              f"plan이 상자 자리에 처음 닿은 때 중앙 {np.nanmedian(t_first[m]):.1f} s 전 · 비키는 plan이 끝까지 이어진 통과 "
              f"{pct(int(np.isfinite(tc).sum()), int(m.sum()))}"
              + (f", 그 시작 중앙 {np.nanmedian(tc):.1f} s 전 (늦은 10 %: {np.nanpercentile(tc, 10):.1f} s 전)" if np.isfinite(tc).any() else ""))
    print("(라) 세게 닿은 통과 (왼쪽 = 1.5초 전 -> 오른쪽 = 0.1초 전; 대문자 = 비키는 plan, 소문자 = 상자에 닿는 plan, . = 아직 안 닿음)")
    prev_of = {int(b): int(a) for a, b in A}

    def strip(i):
        return "".join("." if not reach[i, k] else (("L" if side[i, k] > 0 else "R") if gap[i, k] >= CLEAR else ("l" if side[i, k] > 0 else "r"))
                       for k in range(NB - 1, -1, -1))
    hits = np.nonzero(hard)[0]
    for i in hits[:24]:
        r = rows[i]; a = prev_of.get(int(i))
        print(f"  차 {int(r[IB])} 상자 {int(r[IJ])} 통과 {int(r[IN])} | {strip(i)} | 지나간 쪽 {'L' if act[i] > 0 else 'R'}"
              + (f" | 앞선 통과(안 닿음): {strip(a)} 지나간 쪽 {'L' if act[a] > 0 else 'R'}" if a is not None else " | 앞선 통과 없음(첫 통과이거나 앞에서 닿음)"))
    if fig:
        pairs = [(prev_of[int(i)], int(i)) for i in hits if int(i) in prev_of][:8]
        if pairs:
            nc = 4; nr = int(np.ceil(len(pairs) * 2 / nc))
            f_, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 4.6 * nr), squeeze=False, constrained_layout=True)
            for ax in axes.flat:
                ax.axis("off")
            cm = plt.get_cmap("viridis")
            for n_, (a, b_) in enumerate(pairs):
                for h, i in enumerate((a, b_)):
                    ax = axes.flat[2 * n_ + h]; ax.axis("on")
                    r = rows[i]; b, j, t = int(r[IB]), int(r[IJ]), int(r[IT])
                    R.draw_map(ax, int(R.tid[t, b]))
                    for Qb in R.boxes(t, b):
                        ax.add_patch(Polygon(Qb, closed=True, fc=RL.C_BOX, ec="k", lw=0.5, zorder=3))
                    a0 = max(int(R.k0[t, b]), t - 20)
                    ax.plot(st[a0:t + 6, b, 0], st[a0:t + 6, b, 1], "-", color="0.35", lw=1.0, zorder=4)
                    for kk in (11, 7, 4, 2, 0):
                        s = t - 1 - kk
                        if s < a0:
                            continue
                        X = np.concatenate([st[s, b, :2][None], pl[s, b]], 0)
                        ax.plot(X[:, 0], X[:, 1], "-", color=cm(1.0 - kk / 12), lw=1.8, zorder=5)
                        ax.plot(X[0, 0], X[0, 1], "o", color=cm(1.0 - kk / 12), ms=4, zorder=6)
                    ax.add_patch(Polygon(R.body(*st[t, b, :3]), closed=True, fc="none", ec="red" if hard[i] else "k", lw=1.5, zorder=7))
                    c = pose[t, b, j, :2]
                    ax.set_xlim(c[0] - 3, c[0] + 3); ax.set_ylim(c[1] - 3, c[1] + 3); ax.set_aspect("equal")
                    ax.set_title(f"차 {b} 상자 {j} · 통과 {int(r[IN])} ({t * dt:.1f} s) · " + ("세게 닿음" if hard[i] else "안 닿음")
                                 + f"\n1초 전 {r[IVIN]:.1f} m/s · plan의 쪽 {strip(i)}", fontsize=8.5)
            f_.suptitle(f"{R.name}: 같은 상자의 앞선 통과(왼쪽, 안 닿음)와 세게 닿은 통과(오른쪽). 선 = 1.2 / 0.8 / 0.5 / 0.3 / 0.1초 전의 plan "
                        f"(어두움 -> 밝음), 회색 = 달린 길, 사각형 = 가장 가까웠을 때 차체", fontsize=11)
            f_.savefig(fig, dpi=95); plt.close(f_)
            print("그림:", fig)


for a in range(1, len(sys.argv) - 1, 2):
    one(sys.argv[a], sys.argv[a + 1], FIG if a == 1 else "")
