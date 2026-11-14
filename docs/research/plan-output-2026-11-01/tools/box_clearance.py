"""길 위 상자를 지날 때 차체와 상자 사이에 남긴 틈 (11-12). 장면에서 본 것: 2상자 시험의 세게 닿은 통과 3건이 모두 그 세션에서
이미 여러 번 안 닿고 지나간 상자였고, 닿은 통과의 길은 앞선 통과들과 5~25 cm만 달랐음 -> 가설: 안 닿은 통과도 틈이 몇 cm뿐이라
통과마다의 흔들림만으로 닿는다 (못 봤거나 잊어서가 아니라 여유가 없어서).
반증: 안 닿은 통과의 틈 중앙값이 같은 상자 통과끼리의 흔들림보다 훨씬 크거나, 다음 통과에서 세게 닿는 비율이 이번 틈과 무관하면 틀림.
통과마다 틈 = 가장 가까워진 결정 앞뒤 0.5 s에서 차체 사각형과 상자 사각형 사이 거리의 최솟값. 결정(0.1 s) 사이는 요 각을 고르게
나누고 그 방향으로 나아간 것으로 채움(40등분, 끝점은 기록에 맞춤).
표: (가) 틈의 분포: 안 닿은 통과 / 약하게 닿은 통과  (나) 같은 세션 같은 상자의 통과끼리 틈이 얼마나 달라지나(연속 두 통과의 차이)
    (다) 이번 통과의 틈별로 바로 다음 통과에서 닿는 비율  (라) 혼자 / 상대 차 있을 때
인자: 기록npz 행npy [기록npz 행npy ...]   (행 = exam_boxes.py의 EXAM_ROWS; BOX_CLEAR_FIG=png 이면 그림도)"""
import os
import sys
import numpy as np
from rec_lib import Rec, plt

IB, IJ, IT, IN, IGAP, ISEE, ISEED, ITOUCH, IHARD, IPREV, IVIN, IDROP, ISHIFT, IPASS, IOPP, IBX, IBY = range(17)
FIG = os.environ.get("BOX_CLEAR_FIG", "")
SUB, HALF = 40, 5
BINS = ((0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.20), (0.20, 9.0))


def wilson(k, n):
    if n == 0:
        return "- (0)"
    p = k / n; zz = 1.96; d = 1 + zz * zz / n
    c = (p + zz * zz / (2 * n)) / d; h = zz * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n)) / d
    return f"{100 * p:.1f} % [{100 * max(0, c - h):.1f}-{100 * min(1, c + h):.1f}] ({k}/{n})"


def poly_gap(P, Q):
    """smallest distance between the convex quads P (m, 4, 2) and one quad Q (4, 2); 0 when they overlap"""
    def pt_seg(p, a, b):                                               # p (m, i, 2) to segments a->b (j, 2)
        ab = b - a
        u = np.clip(((p[:, :, None] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
        return np.linalg.norm(a + u[..., None] * ab - p[:, :, None], axis=-1).min((1, 2))

    def inside(p, poly):                                               # any of p (m, i, 2) inside each poly (m, 4, 2) or (4, 2)
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


def one(fn, rows_fn):
    R = Rec(fn); z = R.z; T, dt = R.T, R.dt
    rows = np.load(rows_fn)
    rows = rows[rows[:, IGAP] < 0.40]                                  # boxes on the driven path
    st = R.st.astype(float); pose = R.pose.astype(float)
    q = np.array([[R.hl, R.hw], [R.hl, -R.hw], [-R.hl, -R.hw], [-R.hl, R.hw]])
    gap = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        b, j, t = int(r[IB]), int(r[IJ]), int(r[IT])
        k = int(R.sid[t, b, j])
        if k < 0:
            continue
        fp = z[f"fp_{k}"].astype(float)
        a, e = max(int(R.k0[t, b]), t - HALF), min(T, t + HALF)
        best = np.inf
        for s in range(a, e):
            if R.rs[s, b] or np.linalg.norm(st[s + 1, b, :2] - st[s, b, :2]) > 2.0:
                continue
            u = np.linspace(0.0, 1.0, SUB + 1)
            dy = (st[s + 1, b, 2] - st[s, b, 2] + np.pi) % (2 * np.pi) - np.pi
            yaw = st[s, b, 2] + u * dy
            v = st[s, b, 3] + u * (st[s + 1, b, 3] - st[s, b, 3])
            step = np.stack([v * np.cos(yaw), v * np.sin(yaw)], 1) * dt / SUB
            p = st[s, b, :2] + np.concatenate([np.zeros((1, 2)), np.cumsum(0.5 * (step[1:] + step[:-1]), 0)], 0)
            p += u[:, None] * (st[s + 1, b, :2] - p[-1])               # end on the recorded point
            c, sn = np.cos(yaw), np.sin(yaw)
            P = np.stack([p[:, 0, None] + q[:, 0] * c[:, None] - q[:, 1] * sn[:, None],
                          p[:, 1, None] + q[:, 0] * sn[:, None] + q[:, 1] * c[:, None]], -1)
            bp = pose[s, b, j]; cb, sb = np.cos(bp[2]), np.sin(bp[2])
            Q = np.stack([bp[0] + fp[:, 0] * cb - fp[:, 1] * sb, bp[1] + fp[:, 0] * sb + fp[:, 1] * cb], 1)
            best = min(best, float(poly_gap(P, Q).min()))
        gap[i] = best
    ok = np.isfinite(gap); rows, gap = rows[ok], gap[ok]
    touch, hard = rows[:, ITOUCH] > 0, rows[:, IHARD] > 0
    clean = ~touch
    MODE = dict(map="MAP", track="시간 추종기", ideal="plan대로 정확히")[str(z["mode"])]
    print(f"== {R.name} · {MODE} ({fn.split('/')[-1]}): 길 위 상자 통과 {len(rows)}번 (안 닿음 {int(clean.sum())} · 약하게 {int((touch & ~hard).sum())} · "
          f"세게 {int(hard.sum())}) ==")

    def dist(name, m):
        g = gap[m]
        if len(g) < 5:
            print(f"  {name}: {len(g)}번"); return
        print(f"  {name} {len(g)}번: 틈 중앙 {100 * np.median(g):.0f} cm · 가장 좁은 10 % {100 * np.percentile(g, 10):.0f} cm · 25 % "
              f"{100 * np.percentile(g, 25):.0f} cm | 3 cm 아래 {100 * (g < 0.03).mean():.0f} % · 6 cm 아래 {100 * (g < 0.06).mean():.0f} % · "
              f"10 cm 아래 {100 * (g < 0.10).mean():.0f} % · 20 cm 넘음 {100 * (g >= 0.20).mean():.0f} %")

    print("(가) 차체와 상자 사이 틈")
    dist("안 닿은 통과", clean); dist("  그중 혼자", clean & (rows[:, IOPP] == 0)); dist("  그중 상대 차 있음", clean & (rows[:, IOPP] > 0))
    # pairs: the same car and box, consecutive passes
    key = rows[:, IB] * 1000 + rows[:, IJ]
    order = np.lexsort((rows[:, IN], key))
    A, Bn = [], []
    for a_, b_ in zip(order[:-1], order[1:]):
        if key[a_] == key[b_] and rows[b_, IN] == rows[a_, IN] + 1:
            A.append(a_); Bn.append(b_)
    A, Bn = np.array(A, int), np.array(Bn, int)
    both = clean[A] & clean[Bn]
    if both.sum() >= 5:
        dd = np.abs(gap[A][both] - gap[Bn][both])
        print(f"(나) 같은 세션 같은 상자, 연속 두 통과 모두 안 닿은 {int(both.sum())}쌍: 틈 차이 중앙 {100 * np.median(dd):.0f} cm · 90 % "
              f"{100 * np.percentile(dd, 90):.0f} cm · 가장 큰 5 % {100 * np.percentile(dd, 95):.0f} cm")
    print("(다) 이번에 안 닿고 지난 틈별, 같은 상자 바로 다음 통과")
    for lo, hi in BINS:
        m = clean[A] & (gap[A] >= lo) & (gap[A] < hi)
        n = int(m.sum())
        lab = f"{100 * lo:.0f}~{100 * hi:.0f} cm" if hi < 9 else f"{100 * lo:.0f} cm 넘음"
        print(f"  {lab:<12} | 다음에 세게 {wilson(int(hard[Bn][m].sum()), n)} | 다음에 닿음(약한 것 포함) {wilson(int(touch[Bn][m].sum()), n)}")
    m = touch[A]
    print(f"  이번에 닿음   | 다음에 세게 {wilson(int(hard[Bn][m].sum()), int(m.sum()))} | 다음에 닿음 {wilson(int(touch[Bn][m].sum()), int(m.sum()))}")
    count = dict(uniform="상자 1~18개", max2="상자 2개").get(str(z["count"]), str(z["count"])) if "count" in z.files else ""
    return f"{R.name} · {count}", gap, clean, touch, hard, (A, Bn)


if __name__ == "__main__":
    res = [one(sys.argv[i], sys.argv[i + 1]) for i in range(1, len(sys.argv) - 1, 2)]
    if FIG:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
        ax = axes[0]
        edges = np.arange(0, 0.62, 0.02)
        for k, (name, gap, clean, touch, hard, _) in enumerate(res):
            ax.hist(100 * gap[clean], bins=100 * edges, histtype="step", lw=2, density=True, color=("#1f77b4", "#e6820e", "#2ca02c", "#9467bd")[k % 4],
                    label=f"{name} (안 닿은 통과 {int(clean.sum())}번, 중앙 {100 * np.median(gap[clean]):.0f} cm)")
        ax.set_xlabel("가장 가까웠을 때 차체와 상자 사이 틈 [cm]"); ax.set_ylabel("비율 (밀도)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title("길 위 상자를 안 닿고 지날 때 남긴 틈", fontsize=10)
        ax = axes[1]
        for k, (name, gap, clean, touch, hard, (A, Bn)) in enumerate(res):
            xs, ys, lo_, hi_ = [], [], [], []
            for i, (lo, hi) in enumerate(BINS):
                m = clean[A] & (gap[A] >= lo) & (gap[A] < hi); n = int(m.sum())
                if n < 5:
                    continue
                p = touch[Bn][m].mean(); se = 1.96 * np.sqrt(max(p * (1 - p), 1e-9) / n)
                xs.append(i + 0.12 * (k - 0.5 * (len(res) - 1))); ys.append(100 * p); lo_.append(100 * min(p, se)); hi_.append(100 * se)
            ax.errorbar(xs, ys, yerr=[lo_, hi_], fmt="o", ms=7, capsize=3, color=("#1f77b4", "#e6820e", "#2ca02c", "#9467bd")[k % 4], label=name)
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels([f"{100 * lo:.0f}~{100 * hi:.0f} cm" if hi < 9 else f"{100 * lo:.0f} cm 넘음" for lo, hi in BINS])
        ax.set_xlabel("이번 통과에서 남긴 틈 (안 닿음)"); ax.set_ylabel("같은 상자 바로 다음 통과에서 닿은 비율 [%]"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        ax.set_title("이번에 바짝 지나간 상자를 다음 바퀴에 닿는가 (막대 = 95 % 구간)", fontsize=10)
        fig.savefig(FIG, dpi=110); plt.close(fig)
        print("그림", FIG)
