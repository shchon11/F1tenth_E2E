"""box_response.py 기록의 표와 그림: 상자 옆을 지나간 건을 지나간 순간에 맞춰, 그 전 시각별로 상자가 스캔에 보이나 / 거리 /
상자 때문에 plan의 1.0 s·0.5 s·0.1 s 점이 옮겨진 거리 / 상자 없는 plan과 실제 plan이 상자 위를 지나는 비율.
"길을 막은 상자" = 지나가기 0.1~1.2 s 전에 상자 없는 plan이 그 상자 중심 0.40 m 안을 지남 (비켜야 하는 건).
반응 시작 = 그 점이 옮겨진 거리가 문턱(1.0 s 점 15 cm, 0.1 s 점 2 cm)을 0.3 s 이어서 넘기 시작한 때.
인자: npz [출력 png]"""
import sys
import numpy as np

D = np.load(sys.argv[1])
st, sid, pose, cnt, rs = D["st"], D["sid"], D["pose"], D["cnt"], D["rs"]
beams = D["seen"]; seen = beams >= 2          # the box is in the scan: two beams or more end on it
d1, d5, d10, dv5, cfhit, rlhit, nb = D["d1"], D["d5"], D["d10"], D["dv5"], D["cfhit"], D["rlhit"], D["nb"]
dt = float(D["dt"]); T, B, P = sid.shape
W = 30
tid = D["tid"]
_edt, _off, _H, _W, _res, _org = D["edt"], D["t_off"], D["t_H"].astype(int), D["t_W"].astype(int), D["t_res"], D["t_origin"]
_occ = {}
F = np.linspace(0.0, 1.0, 240)


def wall_free(k, a, b_):
    """(n,) is the straight line from a (n, 2) to b_ (n, 2), stopping 0.45 m short, free of walls (other boxes and cars not looked at)"""
    if k not in _occ:
        _occ[k] = _edt[_off[k]: _off[k] + _H[k] * _W[k]].reshape(_H[k], _W[k]) <= 0.0
    seg = a[:, None, :] + F[None, :, None] * (b_ - a)[:, None, :]
    c = np.floor((seg[..., 0] - _org[k][0]) / _res[k]).astype(int); r = np.floor((seg[..., 1] - _org[k][1]) / _res[k]).astype(int)
    ok = (c >= 0) & (c < _W[k]) & (r >= 0) & (r < _H[k])
    occ = np.ones(seg.shape[:-1], bool); occ[ok] = _occ[k][r[ok], c[ok]]
    dd_ = np.linalg.norm(b_ - a, axis=1)
    fr = np.clip(1.0 - 0.45 / np.maximum(dd_, 0.5), 0.0, 1.0)
    return ~(occ & (F[None] <= fr[:, None])).any(1)
dprop = np.diff(np.concatenate([np.zeros((1, B, 6), np.float32), cnt], 0), axis=0).clip(min=0.0)
live = sid >= 0
xy = st[:, :, :2].astype(float)
dist = np.where(live, np.linalg.norm(pose[..., :2].astype(float) - xy[:, :, None], axis=3), 1e9)
near_j = dist.argmin(2)
rows, S = [], []                    # per pass: (in the way, touched, hard, isolated, first-sight tau, n-th pass) and the aligned series
for b in range(B):
    if not live[:, b].any():
        continue
    start = np.concatenate([[0], np.nonzero(rs[:, b])[0] + 1])
    for j in range(P):
        if not live[:, b, j].any():
            continue
        d = dist[:, b, j]; n_pass = 0; t = 1
        while t < T - 1:
            lo, hi = max(t - 10, 0), min(t + 10, T - 1)
            if d[t] < 1.2 and d[t] <= d[lo: hi + 1].min():
                ep0 = int(start[start <= t].max())
                if t - ep0 >= W:
                    sl = slice(t - W, t + 1)
                    w0, w1 = max(t - 8, 0), min(t + 5, T - 1)
                    mine = near_j[w0: w1 + 1, b] == j
                    touch = bool((dprop[w0: w1 + 1, b, 1] * mine).sum() > 0.5); hard = bool((dprop[w0: w1 + 1, b, 4] * mine).sum() > 0.5)
                    bj = pose[t, b, j, :2]
                    oth = live[t, b] & (np.arange(P) != j)
                    iso = not (oth & (np.linalg.norm(pose[t, b, :, :2] - bj, axis=1) < 4.0)).any()
                    way = bool(cfhit[t - 12: t, b, j].any())
                    sn = seen[sl, b, j]
                    u = W
                    while u > 0 and (sn[u - 1] or (u > 1 and sn[u - 2])):
                        u -= 1
                    rows.append((way, touch, hard, iso, (u - W) * dt, n_pass))
                    los = wall_free(int(tid[t, b]), xy[sl, b], pose[sl, b, j, :2].astype(float)) & (d[sl] < float(D["range_max"]))
                    S.append(np.stack([sn, d[sl], d1[sl, b], d5[sl, b], d10[sl, b], dv5[sl, b], cfhit[sl, b, j], rlhit[sl, b, j], np.abs(st[sl, b, 3]), los, beams[sl, b, j]], 1))
                n_pass += 1; t = hi + 1
            else:
                t += 1
R = np.array(rows, float); S = np.array(S, float)          # (n, W + 1, 11)
if len(rows) < 5:
    raise SystemExit(f"상자 옆을 지나간 건 {len(rows)}: 너무 적음 (출발 3 s 뒤의 건만 셈)")
print(f"{D['name']}: 학습 차 {B}대 x {T * dt:.0f} s, 상자 1.2 m 안으로 지나간 건 {len(R)} (출발 3 s 안은 뺌) | 길을 막은 상자 {int(R[:, 0].sum())} "
      f"({R[:, 0].mean() * 100:.0f} %) | 세게 닿음: 길을 막은 건 {int((R[:, 0] * R[:, 2]).sum())}, 아닌 건 {int(((1 - R[:, 0]) * R[:, 2]).sum())}")


def onset(x, thr, first):
    """earliest tau (s) from the box's first sight on where x stays over thr for 3 decisions; nan if never"""
    k0 = int(round(first / dt)) + W
    for k in range(max(k0, 0), W - 1):
        if (x[k: k + 3] > thr).all():
            return (k - W) * dt
    return np.nan


def block(m, title):
    A_ = S[m]; r = R[m]
    if len(A_) < 5:
        print(f"\n[{title}] 건수 {len(A_)}: 너무 적음"); return
    print(f"\n[{title}] 건수 {len(A_)}, 닿음 {r[:, 1].mean() * 100:.1f} %, 세게 닿음 {r[:, 2].mean() * 100:.1f} % ({int(r[:, 2].sum())}건)")
    print("  지나가기 전 | 벽에 안 가리고 10 m 안 | 스캔에 보임(빔 2개 이상) | 빔 수 중앙 | 거리 중앙 | 차 속도 중앙 | 1.0 s 점 옮김 중앙 | 0.5 s 점 | 0.1 s 점 | 0.5 s 점 속도 차 중앙 | 상자 없는 plan이 상자 위 | 실제 plan이 상자 위")
    for tau in (3.0, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        k = W - int(round(tau / dt)); a = A_[:, k]
        print(f"  {tau:4.1f} s | {a[:, 9].mean() * 100:5.1f} % | {a[:, 0].mean() * 100:5.1f} % | {np.median(a[:, 10]):3.0f} | {np.median(a[:, 1]):5.2f} m | {np.median(a[:, 8]):4.2f} m/s | {np.median(a[:, 4]) * 100:6.1f} cm | "
              f"{np.median(a[:, 3]) * 100:5.1f} cm | {np.median(a[:, 2]) * 100:4.1f} cm | {np.median(a[:, 5]):+5.2f} m/s | {a[:, 6].mean() * 100:5.1f} % | {a[:, 7].mean() * 100:5.1f} %")
    o10 = np.array([onset(A_[i, :, 4], 0.15, r[i, 4]) for i in range(len(A_))]); o1 = np.array([onset(A_[i, :, 2], 0.02, r[i, 4]) for i in range(len(A_))])
    q = lambda v: "없음" if np.isnan(v).all() else f"{np.nanmedian(v):+.1f} s (25~75 %: {np.nanpercentile(v, 25):+.1f} ~ {np.nanpercentile(v, 75):+.1f}, 끝까지 없음 {np.isnan(v).mean() * 100:.0f} %)"
    print(f"  처음 보인 때 중앙 {np.median(r[:, 4]):+.1f} s (3 s 넘게 전부터 보임 {(r[:, 4] <= -W * dt + 1e-6).mean() * 100:.0f} %)")
    print(f"  반응 시작, 1.0 s 점(15 cm): {q(o10)}")
    print(f"  반응 시작, 0.1 s 점 = 차가 실제로 가는 점(2 cm): {q(o1)}")
    return o10, o1


way = R[:, 0] > 0.5
block(way, "길을 막은 상자, 전체")
block(way & (R[:, 2] > 0.5), "길을 막은 상자, 세게 닿은 건")
block(way & (R[:, 1] < 0.5), "길을 막은 상자, 안 닿은 건")
block(way & (R[:, 3] > 0.5), "길을 막은 상자, 4 m 안에 다른 상자 없음")
block(~way, "길을 막지 않은 상자")
# what the memory carries: decisions of cars with boxes where no box beam was in the scan for the last 1 s
hasb = live.any(2)
quiet = np.ones_like(hasb)
for k in range(10):
    quiet[k:] &= (nb[: T - k] == 0) if k else (nb == 0)
quiet[:10] = False
q = quiet & hasb
if q.any():
    print(f"\n[상자가 1 s 동안 스캔에 하나도 안 보인 결정 (상자 있는 경주), {int(q.sum())}개] 상자를 본 적 있는 기억 때문에 1.0 s 점이 옮겨진 거리: "
          f"중앙 {np.median(d10[q]) * 100:.1f} cm, 90 % {np.percentile(d10[q], 90) * 100:.1f} cm | 0.1 s 점 중앙 {np.median(d1[q]) * 100:.2f} cm")

if len(sys.argv) > 2 and way.sum() >= 20:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
        if any(f == x.name for x in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = f; break
    plt.rcParams["axes.unicode_minus"] = False
    INK, MUTED = "#1f2328", "#6b7280"
    C = ("#3b6fb6", "#c2410c", "#2f855a")
    tau = (np.arange(W + 1) - W) * dt
    A_ = S[way]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), dpi=110)
    ax = axs[0]
    for y, lab, c in ((A_[:, :, 0].mean(0) * 100, "상자가 스캔에 보임", C[0]), (A_[:, :, 6].mean(0) * 100, "상자 없는 plan이 상자 위를 지남", C[2]),
                      (A_[:, :, 7].mean(0) * 100, "실제 plan이 상자 위를 지남", C[1])):
        ax.plot(tau, y, color=c, lw=2, label=lab)
    ax.set_ylabel("길을 막은 상자를 지나간 건 중 비율 (%)", color=INK); ax.set_ylim(0, 100)
    ax.set_title(f"길을 막은 상자 {int(way.sum())}건: 보이는 때와 plan이 비키는 때", loc="left", fontsize=11, color=INK)
    ax = axs[1]
    for k, lab, c in ((4, "1.0 s 점", C[0]), (3, "0.5 s 점", C[2]), (2, "0.1 s 점 (차가 실제로 가는 점)", C[1])):
        ax.plot(tau, np.median(A_[:, :, k], 0) * 100, color=c, lw=2, label=lab)
    ax.set_ylabel("상자 때문에 plan의 점이 옮겨진 거리, 중앙 (cm)", color=INK)
    ax.set_title("상자를 지운 스캔으로 낸 plan과의 차이", loc="left", fontsize=11, color=INK)
    for ax in axs:
        ax.set_xlabel("상자 옆을 지나가기까지 남은 시간 (s)", color=INK)
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        ax.grid(color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(sys.argv[2], facecolor="white")
    print("저장", sys.argv[2])
