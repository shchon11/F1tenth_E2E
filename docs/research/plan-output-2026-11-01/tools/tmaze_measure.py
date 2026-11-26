"""레이싱 T-미로 시험(exam_record.py ... tmaze) 재기 (11-26, RQ3/4/6). 분석만 -- 학습 손실이 아님.
  - b(L / R / 없음)와 통과 번호(1번째 / 2번째 이후)별 세게 닿음 비율 (통과 앞 2 s ~ 뒤 1 s 안 상자 hard 접촉)
  - 처음 보인 때 기준 tau = 2.0 ... 0 s 전(0.1 s 간격) 차의 옆 위치(부호 있음, 왼쪽 +; 같은 시험의 상자 없는 세션이 달린 길 =
    이 policy의 빈 트랙 길에서)와 속도, b별 -> A_exec(tau) = |E[옆|L] - E[옆|R]|, A_v(tau), 효과 크기 d = A / sigma_pooled,
    1번째 통과와 2번째 이후 따로
  - A_plan(tau): 그 결정의 plan이 상자의 트랙 따라 자리까지 닿을 때, 그 자리에서 plan의 옆 위치(같은 빈 트랙 길 기준)
  - I(h; b) [bit]: 처음 보이기 1 s 전 기억 h(_mem.npz의 mem, 그 결정 뒤 상태)에 로지스틱 read-out, 세션 절반으로 배우고 나머지 절반으로 잼
    (H(b) - 교차 엔트로피, 변분 하한). 1번째 / 2번째 이후, 세 갈래(L/R/없음)와 두 갈래(L/R)
처음 보임 = 상자 중심이 스캔에 직접 보이기 시작한 때 (exam_boxes.py와 같은 규칙: 시야각 135도, 10 m, 벽에 안 가림). 상자 없는 세션은 L·R
가운데 자리(빈 트랙 길 위 점)를 가상의 상자로 봄.
인자: 시험npz [더 ...]  (같은 이름의 _mem.npz가 옆에 있어야 I(h;b)를 잼)"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rec_lib import Rec, along_track

TAUS = np.round(np.arange(0.0, 2.0001, 0.1), 2)
FOV, RMAX = np.radians(135.0), 10.0
BIN = 0.1                    # [m] the reference line's along-track bins
NAME_B = ("L", "R", "없음")


def ref_line(xy, s, ok, L):
    """(n, 2) the policy's own empty-track line: per BIN of along-track position, the median of the positions there"""
    nb = int(np.ceil(L / BIN))
    k = np.floor(s[ok] / BIN).astype(int) % nb
    P = np.full((nb, 2), np.nan)
    for j in range(nb):
        m = k == j
        if m.sum() >= 3:
            P[j] = np.median(xy[ok][m], axis=0)
    good = np.isfinite(P[:, 0])
    if good.mean() < 0.8:                                  # the policy does not drive the lap: no line to measure against
        return None
    idx = np.arange(nb)
    for c in range(2):                                     # fill the gaps around the lap
        P[:, c] = np.interp(idx, np.concatenate([idx[good] - nb, idx[good], idx[good] + nb]),
                            np.concatenate([P[good, c]] * 3))
    ker = np.ones(5) / 5.0                                 # 0.5 m moving average, wrapped
    P = np.stack([np.convolve(np.concatenate([P[-2:, c], P[:, c], P[:2, c]]), ker, "valid") for c in range(2)], 1)
    return P


def project(P, pts, s_guess, L, win=3.0):
    """each point (n, 2) onto the line near its along-track guess (n,) [m]: (along-track s, signed offset, left +)"""
    nb = P.shape[0]
    w = int(round(win / BIN))
    j0 = np.floor(s_guess / BIN).astype(int)
    js = (j0[:, None] + np.arange(-w, w + 1)[None, :]) % nb                 # (n, 2w+1)
    A, B = P[js], P[(js + 1) % nb]
    D = B - A; Lq = (D ** 2).sum(-1).clip(min=1e-12)
    u = (((pts[:, None] - A) * D).sum(-1) / Lq).clip(0.0, 1.0)
    F = A + u[..., None] * D
    d = np.linalg.norm(pts[:, None] - F, axis=-1)
    i = d.argmin(1); r = np.arange(len(pts))
    Ds = D[r, i] / np.sqrt(Lq[r, i])[:, None]
    n = np.stack([-Ds[:, 1], Ds[:, 0]], 1)
    off = ((pts - F[r, i]) * n).sum(1)
    s = ((js[r, i] + u[r, i]) * BIN) % L
    return s, off


def wrap(x, L):
    return np.mod(x + 0.5 * L, L) - 0.5 * L


def d_pooled(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    A = abs(a.mean() - b.mean())
    return A, A / sp if sp > 0 else np.nan


def info_bits(H, y, sess, seed=0):
    """I(h; y) lower bound [bit]: a logistic read-out trained on half the sessions (stratified by class), H(y) - held-out
    cross-entropy on the other half; the L2 weight chosen by 2-fold CV inside the training half"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    us = np.unique(sess)
    cls_of = {s_: y[sess == s_][0] for s_ in us}
    tr = set()
    for c in np.unique(y):
        ss = np.array([s_ for s_ in us if cls_of[s_] == c]); rng.shuffle(ss)
        tr.update(ss[: len(ss) // 2].tolist())
    m_tr = np.array([s_ in tr for s_ in sess]); m_te = ~m_tr
    if len(np.unique(y[m_tr])) < len(np.unique(y)) or m_te.sum() < 4:
        return np.nan, 0
    sc = StandardScaler().fit(H[m_tr])
    Xtr, Xte = sc.transform(H[m_tr]), sc.transform(H[m_te])
    best, bc = np.inf, 1.0
    s_tr = sess[m_tr]; u_tr = np.unique(s_tr); rng.shuffle(u_tr)
    fold = np.isin(s_tr, u_tr[: len(u_tr) // 2])
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        ce = 0.0
        for f in (fold, ~fold):
            if len(np.unique(y[m_tr][~f])) < len(np.unique(y)) or f.sum() == 0:
                ce = np.inf; break
            lr = LogisticRegression(C=C, max_iter=2000).fit(Xtr[~f], y[m_tr][~f])
            p = lr.predict_proba(Xtr[f])
            col = {c: i for i, c in enumerate(lr.classes_)}
            ce += -np.mean(np.log2(np.clip(p[np.arange(f.sum()), [col[c] for c in y[m_tr][f]]], 1e-9, 1.0)))
        if ce < best:
            best, bc = ce, C
    lr = LogisticRegression(C=bc, max_iter=2000).fit(Xtr, y[m_tr])
    p = lr.predict_proba(Xte)
    col = {c: i for i, c in enumerate(lr.classes_)}
    ce = -np.mean(np.log2(np.clip(p[np.arange(m_te.sum()), [col[c] for c in y[m_te]]], 1e-9, 1.0)))
    pri = np.array([np.mean(y[m_te] == c) for c in np.unique(y[m_te])])
    Hy = -(pri * np.log2(pri)).sum()
    return Hy - ce, int(m_te.sum())


def one(fn):
    R = Rec(fn); z = R.z; T, Bn, dt = R.T, R.B, R.dt
    assert "tm_b" in z.files, f"{fn}: not a tmaze exam (exam_record.py ... tmaze)"
    b = z["tm_b"].astype(int)
    st = R.st.astype(float); xy = st[:, :, :2]; yaw = st[:, :, 2]; V = np.abs(st[:, :, 3])
    s, lap, L = along_track(z, T + 1)
    prog = s + lap * L                                     # (T+1, B) unwrapped along-track [m]
    mount = float(z["lidar_mount_x"]) if float(z["lidar_mount_x"]) > 0 else 0.297
    tk = int(R.tid[0, 0])
    Lb, Rb = z["tm_L"].astype(float), z["tm_R"].astype(float)
    # this policy's own empty-track line: the sessions without a box, after their first 3 s
    none = b == 2
    okr = np.zeros((T + 1, Bn), bool); okr[int(round(3.0 / dt)):, none] = True
    okr &= V > 0.5
    P = ref_line(xy[okr], s[okr], np.ones(int(okr.sum()), bool), L) if okr.any() else None
    if P is None:
        return R, [], None
    # the boxes' along-track places (each on the line; the empty sessions take the middle of L and R)
    box_xy = {0: Lb[:2], 1: Rb[:2], 2: 0.5 * (Lb[:2] + Rb[:2])}
    sb = {}
    for k_, c_ in box_xy.items():
        d_ = np.linalg.norm(P - c_, axis=1); j_ = int(d_.argmin())
        sb[k_] = float(project(P, c_[None], np.array([j_ * BIN]), L)[0][0])
    hard_prop = R.d[:, :, 4]; hard_any = R.d[:, :, 3:6].sum(-1)
    W = int(round(3.0 / dt))
    rows = []                                              # one per pass
    mem = None
    mf = fn[:-4] + "_mem.npz"
    if os.path.exists(mf):
        mem = np.load(mf)["mem"]                           # (T, B, memory): the state after each decision
    for i in range(Bn):
        c_ = box_xy[b[i]]; s_b = sb[b[i]]
        n0 = np.floor((prog[0, i] - s_b) / L)
        nmax = np.maximum.accumulate(np.floor((prog[:, i] - s_b) / L))
        ev = np.nonzero(np.diff(nmax) > 0)[0]             # the pass happens during decision t
        for kp, t in enumerate(ev):
            npass = int(nmax[t + 1] - n0)
            if t < 2:
                continue
            # first sight: going back from the pass, the visible stretch that ends at it
            def sees(k):
                sx, sy = xy[k, i, 0] + mount * np.cos(yaw[k, i]), xy[k, i, 1] + mount * np.sin(yaw[k, i])
                dx, dy = c_[0] - sx, c_[1] - sy; dist = np.hypot(dx, dy)
                a = (np.arctan2(dy, dx) - yaw[k, i] + np.pi) % (2 * np.pi) - np.pi
                if dist > RMAX or abs(a) > FOV:
                    return False
                n = max(int((dist - 0.25) / 0.05), 1)
                f = np.linspace(0.0, max(dist - 0.25, 0.0) / max(dist, 1e-6), n)
                return bool((R.wall_dist(tk, np.stack([sx + f * dx, sy + f * dy], 1)) > 0.0).all())
            k = t
            while k > 0 and not sees(k) and t - k < 5:
                k -= 1
            if not sees(k):
                k_s = t                                    # never in view before it: sight = the pass
            else:
                while k > 0 and sees(k - 1) and R.k0[k - 1, i] == R.k0[t, i]:
                    k -= 1
                k_s = k
            w0, w1 = max(t - 20, 0), min(t + 10, T - 1)
            hit = float(hard_prop[w0:w1 + 1, i].sum() > 0) if b[i] < 2 else np.nan
            hit_any = float(hard_any[w0:w1 + 1, i].sum() > 0)
            ks = k_s - np.round(TAUS / dt).astype(int)
            okk = (ks >= 0) & (ks >= R.k0[min(t, T - 1), i])
            ksc = np.clip(ks, 0, T)
            s_c, off_c = project(P, xy[ksc, i], s[ksc, i], L)
            off_c = np.where(okk, off_c, np.nan); v_c = np.where(okk, V[ksc, i], np.nan)
            # the plan drawn at each of those decisions, at the box's along-track place (if it gets there)
            offp = np.full(len(TAUS), np.nan)
            for q, kk in enumerate(ksc):
                if not okk[q] or kk >= T:
                    continue
                pl = R.pl[kk, i].astype(float)            # (10, 2) the plan's points in the world
                pts = np.concatenate([xy[kk, i][None], pl], 0)
                seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
                sp, op = project(P, pts, s[kk, i] + seg, L)
                rel = wrap(sp - s_b, L)
                j = np.nonzero((rel[:-1] < 0) & (rel[1:] >= 0))[0]
                if len(j):
                    j = j[0]; f_ = -rel[j] / max(rel[j + 1] - rel[j], 1e-9)
                    offp[q] = op[j] + f_ * (op[j + 1] - op[j])
            h = mem[k_s - 10, i].astype(np.float32) if (mem is not None and k_s - 10 >= 0 and k_s - 10 >= R.k0[min(t, T - 1), i]) else None
            rows.append(dict(sess=i, b=b[i], n=npass, t=t, k_s=k_s, see=(t - k_s) * dt, hit=hit, hit_any=hit_any,
                             off=off_c, v=v_c, offp=offp, h=h))
    return R, rows, dict(L=L, sb=sb, P=P)


def main():
    files = sys.argv[1:]
    for fn in files:
        R, rows, g = one(fn)
        z = R.z
        n_s = len(z["tm_b"])
        if g is None:
            print(f"== {os.path.basename(fn)} · {z['name']}: 상자 없는 세션이 한 바퀴의 80 % 넘게 달리지 못해(0.5 m/s 넘게) 빈 트랙 길이 "
                  f"없음 -> 통과·옆 위치를 잴 수 없음 (세션 {n_s}개, 평균 속도 {np.abs(R.st[:, :, 3]).mean():.2f} m/s)")
            continue
        print(f"== {os.path.basename(fn)} · {z['name']} · {z['mode']} · 세션 {n_s}개 x {R.T * R.dt:.0f} s · L {z['tm_L'][:2]} R {z['tm_R'][:2]}"
              + (f" · c {np.unique(np.round(z['cond'], 3))}" if "cond" in z.files else ""))
        print(f"상자 통과 {len(rows)}번 (상자 자리의 트랙 따라 위치: L {g['sb'][0]:.2f} m, R {g['sb'][1]:.2f} m / 한 바퀴 {g['L']:.2f} m)")
        P1 = lambda r: r["n"] == 1
        P2 = lambda r: r["n"] >= 2
        print("\n세게 닿음 비율 (통과 2 s 앞 ~ 1 s 뒤 상자 hard; 없음 = 아무 hard)")
        print(f"{'b':<6}{'1번째':>18}{'2번째 이후':>18}{'처음 보임 중앙 (1 / 2+)':>26}")
        for k in range(3):
            cell = []
            for f in (P1, P2):
                rr = [r for r in rows if r["b"] == k and f(r)]
                x = [r["hit"] if k < 2 else r["hit_any"] for r in rr]
                cell.append(f"{100 * np.mean(x):5.1f} % ({len(x):4d})" if x else "-")
            se = [np.median([r["see"] for r in rows if r["b"] == k and f(r)]) if any(r["b"] == k and f(r) for r in rows) else np.nan
                  for f in (P1, P2)]
            print(f"{NAME_B[k]:<6}{cell[0]:>18}{cell[1]:>18}{se[0]:>12.2f} / {se[1]:.2f} s")
        print("\n처음 보이기 tau s 전: 옆 위치(빈 트랙 길에서, 왼쪽 +) 평균 [m] b별, A_exec = |L-R|, d = A/sigma; 속도 A_v, d_v; plan A_plan, d_plan")
        for lab, f in (("1번째 통과", P1), ("2번째 이후", P2)):
            print(f"  {lab}")
            print(f"  {'tau':>5}{'옆 L':>8}{'옆 R':>8}{'옆 없음':>8}{'A_exec':>8}{'d':>7}{'v L':>7}{'v R':>7}{'A_v':>7}{'d_v':>7}{'A_plan':>8}{'d_plan':>8}{'n plan':>8}")
            for q, tau in enumerate(TAUS):
                g_ = lambda k, key: np.array([r[key][q] for r in rows if r["b"] == k and f(r)], float)
                oL, oR, oN = g_(0, "off"), g_(1, "off"), g_(2, "off")
                A, d = d_pooled(oL, oR)
                vL, vR = g_(0, "v"), g_(1, "v")
                Av, dv = d_pooled(vL, vR)
                pL, pR = g_(0, "offp"), g_(1, "offp")
                Ap, dp = d_pooled(pL, pR)
                npl = int(np.isfinite(pL).sum() + np.isfinite(pR).sum())
                mn = lambda a: np.nanmean(a) if np.isfinite(a).any() else np.nan
                print(f"  {tau:5.1f}{mn(oL):+8.3f}{mn(oR):+8.3f}{mn(oN):+8.3f}{A:8.3f}{d:7.2f}{mn(vL):7.2f}{mn(vR):7.2f}{Av:7.2f}{dv:7.2f}"
                      f"{Ap:8.3f}{dp:8.2f}{npl:8d}")
        print("\nI(h; b) [bit], 처음 보이기 1 s 전 기억 (세션 절반 학습 / 절반 시험, 시험 통과 수)")
        if not any(r["h"] is not None for r in rows):
            print("  기억 기록(_mem.npz) 없음")
            continue
        for lab, f in (("1번째 통과", P1), ("2번째 이후", P2)):
            rr = [r for r in rows if f(r) and r["h"] is not None]
            if len(rr) < 12:
                print(f"  {lab}: 통과 {len(rr)}번 -- 너무 적음"); continue
            H = np.stack([r["h"] for r in rr]); y = np.array([r["b"] for r in rr]); ss = np.array([r["sess"] for r in rr])
            i3, n3 = info_bits(H, y, ss)
            m2 = y < 2
            i2, n2 = info_bits(H[m2], y[m2], ss[m2]) if m2.sum() >= 12 else (np.nan, 0)
            print(f"  {lab}: L/R/없음 {i3:+.3f} bit (최대 {np.log2(3):.2f}, 시험 {n3}) | L/R {i2:+.3f} bit (최대 1, 시험 {n2})")


if __name__ == "__main__":
    main()
