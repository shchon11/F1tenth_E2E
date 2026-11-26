"""레이싱 T-미로 시험(exam_record.py ... tmaze) 재기 (11-26, RQ3/4/6). 분석만 -- 학습 손실이 아님.
  - b(L / R / 없음)와 통과 번호(1번째 / 2번째 이후)별 세게 닿음 비율 (통과 앞 2 s ~ 뒤 1 s 안 상자 hard 접촉)
  - 처음 보인 때 기준 tau = 2.0 ... 0 s 전(0.1 s 간격) 차의 옆 위치(부호 있음, 왼쪽 +; 같은 시험의 상자 없는 세션이 달린 길 =
    이 policy의 빈 트랙 길에서)와 속도, b별 -> A_exec(tau) = |E[옆|L] - E[옆|R]|, A_v(tau), 효과 크기 d = A / sigma_pooled,
    1번째 통과와 2번째 이후 따로
  - A_plan(tau): 그 결정의 plan이 상자의 트랙 따라 자리까지 닿을 때, 그 자리에서 plan의 옆 위치(같은 빈 트랙 길 기준)
  - 대조: 같은 자리(상자의 트랙 따라 자리 D m 앞)에서 옆 위치·속도 -- L과 R은 처음 보이는 때가 달라(같은 tau = 다른 자리) 시간 맞춤만으로는
    기억이 아닌 차이(제동 구간의 속도 차)가 d에 섞임
  - 세게 닿음은 상자에 다가간 번마다 셈: 통과 + 지나가지 못한 채 세게 닿은 번 (멈춰 선 차)
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
# [m] before the box's along-track place: the control for the time alignment -- L and R are first seen at different times
# (another place on the track for the same tau, in a braking zone a speed difference that is not memory); both boxes sit at
# the same along-track place, so the same D is the same place for both
DISTS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)
# [m] before the box: I(h; b) read at the same place (the control for the time-aligned one). A read is clean only before ANY
# part of the placement could be seen on this approach (`glimpse`): its first pass reads ~0 bit
H_DISTS = (6.0, 10.0, 14.0, 18.0)
GLIMPSE_M = 20.0             # [m] how far back along the approach the first glimpse of any part of the boxes is looked for


def seen_any(R, tk, sp, syaw, pts):
    """any of `pts` (k, 2) -- points on the boxes' outlines -- in direct view from a scanner at sp heading syaw (the exam's sight
    rule: field of view, range, no wall cell up to 0.05 m short of the point)"""
    d = pts - sp[None]; dist = np.linalg.norm(d, axis=1)
    a = (np.arctan2(d[:, 1], d[:, 0]) - syaw + np.pi) % (2 * np.pi) - np.pi
    m = (dist <= RMAX) & (np.abs(a) <= FOV)
    if not m.any():
        return False
    d, dist = d[m], dist[m]
    f = np.linspace(0.0, 1.0, 100)[None, :] * (np.maximum(dist - 0.05, 0.0) / np.maximum(dist, 1e-6))[:, None]
    seg = sp[None, None, :] + f[..., None] * d[:, None, :]
    return bool((R.wall_dist(tk, seg.reshape(-1, 2)).reshape(seg.shape[:2]) > 0.0).all(1).any())


def outline_pts(polys, step=0.1):
    out = []
    for q in polys:
        ring = np.concatenate([q, q[:1]])
        for a, b in zip(ring[:-1], ring[1:]):
            m = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
            out += [a + (b - a) * i / m for i in range(m)]
    return np.array(out)
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


def _logreg(X, y, C, iters=200):
    """multinomial logistic regression with an L2 weight 1 / (C n) on the weights (sklearn's convention), L-BFGS (torch, CPU)
    -> a function giving class probabilities (classes 0..K-1 of np.unique(y))"""
    import torch
    cls = np.unique(y); K = len(cls)
    yi = torch.as_tensor(np.searchsorted(cls, y)); Xt = torch.as_tensor(X, dtype=torch.float64)
    W = torch.zeros(X.shape[1], K, dtype=torch.float64, requires_grad=True)
    b0 = torch.zeros(K, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b0], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")
    n = X.shape[0]

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b0, yi) + 0.5 * (W ** 2).sum() / (C * n)
        loss.backward()
        return loss
    opt.step(closure)
    Wd, bd = W.detach(), b0.detach()
    return cls, (lambda Z: torch.softmax(torch.as_tensor(Z, dtype=torch.float64) @ Wd + bd, 1).numpy())


def _ce_bits(fit, Z, y):
    cls, f = fit
    p = f(Z)
    return -np.mean(np.log2(np.clip(p[np.arange(len(y)), np.searchsorted(cls, y)], 1e-9, 1.0)))


def info_bits(H, y, sess, seed=0):
    """I(h; y) lower bound [bit]: a logistic read-out trained on half the sessions (stratified by class), H(y) - held-out
    cross-entropy on the other half; the L2 weight chosen by 2-fold CV inside the training half"""
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
    mu, sd = H[m_tr].mean(0), H[m_tr].std(0) + 1e-6
    Xtr, Xte = (H[m_tr] - mu) / sd, (H[m_te] - mu) / sd
    best, bc = np.inf, 1.0
    s_tr = sess[m_tr]; u_tr = np.unique(s_tr); rng.shuffle(u_tr)
    fold = np.isin(s_tr, u_tr[: len(u_tr) // 2])
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        ce = 0.0
        for f in (fold, ~fold):
            if len(np.unique(y[m_tr][~f])) < len(np.unique(y)) or f.sum() == 0:
                ce = np.inf; break
            ce += _ce_bits(_logreg(Xtr[~f], y[m_tr][~f], C), Xtr[f], y[m_tr][f])
        if ce < best:
            best, bc = ce, C
    ce = _ce_bits(_logreg(Xtr, y[m_tr], bc), Xte, y[m_te])
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
    # a placement: one box [x, y, yaw, id] or a row of them -> its centre
    Lb, Rb = (z[k_].astype(float).reshape(-1, 4)[:, :2].mean(0) for k_ in ("tm_L", "tm_R"))
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
            # the same place: the car where its along-track progress is D m short of the box's (the last crossing before the pass)
            tgt = s_b + nmax[t + 1] * L
            k_lo = max(int(R.k0[min(t, T - 1), i]), t - 80)
            doff = np.full(len(DISTS), np.nan); dv = np.full(len(DISTS), np.nan); dj = np.full(len(DISTS), -1)
            for q, D_ in enumerate(DISTS):
                v_ = tgt - D_
                js = [j for j in range(t, k_lo - 1, -1) if prog[j, i] <= v_ < prog[j + 1, i]]
                if not js:
                    continue
                j = js[0]; f_ = (v_ - prog[j, i]) / max(prog[j + 1, i] - prog[j, i], 1e-9)
                pq = xy[j, i] + f_ * (xy[j + 1, i] - xy[j, i])
                doff[q] = project(P, pq[None], np.array([v_ % L]), L)[1][0]
                dv[q] = V[j, i] + f_ * (V[j + 1, i] - V[j, i])
                dj[q] = j
            h = mem[k_s - 10, i].astype(np.float32) if (mem is not None and k_s - 10 >= 0 and k_s - 10 >= R.k0[min(t, T - 1), i]) else None
            # the first glimpse of any part of the boxes on this approach (from the last pass, at most GLIMPSE_M back): how far
            # before the box's along-track place [m]; 0 = never in view before the pass
            glimpse = np.nan
            if b[i] < 2:
                pts = outline_pts(R.boxes(t, i))
                t_prev = max([r_["t"] for r_ in rows if r_["sess"] == i] + [int(R.k0[min(t, T - 1), i])])
                ks = [k_ for k_ in range(t_prev + 1, t + 1) if prog[k_, i] >= tgt - GLIMPSE_M]
                glimpse = 0.0
                for k_ in ks:
                    if seen_any(R, tk, xy[k_, i] + mount * np.array([np.cos(yaw[k_, i]), np.sin(yaw[k_, i])]), yaw[k_, i], pts):
                        glimpse = float(tgt - prog[k_, i]); break
            # (first pass) the first decision of the episode at which any part of the boxes was in view -- the scanner covers 270
            # deg, so a car that started past the box saw it behind it: its 'first pass' is not a first look
            k_seen = 10 ** 9
            if b[i] < 2 and npass == 1:
                pts = outline_pts(R.boxes(t, i))
                for k_ in range(int(R.k0[min(t, T - 1), i]), t + 1):
                    if seen_any(R, tk, xy[k_, i] + mount * np.array([np.cos(yaw[k_, i]), np.sin(yaw[k_, i])]), yaw[k_, i], pts):
                        k_seen = k_; break
            hD = {D_: (mem[int(dj[DISTS.index(D_)]), i].astype(np.float32) if (mem is not None and dj[DISTS.index(D_)] >= 0) else None)
                  for D_ in H_DISTS}
            rows.append(dict(sess=i, b=b[i], n=npass, t=t, k_s=k_s, see=(t - k_s) * dt, hit=hit, hit_any=hit_any, hD=hD, glimpse=glimpse, k_seen=k_seen, dj=dj,
                             off=off_c, v=v_c, offp=offp, doff=doff, dv=dv, h=h))
    # attempts at the box: every pass, and every hard box hit not followed by a pass within 1 s (the car stopped at the box); the
    # attempt's number = passes before it + 1
    att = {}
    for i in range(Bn):
        if b[i] >= 2:
            continue
        tp = sorted(r["t"] for r in rows if r["sess"] == i)
        for t in tp:
            att[(i, tp.index(t) + 1)] = 0.0
        for th in np.nonzero(hard_prop[:, i] > 0)[0]:
            n_before = sum(1 for t in tp if t < th - 10)
            att[(i, n_before + 1)] = 1.0
    return R, rows, dict(L=L, sb=sb, P=P, att=att, b=b)


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
        print(f"== {os.path.basename(fn)} · {z['name']} · {z['mode']} · 세션 {n_s}개 x {R.T * R.dt:.0f} s · L {np.round(z['tm_L'].reshape(-1, 4)[:, :2].mean(0), 3)} R {np.round(z['tm_R'].reshape(-1, 4)[:, :2].mean(0), 3)} "
              f"(자리마다 상자 {z['tm_L'].reshape(-1, 4).shape[0]}개)"
              + (f" · c {np.unique(np.round(z['cond'], 3))}" if "cond" in z.files else ""))
        print(f"상자 통과 {len(rows)}번 (상자 자리의 트랙 따라 위치: L {g['sb'][0]:.2f} m, R {g['sb'][1]:.2f} m / 한 바퀴 {g['L']:.2f} m)")
        P1 = lambda r: r["n"] == 1
        P2 = lambda r: r["n"] >= 2
        print("\n세게 닿음 비율: 상자에 다가간 번마다 (통과, 또는 지나가지 못한 채 세게 닿음; 몇 번째 = 그 전 통과 수 + 1) / 없음 = 통과 2 s 앞 ~ 1 s 뒤 아무 hard")
        print(f"{'b':<6}{'1번째':>18}{'2번째 이후':>18}{'처음 보임 중앙 (1 / 2+)':>26}")
        for k in range(3):
            cell = []
            for f in (P1, P2):
                if k < 2:
                    x = [v_ for (i_, n_), v_ in g["att"].items() if g["b"][i_] == k and f({"n": n_})]
                else:
                    x = [r["hit_any"] for r in rows if r["b"] == k and f(r)]
                cell.append(f"{100 * np.mean(x):5.1f} % ({len(x):4d})" if x else "-")
            se = [np.median([r["see"] for r in rows if r["b"] == k and f(r)]) if any(r["b"] == k and f(r) for r in rows) else np.nan
                  for f in (P1, P2)]
            print(f"{NAME_B[k]:<6}{cell[0]:>18}{cell[1]:>18}{se[0]:>12.2f} / {se[1]:.2f} s")
        # the first attempt where the car had seen nothing of the boxes until 10 m before them (a first look: reaction only)
        cl = {r["sess"]: bool(r["dj"][DISTS.index(10.0)] < r["k_seen"]) for r in rows if r["n"] == 1 and r["b"] < 2}
        for k in range(2):
            for lab, want in (("10 m 앞까지 안 보고 온 첫 번", True), ("그 전에 본 첫 번 (상자를 지나 출발 등)", False)):
                x = [v_ for (i_, n_), v_ in g["att"].items() if g["b"][i_] == k and n_ == 1 and cl.get(i_, False) == want]
                print(f"  {NAME_B[k]} {lab}: 세게 닿음 {100 * np.mean(x):5.1f} % ({len(x)})" if x else f"  {NAME_B[k]} {lab}: -")
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
        print("\n대조: 같은 자리 -- 상자의 트랙 따라 자리 D m 앞 (L·R 같은 자리라 시간 맞춤의 차이가 없음)")
        for lab, f in (("1번째 통과", P1), ("2번째 이후", P2)):
            print(f"  {lab}")
            print(f"  {'D m':>5}{'옆 L':>8}{'옆 R':>8}{'옆 없음':>8}{'A_exec':>8}{'d':>7}{'v L':>7}{'v R':>7}{'A_v':>7}{'d_v':>7}")
            for q, D_ in enumerate(DISTS):
                g_ = lambda k, key: np.array([r[key][q] for r in rows if r["b"] == k and f(r)], float)
                oL, oR, oN = g_(0, "doff"), g_(1, "doff"), g_(2, "doff")
                A, d = d_pooled(oL, oR)
                vL, vR = g_(0, "dv"), g_(1, "dv")
                Av, dv = d_pooled(vL, vR)
                mn = lambda a: np.nanmean(a) if np.isfinite(a).any() else np.nan
                print(f"  {D_:5.1f}{mn(oL):+8.3f}{mn(oR):+8.3f}{mn(oN):+8.3f}{A:8.3f}{d:7.2f}{mn(vL):7.2f}{mn(vR):7.2f}{Av:7.2f}{dv:7.2f}")
        print("\nI(h; b) [bit], 처음 보이기 1 s 전 기억 (세션 절반 학습 / 절반 시험, 시험 통과 수)")
        if not any(r["h"] is not None for r in rows) and not any(v_ is not None for r in rows for v_ in r["hD"].values()):
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
        print(f"상자 어느 부분이든 처음 보인 자리 (통과 앞 m, 지난 통과 뒤 {GLIMPSE_M:g} m 안; 그보다 앞에서 읽어야 1번째 통과가 ~0 bit):")
        for k in range(2):
            for lab, f in (("1번째", P1), ("2번째 이후", P2)):
                g_ = np.array([r["glimpse"] for r in rows if r["b"] == k and f(r)], float)
                if g_.size:
                    print(f"  {NAME_B[k]} {lab}: 중앙 {np.median(g_):.1f} m, 90 % {np.percentile(g_, 90):.1f} m, 최대 {g_.max():.1f} m ({g_.size}번)")
        print("I(h; b) [bit], 같은 자리 기억 (상자의 트랙 따라 자리 D m 앞 -- 시간 맞춤은 b마다 다른 자리를 읽어 '어디인지'가 섞임; 1번째 통과는 "
              "에피소드 시작부터 그 자리까지 상자 어느 부분도 안 보인 것만 -- 뒤로도 보는 270도 스캐너라 상자를 지나서 출발한 차는 이미 봄)")
        for k in range(2):
            n1 = [r for r in rows if r["b"] == k and r["n"] == 1]
            print(f"  {NAME_B[k]} 1번째 통과 {len(n1)}번 가운데 10 m 앞까지 안 보임 {sum(r['dj'][DISTS.index(10.0)] < r['k_seen'] for r in n1)}번")
        P1c = lambda r: r["n"] == 1 and (r["b"] == 2 or r["dj"][DISTS.index(D_)] < r["k_seen"])
        for D_ in H_DISTS:
            for lab, f in (("1번째 통과, 그때까지 안 보임", P1c), ("2번째 이후", P2)):
                rr = [r for r in rows if f(r) and r["hD"][D_] is not None]
                if len(rr) < 12:
                    print(f"  {D_:g} m 앞 {lab}: 통과 {len(rr)}번 -- 너무 적음"); continue
                H = np.stack([r["hD"][D_] for r in rr]); y = np.array([r["b"] for r in rr]); ss = np.array([r["sess"] for r in rr])
                i3, n3 = info_bits(H, y, ss)
                m2 = y < 2
                i2, n2 = info_bits(H[m2], y[m2], ss[m2]) if m2.sum() >= 12 else (np.nan, 0)
                print(f"  {D_:g} m 앞 {lab}: L/R/없음 {i3:+.3f} bit (시험 {n3}) | L/R {i2:+.3f} bit (시험 {n2})")


if __name__ == "__main__":
    main()
