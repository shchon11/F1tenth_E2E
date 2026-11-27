"""(11-27, the user: "분명히 추월할 수 있을 것 같은데 안하는게 문제고 조심하는건 죄가 아니지") Is there ROOM to pass cleanly where a
policy followed? For every 0.5 s of an exam4 session where we are 0.6-5 m behind the other car (along the track), a search over
simple passing manoeuvres against the other car's RECORDED future (it does not react to the manoeuvre -- an optimistic bound
for a defending car, a fair one for this exam's reference car, which drives its own line):
  path   the other car's own driven line (3 s back .. 7 s ahead, 0.05 m arc-length grid) moved sideways by l(s): from our offset
         now to D in R metres (smoothstep), then held; D in -1.1..1.1 m (23), R in {2, 3.5, 5} m
  speed  our own speed at that point of the track when nothing is near (the record's 90th percentile per 0.25 m, decisions with
         the other car > 10 m away), at most sqrt(A_LAT / |curvature|) of
         the moved line (incl. l''), from our speed now with at most +A_ACC / -A_BRK (forward-backward passes)
  clean  every 0.1 s until 1 s after we are 0.8 m ahead along the line: our 3 circles (r 0.16 m at -0.2/0/+0.2 m) keep
         >= 0.005 m from the walls (the record's distance field) and >= 0.05 m from the other car's 3 circles -- as close as
         race33b u975's real clean passes at 0.6x came (5th percentiles; median 0.215 m wall, 0.50 m circle-circle)
  room   some manoeuvre is clean and we are 0.8 m ahead within 6 s
Limits measured from the same record (98th percentiles): lateral 8.1, accel 4.4, brake 4.3 m/s^2 -> A_LAT 8.0, A_ACC 4.0, A_BRK 4.0.
Prints per opponent pace: following samples, the share with room, sessions with room at least once, room windows per session,
and what the policy did after a room sample (passed within 6 s or not).  usage: pass_room.py EXAM4.npz [...]"""
import sys

import numpy as np

A_LAT, A_ACC, A_BRK = 8.0, 4.0, 4.0
R_C, OFF_C = 0.16, np.array([-0.2, 0.0, 0.2])
WALL_M, CAR_M = 0.005, 0.05      # (calibrated) the 5th percentiles of race33b u975's real clean passes at 0.6x: circle-circle 0.371 m, circle-wall 0.169 m
DS, K_MAX, K_AFTER = 0.05, 60, 10
DS_ = np.linspace(-1.1, 1.1, 23); RS = (2.0, 3.5, 5.0)


def smooth(a, n=7):
    k = np.ones(n) / n
    return np.convolve(np.pad(a, (n // 2, n // 2), mode="edge"), k, mode="valid")


def wall_ok(e, res, org, x, y):
    c = np.floor((x - org[0]) / res).astype(np.int64); r = np.floor((y - org[1]) / res).astype(np.int64)
    ok = (c >= 0) & (c < e.shape[1]) & (r >= 0) & (r < e.shape[0])
    d = np.zeros(x.shape); d[ok] = e[r[ok], c[ok]]
    return d >= R_C + WALL_M


def room(z, e, res, org, b, t, pace):
    op, st = z["op"], z["st"]
    t0, t1 = max(0, t - 30), min(op.shape[0], t + K_MAX + K_AFTER + 1)
    P = op[t0:t1, b, :2].astype(float); V = op[t0:t1, b, 3].astype(float)
    seg = np.hypot(*np.diff(P, axis=0).T); s_rec = np.concatenate([[0], np.cumsum(seg)])
    if s_rec[-1] < 5:
        return "skip"
    s = np.arange(0, s_rec[-1], DS)
    X = np.interp(s, s_rec, P[:, 0]); Y = np.interp(s, s_rec, P[:, 1])
    X, Y = smooth(X, 9), smooth(Y, 9)
    th = np.unwrap(np.arctan2(np.gradient(Y), np.gradient(X)))
    kap = smooth(np.gradient(th) / DS, 9)
    # our own speed there when nothing is near (VSOLO: the record's 90th percentile per 0.25 m of track, from decisions with the
    # other car > 10 m away); the other car's speed / its pace where that has no samples
    rr = z["orem"][t0:t1, b].astype(float)
    vs = VSOLO[np.clip((np.mod(rr, LAP) / 0.25).astype(int), 0, VSOLO.size - 1)]
    vref = np.interp(s, s_rec, np.where(np.isfinite(vs), vs, V / max(pace, 0.3)))
    nx, ny = -np.sin(th), np.cos(th)
    # our start: nearest grid point, signed offset
    ux, uy, uyaw, uv = st[t, b].astype(float)
    i0 = int(np.argmin((X - ux) ** 2 + (Y - uy) ** 2))
    l0 = (ux - X[i0]) * nx[i0] + (uy - Y[i0]) * ny[i0]
    io = int(np.argmin((X - op[t, b, 0]) ** 2 + (Y - op[t, b, 1]) ** 2))
    if i0 >= io:
        return "skip"
    s_op = np.interp(np.arange(t, min(t1, t + K_MAX + K_AFTER + 1)) - t0, np.arange(t1 - t0), s_rec)   # the other car's s per 0.1 s
    s_op = s_op - s_rec[0] + 0.0
    O = op[t:t + len(s_op), b].astype(float)
    S = s[i0:]; n = S.size
    u = (S - S[0])
    best = None
    for R in RS:
        w = np.clip(u / R, 0, 1); sm = w * w * (3 - 2 * w)
        d1 = np.where((u > 0) & (u < R), 6 * w * (1 - w) / R, 0.0); d2 = np.where((u > 0) & (u < R), (6 - 12 * w) / R ** 2, 0.0)
        D = DS_[:, None]
        L = l0 + (D - l0) * sm[None]; L1 = (D - l0) * d1[None]; L2 = (D - l0) * d2[None]
        kk = kap[i0:][None]
        kt = kk / np.clip(1 - kk * L, 0.2, None) + L2
        cap = np.minimum(np.sqrt(A_LAT / np.maximum(np.abs(kt), 1e-3)), vref[i0:][None])
        v = cap.copy()
        for i in range(n - 2, -1, -1):
            v[:, i] = np.minimum(v[:, i], np.sqrt(v[:, i + 1] ** 2 + 2 * A_BRK * DS))
        v[:, 0] = np.minimum(v[:, 0], max(uv, 0.3))
        for i in range(1, n):
            v[:, i] = np.minimum(v[:, i], np.sqrt(v[:, i - 1] ** 2 + 2 * A_ACC * DS))
        tt = np.concatenate([np.zeros((v.shape[0], 1)), np.cumsum(DS / np.maximum(v[:, 1:], 0.05), axis=1)], 1)
        K = len(s_op)
        ks = np.arange(K) * 0.1
        for j in range(v.shape[0]):
            sj = np.interp(ks, tt[j], S, right=np.nan)
            if np.isnan(sj).all():
                continue
            ok_k = ~np.isnan(sj)
            idx = np.clip(((sj[ok_k] - S[0]) / DS).astype(int), 0, n - 1)
            gi = i0 + idx
            px = X[gi] + L[j, idx] * nx[gi]; py = Y[gi] + L[j, idx] * ny[gi]
            hd = th[gi] + np.arctan(L1[j, idx])
            cx = px[:, None] + OFF_C[None] * np.cos(hd)[:, None]; cy = py[:, None] + OFF_C[None] * np.sin(hd)[:, None]
            wok = wall_ok(e, res, org, cx, cy).all(1)
            Ok = O[:len(px)]
            ox = Ok[:, 0, None] + OFF_C[None] * np.cos(Ok[:, 2])[:, None]; oy = Ok[:, 1, None] + OFF_C[None] * np.sin(Ok[:, 2])[:, None]
            dd = np.hypot(cx[:, :, None] - ox[:, None, :], cy[:, :, None] - oy[:, None, :]).min((1, 2))
            cok = dd >= 2 * R_C + CAR_M
            ahead = (sj[ok_k] - (s_op[:len(px)] + 0.0)) >= 0.8
            fail = ~(wok & cok)
            first_bad = np.argmax(fail) if fail.any() else len(px)
            hit = np.nonzero(ahead[:K_MAX + 1])[0]
            if hit.size and hit[0] + K_AFTER < first_bad and hit[0] + K_AFTER < len(px):
                if best is None or hit[0] < best:
                    best = int(hit[0])
    return best


for fn in sys.argv[1:]:
    z = np.load(fn)
    dt = float(z["dt"]); T = z["cfault"].shape[0]; B = z["c_pace"].shape[0]
    H, W = int(z["t_H"][0]), int(z["t_W"][0]); e = z["edt"][:H * W].reshape(H, W); res = float(z["t_res"][0]); org = z["t_origin"][0]
    L = float(z["lapL"][int(z["tid"][0, 0])])
    pace = z["c_pace"][:B].astype(float); ors = z["ors"][:T]
    rem, orem = z["rem"].astype(float), z["orem"].astype(float)
    print(f"== {fn.split('/')[-1]}", flush=True)
    LAP = L
    far = (np.abs(np.mod(rem[:-1] - orem[:-1] + 0.5 * L, L) - 0.5 * L) > 10.0)
    sv = z["st"][:T, :, 3].astype(float)[far]; sr = np.mod(rem[:-1][far], L)
    nb = int(np.ceil(L / 0.25)); VSOLO = np.full(nb, np.nan)
    bi = np.clip((sr / 0.25).astype(int), 0, nb - 1)
    for k in range(nb):
        m = bi == k
        if m.sum() >= 5:
            VSOLO[k] = np.percentile(sv[m], 90)
    for lo, hi, nm in ((0.7, 0.9, "0.8배"), (0, 0.7, "0.6배")):
        bs = np.nonzero((pace >= lo) & (pace < hi))[0]
        n_f = n_r = 0; ses_r = 0; win = 0; did = []
        for b in bs:
            any_r = False; prev = False
            for t in range(30, T - K_MAX - K_AFTER - 5, 5):
                if ors[max(0, t - 30):t + K_MAX + K_AFTER + 1, b].any():
                    prev = False; continue
                gap = np.mod(rem[t, b] - orem[t, b] + 0.5 * L, L) - 0.5 * L       # + = the other car ahead
                if not (0.6 <= gap <= 5.0):
                    prev = False; continue
                r = room(z, e, res, (float(org[0]), float(org[1])), b, t, pace[b])
                if isinstance(r, str):
                    prev = False; continue
                n_f += 1
                has = r is not None
                if has:
                    n_r += 1; any_r = True
                    if not prev:
                        win += 1
                    g2 = np.mod(rem[t + K_MAX, b] - orem[t + K_MAX, b] + 0.5 * L, L) - 0.5 * L
                    did.append(g2 < -0.8)
                prev = has
            ses_r += any_r
        print(f"  상대 {nm}: 따라가던 순간 {n_f} | 깔끔하게 앞지를 자리가 있던 순간 {n_r} ({100 * n_r / max(1, n_f):.0f} %) | "
              f"자리가 한 번이라도 있던 세션 {ses_r}/{len(bs)} | 자리 구간 {win / len(bs):.2f}/세션 | "
              f"자리 있던 순간 뒤 6 s 안에 실제로 앞지름 {100 * np.mean(did) if did else 0:.0f} %", flush=True)
