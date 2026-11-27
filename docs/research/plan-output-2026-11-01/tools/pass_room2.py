"""(11-27) Room to pass, corridor version (pass_room.py's fixed-offset manoeuvres saw room before only 22 % of race33b u975's
real clean passes at 0.6x -- the manoeuvre family, not the track, was the limit). Here, for every 0.5 s where we are 0.6-5 m
behind the other car (along the track), against the other car's RECORDED next 6 s (it does not react):
  side room   at each 0.1 s, beside the other car (along its left / right normal, from its side outwards) the free width up to
              the wall (the record's distance field)
  our speed   beside it: our own speed at that point of the track when nothing is near (90th percentile per 0.25 m, decisions
              with the other car > 10 m away), at most sqrt(A_LAT / curvature) of a line NEED m to that side of the other car's,
              and at most our speed now + A_ACC t
  room        on one side, rel (our lead over the other car, centre to centre, along the track) goes from -gap to +0.8 m within
              6 s, and while -0.8 < rel < 0.8 (overlapping) the free width there is >= NEED all the time
NEED = 0.38 m: our 0.31 m width + 0.06 m to the other car + 0.01 m to the wall -- what race33b u975's real clean passes at 0.6x
kept (5th percentiles: circle-circle 0.371 m -> 0.06 m between the bodies; circle-wall 0.169 m -> 0.01 m).
Validated on the records' real clean passes (gap changes sign, no car contact / hard contact / at-fault charge from 3 s before
to 1 s after): the share of them with room seen 2-3 s before.  usage: pass_room2.py EXAM4.npz [...]"""
import sys

import numpy as np

A_LAT, A_ACC = 8.0, 4.0
NEED, HALF = 0.38, 0.155
K_MAX = 60
RAY = np.arange(HALF, 1.8, 0.0125)


def free_side(e, res, org, x, y, yaw, sgn):
    """(K,) free width beside the car on side sgn (+1 left), from its side to the first wall cell along its normal"""
    nx, ny = -np.sin(yaw) * sgn, np.cos(yaw) * sgn
    px = x[:, None] + nx[:, None] * RAY[None]; py = y[:, None] + ny[:, None] * RAY[None]
    c = np.floor((px - org[0]) / res).astype(np.int64); r = np.floor((py - org[1]) / res).astype(np.int64)
    ok = (c >= 0) & (c < e.shape[1]) & (r >= 0) & (r < e.shape[0])
    w = np.ones(px.shape, bool); w[ok] = e[r[ok], c[ok]] <= 0
    first = np.where(w.any(1), np.argmax(w, 1), RAY.size - 1)
    return RAY[first] - HALF


def setup(z):
    T = z["cfault"].shape[0]; B = z["c_pace"].shape[0]
    H, W = int(z["t_H"][0]), int(z["t_W"][0])
    e = z["edt"][:H * W].reshape(H, W); res = float(z["t_res"][0]); org = z["t_origin"][0].astype(float)
    L = float(z["lapL"][int(z["tid"][0, 0])])
    rem, orem = z["rem"].astype(float), z["orem"].astype(float)
    far = (np.abs(np.mod(rem[:-1] - orem[:-1] + 0.5 * L, L) - 0.5 * L) > 10.0)
    sv = z["st"][:T, :, 3].astype(float)[far]; sr = np.mod(rem[:-1][far], L)
    nb = int(np.ceil(L / 0.25)); vs = np.full(nb, np.nan); bi = np.clip((sr / 0.25).astype(int), 0, nb - 1)
    for k in range(nb):
        m = bi == k
        if m.sum() >= 5:
            vs[k] = np.percentile(sv[m], 90)
    vs = np.where(np.isfinite(vs), vs, np.nanmedian(vs))
    return dict(T=T, B=B, e=e, res=res, org=org, L=L, rem=rem, orem=orem, vs=vs, dt=float(z["dt"]))


def room(z, S, b, t):
    """True / False (room on some side within 6 s), or None when the window is not usable"""
    op, st, dt, L = z["op"], z["st"], S["dt"], S["L"]
    if t + K_MAX + 1 > op.shape[0] or z["ors"][t:t + K_MAX + 1, b].any():
        return None
    gap = np.mod(S["rem"][t, b] - S["orem"][t, b] + 0.5 * L, L) - 0.5 * L
    O = op[t:t + K_MAX + 1, b].astype(float)
    yaw = O[:, 2]; vo = O[:, 3]
    yr = np.gradient(np.unwrap(yaw)) / dt
    kap = yr / np.maximum(vo, 0.5)
    vtrack = S["vs"][np.clip((np.mod(S["orem"][t:t + K_MAX + 1, b], L) / 0.25).astype(int), 0, S["vs"].size - 1)]
    v0 = float(st[t, b, 3]); ks = np.arange(K_MAX + 1) * dt
    for sgn in (1, -1):
        fs = free_side(S["e"], S["res"], S["org"], O[:, 0], O[:, 1], yaw, sgn)
        lat = sgn * (2 * HALF + 0.06)
        ks_ = kap / np.clip(1 - kap * lat, 0.2, None)
        vu = np.minimum(np.minimum(vtrack, np.sqrt(A_LAT / np.maximum(np.abs(ks_), 1e-3))), v0 + A_ACC * ks)
        rel = -gap + np.concatenate([[0], np.cumsum((vu[:-1] - vo[:-1]) * dt)])
        done = np.nonzero(rel >= 0.8)[0]
        if not done.size:
            continue
        k1 = done[0]
        over = (rel[:k1 + 1] > -0.8)
        if (fs[:k1 + 1][over] >= NEED).all():
            return True
    return False


def clean_events(z, S):
    B, T = S["B"], S["T"]
    cnt = z["cnt"].astype(float); d = np.clip(np.diff(np.concatenate([np.zeros((1, B, 6)), cnt], 0), axis=0), 0, None)
    bad = d[:, :, 2] + d[:, :, 5] + d[:, :, 3] + d[:, :, 4] + (z["cfault"][:T] > 0)
    g = np.mod(S["rem"] - S["orem"] + 0.5 * S["L"], S["L"]) - 0.5 * S["L"]
    ev = []
    for b in range(B):
        for t in range(80, T - 80):
            if g[t - 1, b] > 0 >= g[t, b] and not bad[t - 30:t + 10, b].any() and not z["ors"][t - 80:t + 20, b].any():
                ev.append((b, t))
    return ev


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        z = np.load(fn); S = setup(z); pace = z["c_pace"][:S["B"]].astype(float); L = S["L"]
        print(f"== {fn.split('/')[-1]}", flush=True)
        ev = clean_events(z, S)
        for lo, hi, nm in ((0, 0.7, "0.6배"), (0.7, 0.9, "0.8배")):
            e_ = [x for x in ev if lo <= pace[x[0]] < hi]
            seen = sum(any(room(z, S, b, tt) for tt in (t - 30, t - 25, t - 20)) for b, t in e_)
            bs = np.nonzero((pace >= lo) & (pace < hi))[0]
            n_f = n_r = win = 0; ses = 0; did = []
            for b in bs:
                prev = False; anyr = False
                for t in range(30, S["T"] - K_MAX - 5, 5):
                    gap = np.mod(S["rem"][t, b] - S["orem"][t, b] + 0.5 * L, L) - 0.5 * L
                    r = room(z, S, b, t) if 0.6 <= gap <= 5.0 else None
                    if r is None:
                        prev = False; continue
                    n_f += 1
                    if r:
                        n_r += 1; anyr = True; win += not prev
                        g2 = np.mod(S["rem"][t + K_MAX, b] - S["orem"][t + K_MAX, b] + 0.5 * L, L) - 0.5 * L
                        did.append(g2 < -0.8)
                    prev = r
                ses += anyr
            print(f"  상대 {nm}: [검증] 실제 깔끔한 앞지름 {len(e_)}번 중 2-3 s 전에 자리를 본 것 {seen} ({100 * seen / max(1, len(e_)):.0f} %) | "
                  f"따라가던 순간 {n_f}, 자리 있던 순간 {n_r} ({100 * n_r / max(1, n_f):.0f} %) | 자리 있던 세션 {ses}/{len(bs)} | "
                  f"자리 구간 {win / len(bs):.1f}/세션 | 자리 있던 순간 뒤 6 s 안에 실제로 앞지름 {100 * np.mean(did) if did else 0:.0f} %", flush=True)
