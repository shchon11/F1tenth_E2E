"""Audit C: competition-format scoring of exam_record.py records (independent of exam_table.py).
Qualifier (solo, every session boxes until rm_t, then removed):
  best lap [s] (fastest complete lap of the session, flying), best lap after removal, median lap,
  longest streak of consecutive complete laps without a HARD contact / without ANY contact,
  contacts per lap (wall / box, any onset and hard), stalled time (2 s window < 1 m along track),
  DNF = restarted by the env, or stuck >= 10 s at a stretch, or >= 5 m driven facing backwards.
Race (1v1): own along-track progress from own start; winner = first to M laps (M*L metres of own progress) within the
  session, else more progress at the end. Passes (relative along-track gap crossing +-1 m), car contacts and who was
  behind at the contact onset (ego behind by > 0.3 m = ego at fault; ahead by > 0.3 m = rear-ended; else side by side),
  time stuck close behind (0.5-4 m behind, both moving).
Every rate: bootstrap 95 % interval over sessions clustered by box layout (session % 64), 2000 draws.
Args: out.json rec.npz [rec.npz ...]   env C_M (laps to win, default 10)"""
import sys, json
import numpy as np

RNG = np.random.default_rng(0)
M_LAPS = int(__import__("os").environ.get("C_M", "10"))


def boot(vals, clusters, stat=np.mean, n=2000):
    v = np.asarray(vals, float); cl = np.asarray(clusters)
    ok = np.isfinite(v); v, cl = v[ok], cl[ok]
    if len(v) == 0:
        return [float("nan")] * 3
    u = np.unique(cl); idx = {c: np.nonzero(cl == c)[0] for c in u}
    out = []
    for _ in range(n):
        pick = RNG.choice(u, len(u))
        s = np.concatenate([idx[c] for c in pick])
        out.append(stat(v[s]))
    return [float(stat(v)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def progress(rem, L):
    r = rem.astype(float).copy()
    for t in range(1, r.shape[0]):
        bad = ~np.isfinite(r[t]); r[t, bad] = r[t - 1, bad]
    r[~np.isfinite(r)] = 0.0
    d = r[:-1] - r[1:]
    d = np.mod(d + 0.5 * L, L) - 0.5 * L
    return np.concatenate([np.zeros((1, r.shape[1])), np.cumsum(d, 0)], 0)


def new_counts(cnt, rs):
    prev = np.concatenate([np.zeros((1,) + cnt.shape[1:], np.float32), cnt[:-1]], 0)
    prev[1:][rs[:-1]] = 0.0
    return np.clip(cnt - prev, 0, None) * (~rs)[..., None]


def laps_from_signal(lapsig, b, dt):
    net = np.cumsum(lapsig[:, b]); hi, t_prev, out = 0, None, []
    for t in np.nonzero(lapsig[:, b] > 0)[0]:
        if net[t] > hi:
            hi = net[t]
            if t_prev is not None:
                out.append((t_prev, t + 1))          # decision indices [start, end)
            t_prev = t + 1
    return out


def laps_from_progress(lapsig, Pb, L, dt):
    """laps between the times the along-track progress crosses P0 + k L (P0 = progress at the first start-line crossing),
    linearly interpolated inside a decision -> lap times not quantised to the 0.1 s decision; returns decision ranges
    [s, e) covering each lap (for contact counting) and the lap times [s]"""
    first = np.nonzero(lapsig > 0)[0]
    if len(first) == 0:
        return [], []
    k0 = first[0] + 1; P0 = Pb[k0]
    marks = []
    kk = 1
    t_prev = k0 * dt
    out_t = [t_prev]
    for t in range(k0 + 1, len(Pb)):
        while Pb[t] >= P0 + kk * L and Pb[t - 1] < P0 + kk * L:
            f = (P0 + kk * L - Pb[t - 1]) / max(Pb[t] - Pb[t - 1], 1e-9)
            out_t.append((t - 1 + f) * dt); kk += 1
    laps, lt = [], []
    for a, b_ in zip(out_t[:-1], out_t[1:]):
        laps.append((int(np.floor(a / dt)), int(np.ceil(b_ / dt)))); lt.append(b_ - a)
    return laps, lt


def stuck_mask(P, dt, n_win=20, m=1.0):
    T = P.shape[0] - 1
    adv = np.full((T, P.shape[1]), np.inf)
    adv[:T - n_win] = P[n_win:T] - P[:T - n_win]
    s = adv < m; s[:10] = False
    # mark the whole window as stuck
    out = s.copy()
    for k in range(1, n_win):
        out[k:] |= s[:-k]
    return out


def longest_run(mask):
    best = np.zeros(mask.shape[1], int); cur = np.zeros(mask.shape[1], int)
    for t in range(mask.shape[0]):
        cur = np.where(mask[t], cur + 1, 0); best = np.maximum(best, cur)
    return best


def backwards_m(st, rem, rs, L):
    T = rs.shape[0]
    vx = st[:T, :, 3].astype(float)
    r = rem[:T + 1].astype(float).copy()
    for t in range(1, r.shape[0]):
        bad = ~np.isfinite(r[t]); r[t, bad] = r[t - 1, bad]
    dr = r[1:] - r[:-1]; dr = np.mod(dr + 0.5 * L, L) - 0.5 * L; dr[~np.isfinite(dr)] = 0
    dr[rs[:T].astype(bool)] = 0
    return np.where((vx > 0.2) & (dr > 0), dr, 0.0).sum(0)


def score(fn):
    z = np.load(fn, allow_pickle=False)
    dt = float(z["dt"]); rs = z["rs"].astype(bool); T, B = rs.shape
    rm_t = float(z["rm_t"]); k_rm = int(round(rm_t / dt))
    L = float(z["lapL"][int(z["tid"][0, 0])])
    st = z["st"].astype(float); sid = z["sid"]; lapsig = z["lap"]
    box = (sid[0] >= 0).any(-1)
    on = z["on"][:T]; opp = on.any(0)
    d = new_counts(z["cnt"].astype(np.float32), rs)               # (T,B,6) onsets w/p/c, hard w/p/c
    P = progress(z["rem"], L)
    stuck = stuck_mask(P, dt)
    bw = backwards_m(st, z["rem"], rs, L)
    cluster = np.arange(B) % 64
    S = []
    for b in range(B):
        laps, lt = laps_from_progress(lapsig[:, b], P[:, b], L, dt)
        hard_l = [float(d[s:e, b, 3:5].sum()) for s, e in laps]
        any_l = [float(d[s:e, b, 0:2].sum()) for s, e in laps]
        car_hard_l = [float(d[s:e, b, 5].sum()) for s, e in laps]
        def streak(bad):
            best = cur = 0
            for x in bad:
                cur = 0 if x > 0 else cur + 1; best = max(best, cur)
            return best
        before = [i for i, (s, e) in enumerate(laps) if e <= k_rm]
        after = [i for i, (s, e) in enumerate(laps) if s >= k_rm]
        dnf = bool(rs[:, b].any()) or longest_run(stuck[:, b:b + 1])[0] * dt >= 10.0 or bw[b] >= 5.0
        S.append(dict(
            box=bool(box[b]), opp=bool(opp[b]), cluster=int(cluster[b]),
            n_laps=len(laps), best=min(lt) if lt else np.nan, median=float(np.median(lt)) if lt else np.nan,
            best_before=min([lt[i] for i in before]) if before else np.nan,
            best_after=min([lt[i] for i in after]) if after else np.nan,
            streak_hard=streak(hard_l), streak_any=streak(any_l),
            streak_hard_inc_car=streak([h + c for h, c in zip(hard_l, car_hard_l)]),
            dist=float(P[-1, b]),
            wall_any=float(d[:, b, 0].sum()), box_any=float(d[:, b, 1].sum()), car_any=float(d[:, b, 2].sum()),
            wall_hard=float(d[:, b, 3].sum()), box_hard=float(d[:, b, 4].sum()), car_hard=float(d[:, b, 5].sum()),
            box_hard_before=float(d[:k_rm, b, 4].sum()), wall_hard_before=float(d[:k_rm, b, 3].sum()),
            wall_hard_after=float(d[k_rm:, b, 3].sum()),
            stuck_s=float(stuck[:, b].sum() * dt), dnf=dnf, backwards_m=float(bw[b]), restarts=int(rs[:, b].sum()),
            laps_before=len(before), laps_after=len(after)))
    R = dict(file=fn, name=str(z["name"]), mode=str(z["mode"]), count=str(z["count"]), secs=T * dt, rm_t=rm_t, L=L, B=B,
             sessions=S)
    if opp.any():
        R["race"] = race(z, dt, T, B, L, P, d, rs, cluster)
    return R


def race(z, dt, T, B, L, P, d, rs, cluster):
    Po = progress(z["orem"], L)
    st, op = z["st"].astype(float), z["op"].astype(float)
    # initial along-track offset of ego relative to opp: from the remaining distances (wrapped to (-L/2, L/2]; a half-lap
    # start is ambiguous by sign -> taken as the ego BEHIND by L/2 (it has to catch up either way))
    r0, ro0 = z["rem"][0].astype(float), z["orem"][0].astype(float)
    off = np.mod((ro0 - r0) + 0.5 * L, L) - 0.5 * L                     # ego ahead of opp by off [m]
    off = np.where(np.abs(np.abs(off) - 0.5 * L) < 1.5, -np.abs(off), off)
    gap = (P + off[None]) - Po                                         # (T+1, B) ego ahead of opp [m]
    goal = M_LAPS * L
    has_c = "c_start" in z.files
    rows = []
    for b in range(B):
        # half a lap apart (the user's events): each car's laps from its own start. Otherwise one start line = the leading
        # car's: the car starting d m behind must cover M L + d (finishing order = physical order)
        half = abs(abs(off[b]) - 0.5 * L) < 1.5
        ge = goal + (0.0 if half else max(0.0, -off[b])); go = goal + (0.0 if half else max(0.0, off[b]))
        te = np.nonzero(P[:, b] >= ge)[0]; to = np.nonzero(Po[:, b] >= go)[0]
        te = te[0] * dt if len(te) else np.inf; to = to[0] * dt if len(to) else np.inf
        if np.isfinite(te) or np.isfinite(to):
            win = 1.0 if te < to else (0.0 if to < te else 0.5); how = "M laps"
        else:
            de, do = P[-1, b] - (ge - goal), Po[-1, b] - (go - goal)
            win = 1.0 if de > do + 0.5 else (0.0 if do > de + 0.5 else 0.5); how = "time"
        # passes with hysteresis 1 m on the gap (only while |gap| < L/2 - 2: a lap ahead is not a pass)
        # relative place wrapped to (-L/2, L/2]: lapping a slow car counts as a pass each time; far apart (|g| > L/2 - 2)
        # the order is unknown (state 0) so a wrap across +-L/2 is never a pass
        g = np.mod(gap[:, b] + 0.5 * L, L) - 0.5 * L; state = np.sign(g[0]) if 1 < abs(g[0]) < 0.5 * L - 2 else 0
        made = lost = 0
        for t in range(1, len(g)):
            if abs(g[t]) > 0.5 * L - 2:
                state = 0
                continue
            if g[t] > 1 and state < 0:
                made += 1; state = 1
            elif g[t] < -1 and state > 0:
                lost += 1; state = -1
            elif state == 0 and abs(g[t]) > 1:
                state = np.sign(g[t])
        vego, vopp = np.abs(st[:T, b, 3]), np.abs(op[:T, b, 3])
        behind = (g[:T] < -0.5) & (g[:T] > -4.0) & (vego > 0.5)
        onset = d[:, b, 2] > 0
        gt = g[:T]
        ego_fault = float((onset & (gt < -0.3)).sum()); rear_ended = float((onset & (gt > 0.3)).sum())
        side = float((onset & (np.abs(gt) <= 0.3)).sum())
        hard_c = d[:, b, 5] > 0
        r = dict(win=win, how=how, te=te, to=to, passes=made, passed_by=lost, behind_s=float(behind.sum() * dt),
                 car_onsets=float(onset.sum()), car_hard=float(hard_c.sum()), ego_fault=ego_fault, rear_ended=rear_ended,
                 side=side, ego_fault_hard=float((hard_c & (gt < -0.3)).sum()),
                 lap_lead=float(((P[-1, b] - (ge - goal)) - (Po[-1, b] - (go - goal))) / L), cluster=int(cluster[b]))
        if has_c:
            r.update(c_start=int(z["c_start"][b]), c_pace=float(z["c_pace"][b]), c_box=bool(z["c_box"][b]),
                     cfault=float(z["cfault"][:, b].sum()) if "cfault" in z.files else np.nan)
        rows.append(r)
    return rows


if __name__ == "__main__":
    out = [score(f) for f in sys.argv[2:]]
    json.dump(out, open(sys.argv[1], "w"), default=float)
    for R in out:
        S = R["sessions"]; cl = [s["cluster"] for s in S]
        f = lambda k, st=np.mean: boot([s[k] for s in S], cl, st)
        print(f"== {R['name']} mode={R['mode']} count={R['count']} sessions={len(S)} secs={R['secs']:.0f} L={R['L']:.2f}")
        for k in ("best", "best_after", "median"):
            print(f"  {k:12s} median {f(k, np.nanmedian)}")
        for k in ("streak_hard", "streak_any", "n_laps", "wall_hard", "box_hard", "wall_any", "box_any", "stuck_s", "dnf"):
            print(f"  {k:12s} mean {f(k)}")
