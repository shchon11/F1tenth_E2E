"""Competition scoring of session records (exam_record.py's format, or engine_v2's subset of it).

Per qualifier session (one car, boxes until `rm_t`, then removed):
  laps         flying laps from the first start-line crossing, each timed by interpolating the along-track progress
               (the walls-only distance-to-go field, no centerline) inside a 0.1 s decision
  best / median lap [s], best lap after the removal
  streak_hard  the longest run of consecutive complete laps with no HARD wall / box contact (the qualifier's score)
  streak_any   the same with no contact at all
  clean        no hard contact in the whole session
  dnf          restarted by the env, or < 1 m along the track over 10 s at a stretch, or >= 5 m driven backwards
  contacts     wall / box onsets, all and hard (>= the env's contact_severity_min), and by severity bucket (the
               decision's held contact severity [m/s]: < 0.5, 0.5-1.5, 1.5-3, >= 3)
Per race (1 v 1): winner = first to M laps (own laps from own start when half a lap apart, else the leading car's line
  and the car behind has to cover the gap too) within the session, else further along (ties within 0.5 m = 0.5);
  passes (the lap-wrapped gap crossing -1 -> +1 m), stuck behind (0.5-4 m behind, moving), car contacts and who was
  behind at the onset (ego behind by > 0.3 m = ego at fault), the env rule's charge (cfault) where recorded.
Intervals: cluster bootstrap over box layouts (session index mod the bank's layout count, within a map), 2000 draws,
  fixed generator seed -> the same records always give the same numbers."""
import os
import json

import numpy as np

SEV_EDGES = (0.5, 1.5, 3.0)
SEV_NAMES = ("lt0.5", "0.5-1.5", "1.5-3", "ge3")


def fill_nan(rem):
    r = np.asarray(rem, float).copy()
    for t in range(1, r.shape[0]):
        bad = ~np.isfinite(r[t])
        r[t, bad] = r[t - 1, bad]
    r[~np.isfinite(r)] = 0.0
    return r


def progress(rem, L):
    """(T+1, B) [m] along-track progress from the start; L scalar or (B,)"""
    r = fill_nan(rem)
    L = np.broadcast_to(np.asarray(L, float), (r.shape[1],))
    d = r[:-1] - r[1:]
    d = np.mod(d + 0.5 * L, L) - 0.5 * L
    return np.concatenate([np.zeros((1, r.shape[1])), np.cumsum(d, 0)], 0)


def first_crossing(rem_b, L):
    """index k0 of the first snapshot after the car first crossed the start line forwards (rem jumps up by > L/2), or None"""
    r = fill_nan(rem_b[:, None])[:, 0]
    j = np.nonzero((r[1:] - r[:-1]) > 0.5 * L)[0]
    return int(j[0]) + 1 if len(j) else None


def laps_from_progress(k0, Pb, L, dt):
    """flying laps from snapshot k0: (decision ranges [s, e) covering each lap, lap times [s])"""
    if k0 is None or k0 >= len(Pb):
        return [], []
    P0 = Pb[k0]
    kk = 1
    marks = [k0 * dt]
    for t in range(k0 + 1, len(Pb)):
        while Pb[t] >= P0 + kk * L and Pb[t - 1] < P0 + kk * L:
            f = (P0 + kk * L - Pb[t - 1]) / max(Pb[t] - Pb[t - 1], 1e-9)
            marks.append((t - 1 + f) * dt)
            kk += 1
    laps, lt = [], []
    for a, b in zip(marks[:-1], marks[1:]):
        laps.append((int(np.floor(a / dt)), int(np.ceil(b / dt))))
        lt.append(b - a)
    return laps, lt


def new_counts(cnt, rs):
    """per-decision NEW onsets from the cumulative per-episode counters (reset at a restart)"""
    prev = np.concatenate([np.zeros((1,) + cnt.shape[1:], np.float32), cnt[:-1]], 0)
    prev[1:][rs[:-1]] = 0.0
    return np.clip(cnt - prev, 0, None) * (~rs)[..., None]


def stuck_mask(P, n_win=20, m=1.0):
    T = P.shape[0] - 1
    adv = np.full((T, P.shape[1]), np.inf)
    adv[:T - n_win] = P[n_win:T] - P[:T - n_win]
    s = adv < m
    s[:10] = False
    out = s.copy()
    for k in range(1, n_win):
        out[k:] |= s[:-k]
    return out


def longest_run(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def streak(bad):
    best = cur = 0
    for x in bad:
        cur = 0 if x > 0 else cur + 1
        best = max(best, cur)
    return best


def backwards_m(vx, rem, rs, L):
    T = rs.shape[0]
    r = fill_nan(rem[:T + 1])
    L = np.broadcast_to(np.asarray(L, float), (r.shape[1],))
    dr = r[1:] - r[:-1]
    dr = np.mod(dr + 0.5 * L, L) - 0.5 * L
    dr[rs[:T].astype(bool)] = 0
    return np.where((vx[:T] > 0.2) & (dr > 0), dr, 0.0).sum(0)


def n_layouts(bank_dir):
    """the bank's layout count: session b drove layout b % n (exam_record's LINE3 order) -> the bootstrap's cluster"""
    import torch
    f = sorted(x for x in os.listdir(bank_dir) if x.endswith(".pt"))[0]
    per = torch.load(os.path.join(bank_dir, f), map_location="cpu", weights_only=False)["per_track"]
    e = next(iter(per.values()))
    if not bool((e["sid"] >= 0).any()):
        return 10 ** 9                       # a box-free bank: every session is its own cluster
    return int(e["sid"].shape[0])


def score_record(fn, M=10):
    z = np.load(fn, allow_pickle=False)
    dt = float(z["dt"])
    rs = z["rs"].astype(bool)
    T, B = rs.shape
    rm_t = float(z["rm_t"])
    k_rm = int(round(rm_t / dt))
    tid = z["tid"][0].astype(int)
    L = z["lapL"][tid].astype(float)                         # (B,) each session's lap
    st = z["st"].astype(float)
    sid = z["sid"]
    box = (sid[0] >= 0).any(-1)
    on = z["on"][:T]
    opp = on.any(0)
    d = new_counts(z["cnt"].astype(np.float32), rs)          # (T, B, 6) onsets w/p/c, hard w/p/c
    P = progress(z["rem"], L)
    stuck = stuck_mask(P)
    bw = backwards_m(st[:, :, 3], z["rem"], rs, L)
    held = z["held"] if "held" in z.files else None
    job = {}
    jf = fn[:-4] + ".job.json"
    if os.path.exists(jf):
        job = json.load(open(jf))
    bank = job.get("bank")
    if bank and not os.path.isdir(bank):                     # a record made on another machine: this machine's copy
        from . import DATA
        bank = os.path.join(DATA, "banks", os.path.basename(bank.rstrip("/")))
    n_lay = n_layouts(bank) if bank else int(os.environ.get("COMPBENCH_LAYOUTS", "64"))
    S = []
    for b in range(B):
        k0 = first_crossing(z["rem"][:, b], L[b])
        laps, lt = laps_from_progress(k0, P[:, b], L[b], dt)
        hard_l = [float(d[s:e, b, 3:5].sum()) for s, e in laps]
        any_l = [float(d[s:e, b, 0:2].sum()) for s, e in laps]
        car_l = [float(d[s:e, b, 5].sum()) for s, e in laps]
        before = [i for i, (s, e) in enumerate(laps) if e <= k_rm]
        after = [i for i, (s, e) in enumerate(laps) if s >= k_rm]
        stuck_run = longest_run(stuck[:, b]) * dt
        dnf_why = "restart" if rs[:, b].any() else ("stuck10s" if stuck_run >= 10.0 else ("backwards" if bw[b] >= 5.0 else ""))
        sev = {n: 0.0 for n in SEV_NAMES}
        if held is not None:
            ons = d[:, b, 0:2].sum(1)
            hs = held[:T, b, 3].astype(float)
            for t in np.nonzero(ons > 0)[0]:
                sev[SEV_NAMES[int(np.searchsorted(SEV_EDGES, hs[t], side="right"))]] += float(ons[t])
        row = dict(b=b, layout=int(b % n_lay), box=bool(box[b]), opp=bool(opp[b]), L=float(L[b]),
                   n_laps=len(laps), best=min(lt) if lt else np.nan, median=float(np.median(lt)) if lt else np.nan,
                   best_after=min([lt[i] for i in after]) if after else np.nan,
                   best_before=min([lt[i] for i in before]) if before else np.nan,
                   streak_hard=streak(hard_l), streak_any=streak(any_l),
                   streak_hard_car=streak([h + c for h, c in zip(hard_l, car_l)]),
                   dist=float(P[-1, b]),
                   wall_any=float(d[:, b, 0].sum()), box_any=float(d[:, b, 1].sum()), car_any=float(d[:, b, 2].sum()),
                   wall_hard=float(d[:, b, 3].sum()), box_hard=float(d[:, b, 4].sum()), car_hard=float(d[:, b, 5].sum()),
                   box_hard_before=float(d[:k_rm, b, 4].sum()), wall_hard_after=float(d[k_rm:, b, 3].sum()),
                   clean=float(d[:, b, 3:5].sum() == 0), stuck_s=float(stuck[:, b].sum() * dt),
                   dnf=float(dnf_why != ""), dnf_why=dnf_why, backwards_m=float(bw[b]), restarts=int(rs[:, b].sum()),
                   laps_before=len(before), laps_after=len(after), **{f"sev_{k}": v for k, v in sev.items()})
        S.append(row)
    R = dict(file=os.path.abspath(fn), name=str(z["name"]), mode=str(z["mode"]), secs=T * dt, rm_t=rm_t, B=B,
             map_id=job.get("map_id", str(z["map"]) if "map" in z.files else "?"), seed=int(z["seed"]) if "seed" in z.files else None,
             suite=job.get("suite"), suite_sha256=job.get("suite_sha256"), condition=job.get("condition"),
             driver=(job.get("driver") or {}).get("id"), profile=(job.get("profile") or {}).get("name"), sessions=S)
    if opp.any():
        R["race"] = race(z, dt, T, B, L, P, d, rs, M, n_lay)
        R["M"] = M
    return R


def race(z, dt, T, B, L, P, d, rs, M, n_lay):
    Po = progress(z["orem"], L)
    st, op = z["st"].astype(float), z["op"].astype(float)
    r0, ro0 = fill_nan(z["rem"])[0], fill_nan(z["orem"])[0]
    off = np.mod((ro0 - r0) + 0.5 * L, L) - 0.5 * L                    # ego ahead of opp by off [m]
    half = np.abs(np.abs(off) - 0.5 * L) < 1.5
    off = np.where(half, -np.abs(off), off)
    gap = (P + off[None]) - Po
    has_c = "c_start" in z.files
    rows = []
    for b in range(B):
        goal = M * L[b]
        ge = goal + (0.0 if half[b] else max(0.0, -off[b]))
        go = goal + (0.0 if half[b] else max(0.0, off[b]))
        te = np.nonzero(P[:, b] >= ge)[0]
        to = np.nonzero(Po[:, b] >= go)[0]
        te = te[0] * dt if len(te) else np.inf
        to = to[0] * dt if len(to) else np.inf
        if np.isfinite(te) or np.isfinite(to):
            win = 1.0 if te < to else (0.0 if to < te else 0.5)
            how = "M laps"
        else:
            de, do = P[-1, b] - (ge - goal), Po[-1, b] - (go - goal)
            win = 1.0 if de > do + 0.5 else (0.0 if do > de + 0.5 else 0.5)
            how = "time"
        g = np.mod(gap[:, b] + 0.5 * L[b], L[b]) - 0.5 * L[b]
        state = np.sign(g[0]) if 1 < abs(g[0]) < 0.5 * L[b] - 2 else 0
        made = lost = 0
        for t in range(1, len(g)):
            if abs(g[t]) > 0.5 * L[b] - 2:
                state = 0
                continue
            if g[t] > 1 and state < 0:
                made += 1
                state = 1
            elif g[t] < -1 and state > 0:
                lost += 1
                state = -1
            elif state == 0 and abs(g[t]) > 1:
                state = np.sign(g[t])
        vego = np.abs(st[:T, b, 3])
        behind = (g[:T] < -0.5) & (g[:T] > -4.0) & (vego > 0.5)
        onset = d[:, b, 2] > 0
        gt = g[:T]
        hard_c = d[:, b, 5] > 0
        # catch-up (the opposite-side start's key question): the first time the ego is within 3 m behind the other car;
        # afterwards its contacts, its passes, the time it lost against its own pace before the catch (from 5 s on)
        cu = np.nonzero((g[:T] <= -0.5) & (g[:T] >= -3.0))[0]
        k_cu = int(cu[0]) if len(cu) else None
        cu_row = dict(caught=float(k_cu is not None), t_catch=(k_cu * dt) if k_cu is not None else np.nan,
                      car_after=np.nan, car_hard_after=np.nan, fault_after=np.nan, passes_after=np.nan, lost_after_s=np.nan,
                      behind_after_s=np.nan, wallbox_hard_after=np.nan)
        if k_cu is not None:
            k5 = int(round(5.0 / dt))
            # time lost to following: while stuck 0.5-4 m behind after the catch, the share of the ego's free speed (its
            # median speed whenever it is not behind, from 5 s on) it did not drive
            free = ~behind.copy()
            free[:k5] = False
            v_f = float(np.median(vego[free])) if free.any() else np.nan
            bh = behind.copy()
            bh[:k_cu] = False
            lost_s = float(((1.0 - np.clip(vego[bh] / v_f, 0.0, 1.0)) * dt).sum()) if np.isfinite(v_f) and v_f > 0.5 else np.nan
            made_after = 0
            st_ = -1
            for t in range(k_cu + 1, len(g)):
                if abs(g[t]) > 0.5 * L[b] - 2:
                    st_ = 0
                    continue
                if g[t] > 1 and st_ < 0:
                    made_after += 1
                    st_ = 1
                elif g[t] < -1 and st_ > 0:
                    st_ = -1
                elif st_ == 0 and abs(g[t]) > 1:
                    st_ = np.sign(g[t])
            cu_row.update(car_after=float(onset[k_cu:].sum()), car_hard_after=float(hard_c[k_cu:].sum()),
                          fault_after=float((onset[k_cu:] & (gt[k_cu:] < -0.3)).sum()), passes_after=float(made_after),
                          lost_after_s=float(lost_s), behind_after_s=float(behind[k_cu:].sum() * dt),
                          wallbox_hard_after=float(d[k_cu:, b, 3:5].sum()))
        r = dict(b=b, layout=int(b % n_lay), win=win, how=how, te=te, to=to, passes=made, passed_by=lost,
                 behind_s=float(behind.sum() * dt), car_onsets=float(onset.sum()), car_hard=float(hard_c.sum()),
                 ego_fault=float((onset & (gt < -0.3)).sum()), rear_ended=float((onset & (gt > 0.3)).sum()),
                 side=float((onset & (np.abs(gt) <= 0.3)).sum()), ego_fault_hard=float((hard_c & (gt < -0.3)).sum()),
                 lap_lead=float(((P[-1, b] - (ge - goal)) - (Po[-1, b] - (go - goal))) / L[b]),
                 won_M=float(how == "M laps" and win == 1.0), half=bool(half[b]), **cu_row)
        if has_c:
            r.update(c_start=int(z["c_start"][b]), c_pace=float(z["c_pace"][b]), c_box=bool(z["c_box"][b]),
                     cfault=float(z["cfault"][:, b].sum()) if "cfault" in z.files else np.nan)
        rows.append(r)
    return rows


def score_files(files, M=10):
    out = []
    for f in files:
        R = score_record(f, M=M)
        json.dump(R, open(f[:-4] + ".score.json", "w"), default=float)
        out.append(R)
    return out


# ----------------------------------------------------------------------------------------------------- statistics
def _clusters(keys):
    keys = [tuple(k) if isinstance(k, (list, tuple)) else (k,) for k in keys]
    u = sorted(set(keys))
    idx = {k: [] for k in u}
    for i, k in enumerate(keys):
        idx[k].append(i)
    return [np.asarray(idx[k]) for k in u]


def boot(values, clusters, stat="mean", n=2000, seed=0, ci=0.95):
    """(estimate, lo, hi): cluster bootstrap of `stat` ('mean', 'median', or a callable on an index array)"""
    v = np.asarray(values, float)
    groups = _clusters(clusters)
    f = {"mean": np.nanmean, "median": np.nanmedian}.get(stat, stat)
    if not np.isfinite(v).any():
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    est = float(f(v))
    bs = np.empty(n)
    G = len(groups)
    for i in range(n):
        pick = rng.integers(0, G, G)
        s = np.concatenate([groups[j] for j in pick])
        bs[i] = f(v[s])
    a = (1.0 - ci) / 2.0
    return est, float(np.nanpercentile(bs, 100 * a)), float(np.nanpercentile(bs, 100 * (1 - a)))


def boot_ratio(num, den, clusters, n=2000, seed=0, ci=0.95):
    num, den = np.asarray(num, float), np.asarray(den, float)
    groups = _clusters(clusters)
    rng = np.random.default_rng(seed)
    est = float(num.sum() / max(den.sum(), 1e-9))
    bs = np.empty(n)
    G = len(groups)
    for i in range(n):
        s = np.concatenate([groups[j] for j in rng.integers(0, G, G)])
        bs[i] = num[s].sum() / max(den[s].sum(), 1e-9)
    a = (1.0 - ci) / 2.0
    return est, float(np.percentile(bs, 100 * a)), float(np.percentile(bs, 100 * (1 - a)))


def paired(a_rows, b_rows, key, stat="mean", n=2000, seed=0):
    """B - A on the same sessions (matched by (map, seed, b)); intervals by cluster bootstrap over (map, layout).
    Returns dict(est, lo, hi, n, same) where same = the interval contains 0."""
    ia = {(r["map_id"], r["seed"], r["b"]): r for r in a_rows}
    pairs = [(ia[(r["map_id"], r["seed"], r["b"])], r) for r in b_rows if (r["map_id"], r["seed"], r["b"]) in ia]
    va = np.array([p[0][key] for p in pairs], float)
    vb = np.array([p[1][key] for p in pairs], float)
    cl = [(p[1]["map_id"], p[1]["layout"]) for p in pairs]
    if stat == "median":
        # difference of medians, both bootstrapped on the same resampled sessions
        def f(idx):
            return np.nanmedian(vb[idx]) - np.nanmedian(va[idx])
        groups = _clusters(cl)
        rng = np.random.default_rng(seed)
        bs = [f(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])) for _ in range(n)]
        est, lo, hi = float(f(np.arange(len(va)))), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
    else:
        est, lo, hi = boot(vb - va, cl, "mean", n, seed)
    return dict(est=est, lo=lo, hi=hi, n=len(pairs), same=bool(lo <= 0.0 <= hi))


def mde(sd_diff, n, z_a=1.96, z_b=0.84):
    """the smallest true mean difference a paired comparison of n sessions detects with 80 % power at 5 %"""
    return (z_a + z_b) * sd_diff / np.sqrt(max(n, 1))


def determinism(a, b):
    za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    out = {}
    for k in ("st", "op", "cnt", "rem", "rs", "sid", "pose", "held"):
        if k in za.files and k in zb.files:
            x, y = za[k], zb[k]
            if x.shape != y.shape:
                out[k] = f"shape {x.shape} vs {y.shape}"
                continue
            if x.dtype.kind == "f":
                both = np.isfinite(x) & np.isfinite(y)
                out[k] = dict(equal=bool(np.array_equal(x[both], y[both]) and np.array_equal(np.isfinite(x), np.isfinite(y))),
                              max_abs=float(np.abs(x[both] - y[both]).max()) if both.any() else 0.0)
            else:
                out[k] = dict(equal=bool(np.array_equal(x, y)))
    out["all_equal"] = all(isinstance(v, dict) and v["equal"] for v in out.values())
    return out
