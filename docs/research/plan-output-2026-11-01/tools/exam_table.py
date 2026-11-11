"""시험 점수표 (exam1): exam_record.py 기록(npz)을 세션 단위로 채점. 세션 = 차 한 대의 90 s(상자는 50 s에 치움).
경우마다(혼자·빈 트랙 / 혼자·상자 / 상대 차 / 상대 차·상자):
  - 깨끗한 세션(세게 닿음 0번, 2 s 넘게 선 적 없음) 비율, 한 번도 안 닿은 세션 비율, 세션당 세게 닿음(벽·상자·차)
  - 한 바퀴 시간(세션 중앙값들의 중앙), 세션 안 흔들림, 세게 닿지 않고 이어 달린 바퀴 수
  - 출발(처음 5 s)에 세게 닿은 세션, 세게 닿은 뒤 5 s 안에 서 버린 비율과 선 뒤 다시 2 m/s로 달리기까지 걸린 시간(스스로 빠져나오나)
  - 상자 세션: 상자 있는 동안 / 치운 뒤로 나눔, 치운 뒤 한 바퀴가 빈 트랙 세션과 얼마나 다른가, 같은 배치에서 바퀴 번호별 상자에 세게 닿음
  - 상대 차 세션: 90 s 동안 상대보다 더 멀리 간 세션 비율(= 같은 거리를 더 빨리), 거리 차, 앞지름·앞지름 당함, 차끼리 세게 닿은 세션,
    상대 속도 배율 구간·종류별, 상대 차 자체가 제대로 달렸나(속도·선 시간·다시 출발)
모든 비율·중앙값에 세션 다시 뽑기(bootstrap 2000번) 95 % 구간. 거리는 보상의 벽 기준 거리장(중심선 없음)으로 잼.
인자: 기록npz [기록npz ...]   (EXAM_ROW=1: 파일마다 한 줄 요약만)"""
import os, sys
import numpy as np

RNG = np.random.default_rng(0)
HYST = 1.0          # [m] a pass counts once the gap has changed sign by this much
STALL_S = 2.0


def ci(v, stat=np.mean, n=2000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(v), (n, len(v)))
    s = np.array([stat(v[i]) for i in idx]) if stat is not np.mean else v[idx].mean(1)
    return tuple(np.percentile(s, [2.5, 97.5]))


def pct(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "  -"
    lo, hi = ci(v)
    return f"{100 * v.mean():3.0f} % [{100 * lo:.0f}-{100 * hi:.0f}]"


def med(v, unit="", f="{:.2f}"):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return "-"
    lo, hi = ci(v, np.median)
    return (f + unit + " [" + f + "-" + f + "]").format(np.median(v), lo, hi)


def mean(v, f="{:.2f}"):
    v = np.asarray(v, float)
    if len(v) == 0:
        return "-"
    lo, hi = ci(v)
    return (f + " [" + f + "-" + f + "]").format(v.mean(), lo, hi)


def progress(rem, L, skip=None):
    """(T+1, B) distance to go -> (T+1, B) cumulative distance driven along the track (a wall / off-grid sample holds the last value;
    `skip` (T, B): decisions whose step is not driving, e.g. a re-placed car)"""
    r = rem.astype(float).copy()
    for t in range(1, r.shape[0]):                       # carry the last valid value forward
        bad = ~np.isfinite(r[t]); r[t, bad] = r[t - 1, bad]
    r[~np.isfinite(r)] = 0.0
    d = r[:-1] - r[1:]
    d = np.mod(d + 0.5 * L, L) - 0.5 * L
    if skip is not None:
        d[skip] = 0.0
    return np.concatenate([np.zeros((1, r.shape[1])), np.cumsum(d, 0)], 0), r


def runs(mask, n_min):
    """per column: the number of runs of True at least n_min long, and the longest run"""
    T, B = mask.shape
    cnt = np.zeros(B, int); longest = np.zeros(B, int); cur = np.zeros(B, int)
    for t in range(T):
        cur = np.where(mask[t], cur + 1, 0)
        cnt += (cur == n_min)
        longest = np.maximum(longest, cur)
    return cnt, longest


def score(fn):
    z = np.load(fn, allow_pickle=False)
    dt = float(z["dt"]); cnt = z["cnt"].astype(float); T, B = cnt.shape[:2]
    secs = T * dt; rm_t = float(z["rm_t"]); k_rm = int(round(rm_t / dt))
    st = z["st"].astype(float); on = z["on"][:T]; sid = z["sid"]; lapsig = z["lap"]
    L = float(z["lapL"][int(z["tid"][0, 0])])
    kinds = [str(k) for k in z["kinds"]]
    box = (sid[0] >= 0).any(-1); opp = on.any(0)
    v = np.abs(st[:T, :, 3])
    d = np.clip(np.diff(np.concatenate([np.zeros((1, B, 6)), cnt], 0), axis=0), 0, None)      # new contacts per decision
    hard = d[:, :, 3:6]; anyc = d[:, :, 0:3]
    n1 = int(round(1.0 / dt)); n_st = int(round(STALL_S / dt))
    slow = v < 0.3; slow[:n1] = False
    stalls, longest = runs(slow, n_st)
    P, _ = progress(z["rem"], L)
    # laps: a lap ends when the net count of line crossings reaches a new high
    net = np.cumsum(lapsig, 0); S = {}
    laps = [[] for _ in range(B)]                        # (start s, end s, hard contacts inside)
    for b in range(B):
        hi, t_prev = 0, None
        for t in np.nonzero(lapsig[:, b] > 0)[0]:
            if net[t, b] > hi:
                hi = net[t, b]
                if t_prev is not None:
                    laps[b].append((t_prev * dt, (t + 1) * dt, float(hard[t_prev:t + 1, b].sum())))
                t_prev = t + 1
    def lap_stats(b, lo=0.0, hi_=1e9):
        x = [e - s for s, e, _ in laps[b] if s >= lo and e <= hi_]
        return (np.median(x), np.min(x), np.std(x) if len(x) > 1 else np.nan, len(x)) if x else (np.nan, np.nan, np.nan, 0)
    def streak(b):
        best = cur = 0
        for _, _, h in laps[b]:
            cur = cur + 1 if h == 0 else 0
            best = max(best, cur)
        return best
    name = f"{z['exam']} · {dict(map='MAP', track='시간 추종기', ideal='plan대로 정확히')[str(z['mode'])]} · 상자 " \
           f"{dict(max2='2개', uniform='1개~전부').get(str(z['count']), str(z['count']))} · {z['name']}"
    out = [f"== {name}  (세션 {B}개 x {secs:.0f} s, 상자는 {rm_t:g} s에 치움, 시드 {int(z['seed'])}, 도중 다시 출발 {int(z['rs'].sum())}번) =="]
    row = {"name": str(z["name"]), "cond": f"{z['mode']}/{z['count']}"}
    h_all = hard.sum(0)                                   # (B, 3) per session
    clean = (h_all.sum(1) == 0) & (stalls == 0)
    lap_med = np.array([lap_stats(b)[0] for b in range(B)]); lap_min = np.array([lap_stats(b)[1] for b in range(B)])
    lap_sd = np.array([lap_stats(b)[2] for b in range(B)]); lap_n = np.array([lap_stats(b)[3] for b in range(B)])
    stk = np.array([streak(b) for b in range(B)])
    empty_solo = ~box & ~opp
    # the start phase and what follows a hard contact (events: decisions with a new hard contact, 1 s apart at least, 10 s of
    # session left): did the car come to a stand (< 0.3 m/s) within 5 s, and if so how long until it drives (>= 2 m/s) again
    n5 = int(round(5.0 / dt)); hsum = hard.sum(2) > 0
    ev_b, ev_stop, ev_back = [], [], []
    for b_ in range(B):
        last = -10 ** 9
        for t in np.nonzero(hsum[:, b_])[0]:
            fresh = t - last >= n1; last = t
            if not fresh or T - t < 2 * n5:
                continue
            k = np.nonzero(v[t + 1:t + 1 + n5, b_] < 0.3)[0]
            ev_b.append(b_); ev_stop.append(len(k) > 0)
            if len(k):
                t_s = t + 1 + k[0]
                g = np.nonzero(v[t_s:, b_] >= 2.0)[0]
                ev_back.append(g[0] * dt if len(g) else np.inf)
            else:
                ev_back.append(np.nan)
    ev_b = np.array(ev_b, int); ev_stop = np.array(ev_stop, bool); ev_back = np.array(ev_back, float)
    for label, m in (("혼자 · 빈 트랙", ~box & ~opp), ("혼자 · 상자", box & ~opp), ("상대 차 · 빈 트랙", ~box & opp), ("상대 차 · 상자", box & opp)):
        n = int(m.sum())
        if n == 0:
            continue
        key = label.replace(" ", "")
        out.append(f"[{label}] 세션 {n}개")
        out.append(f"  깨끗한 세션(세게 닿음 0번·2 s 넘게 선 적 없음) {pct(clean[m])} | 한 번도 안 닿은 세션 {pct(anyc.sum((0, 2))[m] == 0)} | "
                   f"2 s 넘게 선 세션 {pct(stalls[m] > 0)} (5 s 넘게 {pct(longest[m] * dt > 5.0)})")
        out.append(f"  세션당 세게 닿음: 벽 {mean(h_all[m, 0])} · 상자 {mean(h_all[m, 1])} · 차 {mean(h_all[m, 2])} | 달린 거리 {med(P[-1, m], ' m', '{:.0f}')}")
        out.append(f"  한 바퀴(세션 중앙값) {med(lap_med[m], ' s')} · 세션에서 가장 빠른 바퀴 {med(lap_min[m], ' s')} · 세션 안 바퀴 흔들림 "
                   f"{med(lap_sd[m], ' s')} · 온전한 바퀴 수 {med(lap_n[m], '', '{:.0f}')} · 세게 닿지 않고 이어 달린 바퀴 {med(stk[m], '', '{:.0f}')}")
        em = m[ev_b] if len(ev_b) else np.zeros(0, bool)
        es, eb = ev_stop[em], ev_back[em]; eb = eb[es]
        out.append(f"  출발(처음 5 s)에 세게 닿은 세션 {pct(hsum[:n5].any(0)[m])} | 세게 닿은 뒤({len(es)}건): 5 s 안에 선 것 {pct(es)}"
                   + (f" · 선 뒤 다시 달리기(2 m/s)까지 {np.median(eb):.1f} s(중앙) · 5 s 안에 다시 못 달린 것 {pct(eb > 5.0)}" if len(eb) else ""))
        row[key + "_clean"] = clean[m]; row[key + "_lap"] = lap_med[m]
        if label.endswith("상자"):
            hb, ha = hard[:k_rm].sum(0), hard[k_rm:].sum(0)
            sb = runs(slow[:k_rm], n_st)[0]; sa = runs(slow[k_rm:], n_st)[0]
            lb = np.array([lap_stats(b, 0.0, rm_t)[0] for b in range(B)]); la = np.array([lap_stats(b, rm_t)[0] for b in range(B)])
            out.append(f"  상자 있는 동안(0-{rm_t:g} s): 깨끗한 세션 {pct(((hb.sum(1) == 0) & (sb == 0))[m])} | 세게 닿음 벽 {mean(hb[m, 0])} · 상자 "
                       f"{mean(hb[m, 1])} · 차 {mean(hb[m, 2])} | 한 바퀴 {med(lb[m], ' s')}")
            ref = lap_med[empty_solo] if not opp[m].any() else lap_med[~box & opp]
            diff = np.nanmedian(la[m]) - np.nanmedian(ref) if np.isfinite(la[m]).any() and np.isfinite(ref).any() else float("nan")
            out.append(f"  치운 뒤({rm_t:g}-{secs:g} s): 깨끗한 세션 {pct(((ha.sum(1) == 0) & (sa == 0))[m])} | 세게 닿음 벽 {mean(ha[m, 0])} | 한 바퀴 "
                       f"{med(la[m], ' s')} (같은 경우의 상자 없는 세션보다 {diff:+.2f} s)")
            row[key + "_clean_box"] = ((hb.sum(1) == 0) & (sb == 0))[m]; row[key + "_lap_after"] = la[m]
            # the same layout lap after lap: hard box contacts in the k-th time round (by distance driven, the boxes still there)
            lapno = np.floor(P[:T] / L).astype(int)
            txt = []
            for k, lab in ((0, "첫 바퀴"), (1, "둘째"), (2, "셋째"), (3, "넷째 이후")):
                sel = (lapno == k) if k < 3 else (lapno >= 3)
                sel = sel & (np.arange(T)[:, None] < k_rm) & m[None, :]
                mins = sel.sum() * dt / 60.0
                if mins > 1.0:
                    txt.append(f"{lab} {(hard[:, :, 1] * sel).sum() / mins:.2f}")
            out.append("  같은 배치를 도는 바퀴 번호별 상자에 세게 닿음(/분): " + " · ".join(txt))
        if label.startswith("상대"):
            ors = z["ors"][:T]
            Po, _ = progress(z["orem"], L, skip=ors)
            g0 = np.mod(z["orem"][0].astype(float) - z["rem"][0].astype(float) + 0.5 * L, L) - 0.5 * L     # + = we start ahead
            g0 = np.nan_to_num(g0)
            G = g0[None, :] + P - Po
            made = np.zeros(B, int); lost = np.zeros(B, int); cell = np.floor(G[0] / L)
            for t in range(1, T + 1):
                re_ = ors[t - 1]                                # a re-placed opponent: start counting from where it now is
                cell = np.where(re_, np.floor(G[t] / L), cell)
                up = G[t] >= (cell + 1) * L + HYST; dn = G[t] <= cell * L - HYST
                made += up & ~re_; lost += dn & ~re_
                cell = cell + up.astype(int) - dn.astype(int)
            gain = P[-1] - Po[-1]
            hc = h_all[:, 2]
            out.append(f"  상대보다 더 멀리 간 세션 {pct(gain[m] > 0)} · 거리 차 {med(gain[m], ' m', '{:+.0f}')} | 앞지름 {mean(made[m])}번 · 앞지름 당함 "
                       f"{mean(lost[m])}번 /세션 | 차끼리 세게 닿은 세션 {pct(hc[m] > 0)}")
            row[key + "_win"] = (gain[m] > 0); row[key + "_carhit"] = (hc[m] > 0)
            sc = z["scale"][0]; kd = z["kind"][0]
            for lo, hi_ in ((0.0, 0.7), (0.7, 1.0), (1.0, 9.9)):
                mm = m & (sc >= lo) & (sc < hi_)
                if mm.sum() >= 5:
                    out.append(f"    상대 속도 배율 {max(lo, 0.4):.1f}-{min(hi_, 1.2):.1f} ({int(mm.sum())}개): 더 멀리 {pct(gain[mm] > 0)} · 앞지름 "
                               f"{made[mm].mean():.2f} · 당함 {lost[mm].mean():.2f} · 차끼리 세게 닿은 세션 {pct(hc[mm] > 0)} · 깨끗한 세션 {pct(clean[mm])}")
            for k, kn in enumerate(kinds):
                mm = m & (kd == k)
                if mm.sum() >= 5:
                    ov = np.abs(z["op"][:T, mm, 3].astype(float))
                    oc = z["ocnt"][-1][mm].astype(float)
                    out.append(f"    상대 {kn} ({int(mm.sum())}개): 더 멀리 {pct(gain[mm] > 0)} · 차끼리 세게 닿은 세션 {pct(hc[mm] > 0)} || 상대 차 자체: 속도 "
                               f"{ov.mean():.2f} m/s(배율 평균 {sc[mm].mean():.2f}) · 선 시간 {100 * (ov < 0.3).mean():.0f} % · 다시 출발 "
                               f"{ors[:, mm].sum() / mm.sum():.2f}번/세션 · 세게 닿음/세션 벽 {oc[:, 3].mean():.2f} 상자 {oc[:, 4].mean():.2f}")
    return out, row


rows = []
for fn in sys.argv[1:]:
    o, r = score(fn)
    rows.append(r)
    if os.environ.get("EXAM_ROW") != "1":
        print("\n".join(o), flush=True)
if len(rows) > 1 or os.environ.get("EXAM_ROW") == "1":
    print("\n== 한 줄 요약 (깨끗한 세션 비율 / 한 바퀴 s / 더 멀리 간 세션) ==")
    for r in rows:
        g = lambda k: r.get(k, np.array([]))
        print(f"  {r['name']:<28} {r['cond']:<13} | 깨끗: 혼자 빈 {pct(g('혼자·빈트랙_clean'))} · 혼자 상자(있는 동안) {pct(g('혼자·상자_clean_box'))} · 상대 "
              f"{pct(g('상대차·빈트랙_clean'))} · 상대+상자(있는 동안) {pct(g('상대차·상자_clean_box'))} | 빈 트랙 한 바퀴 {med(g('혼자·빈트랙_lap'), ' s')} · 치운 뒤 "
              f"{med(g('혼자·상자_lap_after'), ' s')} | 더 멀리: 상대 {pct(g('상대차·빈트랙_win'))} · 상대+상자 {pct(g('상대차·상자_win'))}")
