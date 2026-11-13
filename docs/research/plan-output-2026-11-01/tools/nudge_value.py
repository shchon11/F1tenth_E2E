"""What is an early deviation worth under the present reward while the same policy drives on?  (11-13, structure audit)

Question: the first 0.1 s reacts to a box only inside ~2.5 m. Is that because an earlier move does not pay (the reward's optimum is
the late move), or because it would pay and the training does not find it?
Data: box_replay.py records with the L / R arms (steering +-2 deg for the first 0.3 s after each 3 s copy, then the policy as it
is) and the S arm (plan speeds x 0.8 / 0.9 for the first 1.0 s). At a copy all cars of a group are the same car in the same world.
Unit = a copy window in which the ORIGINAL car later passes a box within 0.8 m centre to centre or touches it (the copies are
not used to pick the windows, so they can be compared with each other). Binned by how far the box is at the copy.
Per copy over the rest of the window, with the reward's own two terms:
  progress [m] along the reward's distance-to-go field (walls only, recomputed from the record's map), put in seconds with the
    pace of untouched windows;
  contact price [s] = (CONTACT_S + CONTACT_S_PER_MPS x impact speed) per wall / box contact begun (impact = the record's severity
    of the contact held in that decision; a contact that is not hard: at most half the hard threshold when the held one is harder).
"away" = the steering push to the side the original passes the box on (more room), "toward" = the other side.
A window is the 2.9 s after the copy: time lost standing after a hit beyond that is not counted (the cost of a hit is a lower bound).

    python3 nudge_value.py OUT_PREFIX record.npz [record.npz ...]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from rec_lib import Rec, plt
from f1sim.learn.plan_reward import _field_one, CONTACT_S, CONTACT_S_PER_MPS
from f1sim.contact_severity import HARD_MPS

OUT = sys.argv[1]
LOG = open(OUT + ".txt", "w")
BINS = [(0.0, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 6.0), (6.0, 10.0)]
NEAR, FAR_CTRL, TAIL = 0.8, 2.5, 5        # [m] a pass; [m] nothing nearer in a control window; decisions left after the pass
CLS = {1: "그대로", 4: None, 5: None, 6: "느리게"}
_F = {}


def say(*a):
    print(*a, flush=True); print(*a, file=LOG, flush=True)


def field(r):
    """the reward's distance-to-go field of the record's map (any finish line: only differences are used) and the lap length"""
    g, _ = r.grid(0); key = g.tobytes()[:4096] + bytes(str(g.shape), "ascii")
    if key not in _F:
        res = float(r._res[0]); org = r._org[0].astype(float)
        st = r.st.astype(float)
        for t in range(60, r.T, 7):
            p = st[t, 0]
            if r.wall_dist(0, p[None, :2])[0] > 0.7 and p[3] > 3.0:
                try:
                    D = _field_one(g, res, org, p[:2], float(p[2])); break
                except ValueError:
                    continue
        _F[key] = (D, float(np.nanmax(D)))
    return _F[key]


def remaining(r, D, xy):
    res = float(r._res[0]); org = r._org[0]
    j = np.rint((xy[..., 0] - org[0]) / res).astype(int); i = np.rint((xy[..., 1] - org[1]) / res).astype(int)
    ok = (i >= 0) & (i < D.shape[0]) & (j >= 0) & (j < D.shape[1])
    out = np.full(xy.shape[:-1], np.nan)
    out[ok] = D[i[ok], j[ok]]
    return out


wid = 0
rows_out = []      # one row a (window, class): [bin, lead s, dist0, class id, n copies, hard box, any box, hard wall, price s, progress m, end speed, same side, src hard]
ctrl = []          # control windows (no box near the original): [class id, price, progress, end speed]
kinds = {}
for fn in sys.argv[2:]:
    r = Rec(fn); z = r.z
    D, L = field(r)
    st = r.st.astype(float); T = r.T; PER = int(z["period"])
    inc = z["inc"]; grp = z["grp"].astype(int); arm = z["arm"].astype(int); rs = r.rs.astype(bool); sev = r.held[..., 3].astype(float)
    rem = remaining(r, D, st[..., :2])
    for t in range(1, rem.shape[0]):
        bad = ~np.isfinite(rem[t]); rem[t, bad] = rem[t - 1, bad]
    dp = np.mod(rem[:-1] - rem[1:] + 0.5 * L, L) - 0.5 * L          # (T, B) progress a decision [m]
    on = inc[..., 0] + inc[..., 1]
    hard_n = inc[..., 3] + inc[..., 4]
    # the held severity can be an earlier, harder contact's: a contact that is not hard is below HARD_MPS by definition
    imp = np.where(hard_n > 0, sev, np.where(sev < HARD_MPS, sev, 0.5 * HARD_MPS))
    price = (CONTACT_S * on + CONTACT_S_PER_MPS * imp * (on > 0))   # (T, B) [s]
    nudge = f"조향 {float(z['nudge_deg']):g}도 x {int(z['nudge_n']) * r.dt:.1f} s" if (arm == 4).any() else ""
    slow = f"속도 x{float(z['slow_k']):g} x {int(z['slow_n']) * r.dt:.1f} s" if (arm == 6).any() else ""
    tag = (float(z["slow_k"]) if (arm == 6).any() else 0.0)
    kinds[tag] = slow or nudge
    for g in range(int(grp.max()) + 1):
        rows = np.nonzero(grp == g)[0]; s = rows[0]
        assert arm[s] == 0
        for w in range(T // PER):
            t0, t1 = w * PER, (w + 1) * PER
            if rs[t0:t1, rows].any() or (t0 > 0 and rs[t0 - 1, s]):
                continue
            live = np.nonzero(r.sid[t0, s] >= 0)[0]
            X = st[t0:t1, rows]                                      # (PER, K, 4): st[t1] of a copy is already the next copy
            res_ = {}
            for a in np.unique(arm[rows[1:]]):
                rr = rows[arm[rows] == a]
                res_[int(a)] = (rr, price[t0:t1, rr].sum(0), dp[t0:t1 - 1, rr].sum(0), st[t1 - 1, rr, 3])
            if len(live) == 0:
                dist = np.full((PER, len(rows), 1), 99.0)
            else:
                P = r.pose[t0, s, live, :2].astype(float)
                d = P[None, None] - X[:, :, None, :2]
                dist = np.hypot(d[..., 0], d[..., 1])
            ds = dist[:, 0]
            if ds.min() > FAR_CTRL:
                for a, (rr, pr_, pg_, ve_) in res_.items():
                    ctrl.append((tag, a, pr_.mean(), pg_.mean(), ve_.mean()))
                continue
            touch = np.nonzero(inc[t0:t1, s, 1] > 0)[0]
            if len(touch):
                tp = int(touch[0]); b = int(ds[tp].argmin())
            else:
                b = int(ds.min(0).argmin()); tp = int(ds[:, b].argmin())
                if ds[tp, b] >= NEAR:
                    continue
            wid += 1
            if tp > PER - TAIL:
                continue
            c, sn = np.cos(X[..., 2]), np.sin(X[..., 2])
            lat = -sn * d[:, :, b, 0] + c * d[:, :, b, 1]            # (PER, K) the box's side seen from each car (+ left)
            side_s = np.sign(lat[tp, 0]) or 1.0
            d0 = float(ds[0, b]); bi = next((i for i, (lo, hi) in enumerate(BINS) if lo <= d0 < hi), None)
            if bi is None:
                continue
            src_hard = float(inc[t0:t1, s, 4].sum() > 0)
            for a, (rr, pr_, pg_, ve_) in res_.items():
                col = np.nonzero(np.isin(rows, rr))[0]
                tm = dist[:, col, b].argmin(0)
                same = np.mean(np.sign(lat[tm, col]) == side_s)
                cid = a
                if a in (4, 5):
                    cid = 4 if (a == 5) == (side_s > 0) else 5       # 4 = away (box on the left -> pushed right), 5 = toward
                rows_out.append((tag, bi, tp * r.dt, d0, cid, len(rr), np.mean(inc[t0:t1, rr, 4].sum(0) > 0), np.mean(inc[t0:t1, rr, 1].sum(0) > 0),
                                 np.mean(inc[t0:t1, rr, 3].sum(0) > 0), pr_.mean(), pg_.mean(), ve_.mean(), same, src_hard, wid))
    say(f"{os.path.basename(fn)}: 누적 창 {len(rows_out)}행")

A = np.array(rows_out, float); Cn = np.array(ctrl, float)
NAME = {1: "그대로", 4: "상자 반대쪽으로 틂", 5: "상자 쪽으로 틂", 6: "잠깐 느리게"}
COLR = {1: "#0b0b0b", 4: "#1c7ed6", 5: "#e8590c", 6: "#2f9e44"}
fig, axs = plt.subplots(len(kinds), 3, figsize=(16, 4.6 * len(kinds)), squeeze=False)
for ki, tag in enumerate(sorted(kinds)):
    a_ = A[A[:, 0] == tag]; c_ = Cn[Cn[:, 0] == tag]
    pace = c_[c_[:, 1] == 1][:, 3].mean() / 2.9                      # [m/s] an as-is copy in a window with no box near
    say(f"\n=== {kinds[tag]} (복사 직후에만, 그 뒤는 policy 그대로) · 상자 없는 구간의 진행 {pace:.2f} m/s ===")
    alt = c_[c_[:, 1] != 1]; asis = c_[c_[:, 1] == 1]
    say(f"  상자가 근처에 없는 창 {len(asis)}개(바꾼 것 자체의 값): 진행 {(alt[:, 3].mean() - asis[:, 3].mean()) / pace:+.3f} s, 접촉 값 {alt[:, 2].mean() - asis[:, 2].mean():+.3f} s")
    say("  원본이 상자를 0.8 m 안으로 지나가거나 닿는 창, 복사 순간 상자까지 거리별:")
    say("    거리 [m]     갈래                  창 수  세게 닿음 %   닿음 %  벽 세게 %  접촉 값 [s]   진행 차 [s]    합 (+ 이득) [s]     끝 속도 차   같은 쪽으로 지남 %")
    cls = [1, 4, 5] if tag == 0 else [1, 6]
    for bi, (lo, hi) in enumerate(BINS):
        b_ = a_[a_[:, 1] == bi]
        base = b_[b_[:, 4] == 1]
        if len(base) < 20:
            continue
        # paired by window: the rows of one window follow each other, the as-is row first
        for c in cls:
            x = b_[b_[:, 4] == c]
            n = len(x); assert n == len(base) and (x[:, 14] == base[:, 14]).all()
            dpr = x[:n, 9] - base[:n, 9]; dpg = (x[:n, 10] - base[:n, 10]) / pace; net = dpg - dpr
            se = lambda v: v.std() / np.sqrt(len(v))
            say(f"    {lo:>3.1f}~{hi:<4.1f}  {NAME[c]:<18} {n:>6d}   {x[:, 6].mean() * 100:>6.1f}     {x[:, 7].mean() * 100:>6.1f}   {x[:, 8].mean() * 100:>6.1f}    {x[:, 9].mean():>6.3f}"
                + (f"     {dpg.mean():>+6.3f}±{se(dpg):.3f}   {net.mean():>+6.3f}±{se(net):.3f}    {(x[:n, 11] - base[:n, 11]).mean():>+5.2f} m/s" if c != 1 else " " * 48)
                + f"     {x[:, 12].mean() * 100:>5.0f}")
    xs = np.arange(len(BINS))
    for c in cls:
        H_, N_, P_, G_, S_ = [], [], [], [], []
        for bi in range(len(BINS)):
            b_ = a_[a_[:, 1] == bi]; x = b_[b_[:, 4] == c]; base = b_[b_[:, 4] == 1]; n = len(x)
            H_.append(x[:, 6].mean() * 100 if len(x) else np.nan); N_.append(np.sqrt(max(x[:, 6].mean() * (1 - x[:, 6].mean()), 1e-9) / max(len(x), 1)) * 100)
            net = (x[:n, 10] - base[:n, 10]) / pace - (x[:n, 9] - base[:n, 9])
            P_.append(((x[:n, 10] - base[:n, 10]) / pace).mean() if n else np.nan)
            G_.append(net.mean() if n else np.nan); S_.append(net.std() / np.sqrt(max(n, 1)))
        o = (cls.index(c) - (len(cls) - 1) / 2) * 0.12
        axs[ki, 0].errorbar(xs + o, H_, yerr=N_, color=COLR[c], marker="o", ms=5, lw=1.8, capsize=3, label=NAME[c])
        if c != 1:
            axs[ki, 1].plot(xs + o, P_, color=COLR[c], marker="o", ms=5, lw=1.8, label=NAME[c])
            axs[ki, 2].errorbar(xs + o, G_, yerr=S_, color=COLR[c], marker="o", ms=5, lw=1.8, capsize=3, label=NAME[c])
    axs[ki, 0].set_ylabel("그 창에서 상자에 세게 닿은 복사본 [%]"); axs[ki, 0].set_title(f"{kinds[tag]}: 세게 닿는 비율", fontsize=11)
    axs[ki, 1].set_ylabel("그대로 둔 복사본보다 앞선 시간 [s] (- = 늦음)"); axs[ki, 1].set_title("진행에서 잃거나 번 시간", fontsize=11)
    axs[ki, 2].set_ylabel("진행 + 접촉 값, 그대로 둔 것 대비 [s] (+ = 이득)"); axs[ki, 2].set_title("지금 보상으로 친 값 (창 3 s 안)", fontsize=11)
    for a in axs[ki]:
        a.set_xticks(xs); a.set_xticklabels([f"{lo:g}~{hi:g}" for lo, hi in BINS]); a.set_xlabel("복사한 순간 상자까지 거리 [m]")
        a.grid(alpha=0.25); a.legend(fontsize=9)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    axs[ki, 1].axhline(0, color="#868e96", lw=0.8); axs[ki, 2].axhline(0, color="#868e96", lw=0.8)
fig.suptitle("plan25y u150: 상자 앞에서 일찍 잠깐 바꿔 준 행동의 값 (그 뒤는 policy 그대로)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT + ".png", dpi=105)
say("그림", OUT + ".png")
