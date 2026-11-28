"""Direct measurements a single bag supports, each with its error and the evidence it rests on.

Every function returns plain dicts (value, err, n, how) so `report.py` can write them as profile entries.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from . import thresholds as TH
from .signals import HZ, G, L_WHEELBASE, smooth, deriv


def _runs(mask: np.ndarray, min_len: int):
    """(start, stop) of runs of True at least `min_len` long."""
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(int))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a, b) for a, b in zip(st, en) if b - a >= min_len]


def _block_ci(values: np.ndarray, blocks: np.ndarray, stat=np.median, n_rep: int = 200, seed: int = 0):
    """bootstrap over blocks (e.g. 10 s chunks): 5-95 % interval of `stat`."""
    ub = np.unique(blocks)
    if len(ub) < 3:
        return (float("nan"), float("nan"))
    by = [values[blocks == b] for b in ub]
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_rep):
        pick = rng.integers(0, len(ub), len(ub))
        v = np.concatenate([by[i] for i in pick])
        if len(v):
            reps.append(stat(v))
    return (float(np.percentile(reps, 5)), float(np.percentile(reps, 95)))


def moving_mask(S, v_min=1.0, sel=None):
    """moving, with a command on the bus (`cmd_valid`), and inside `sel` (the marked test segments) when given."""
    m = (S["v_wheel"] > v_min) & ~S["still"]
    if "cmd_valid" in S:
        m &= S["cmd_valid"]
    if sel is not None:
        m &= sel
    return m


def current_plateau(i: np.ndarray, t: np.ndarray, side: str) -> Dict:
    """Is the current limit visible in this bag? A limit shows as the current sitting FLAT at one value: the samples
    within CUR_PLATEAU_BAND (2 %) of the extreme (p99.9 for drive, p0.1 for regen) form runs of at least
    CUR_PLATEAU_MIN_S (0.2 s), and at least CUR_PLATEAU_MIN_RUNS (2) such runs lie CUR_PLATEAU_SEP_S (0.5 s) apart.
    Then the limit is 'measured' (value = median of the plateau samples, err = their std); otherwise the extreme is
    only a lower bound on the limit's magnitude (the car may never have reached it)."""
    ok = np.isfinite(i)
    if not ok.any():
        return dict(status="not_identified", value=float("nan"), how="no motor current")
    ext = float(np.nanpercentile(i, 99.9 if side == "max" else 0.1))
    out = dict(extreme=ext, runs=0, seconds=0.0)
    if abs(ext) < 1.0:
        return dict(out, status="lower_bound", value=ext, how=f"|extreme| {abs(ext):.2f} A < 1 A: no limit visible")
    band = TH.CUR_PLATEAU_BAND * abs(ext)
    m = ok & (np.abs(np.where(ok, i, 0.0) - ext) <= band)
    runs = _runs(m, int(TH.CUR_PLATEAU_MIN_S * HZ))
    sep = []
    for a, b in runs:
        if not sep or (a - sep[-1][1]) >= TH.CUR_PLATEAU_SEP_S * HZ:
            sep.append((a, b))
        else:
            sep[-1] = (sep[-1][0], b)
    out["runs"] = len(sep)
    out["seconds"] = float(sum(b - a for a, b in runs) / HZ)
    if len(sep) >= TH.CUR_PLATEAU_MIN_RUNS:
        idx = np.concatenate([np.arange(a, b) for a, b in runs])
        v = float(np.median(i[idx]))
        return dict(out, status="measured", value=v, err=float(np.std(i[idx])),
                    how=(f"{len(sep)} flat stretches (within {100 * TH.CUR_PLATEAU_BAND:.0f} % of {ext:.1f} A, "
                         f">= {TH.CUR_PLATEAU_MIN_S} s each, >= {TH.CUR_PLATEAU_SEP_S} s apart), {out['seconds']:.1f} s in all"))
    return dict(out, status="lower_bound", value=ext,
                how=(f"extreme {ext:.1f} A, but only {len(sep)} flat stretch(es) (>= {TH.CUR_PLATEAU_MIN_RUNS} needed): "
                     "the limit may be higher"))


def low_speed_turn(S, sel=None) -> Dict:
    """Steady low-speed turning (T2): samples with v in [T2_V_MIN, T2_V_MAX], |steer| > T2_MIN_STEER and |dr/dt| <
    T2_MAX_DR, summed (they need not be contiguous)."""
    v = S["v_wheel"]
    m = (v > TH.T2_V_MIN) & (v < TH.T2_V_MAX) & (np.abs(S["steer"]) > TH.T2_MIN_STEER) \
        & (np.abs(deriv(S["r"], 5)) < TH.T2_MAX_DR)
    if "cmd_valid" in S:
        m &= S["cmd_valid"]
    if sel is not None:
        m &= sel
    return dict(seconds=float(m.sum() / HZ), need_s=TH.T2_MIN_SECONDS)


# ---- steering: command -> yaw-rate lag, trim ------------------------------------------------------------------------

def delay_xcorr(S, max_lag_s: float = TH.DELAY_MAX_LAG_S, sel=None) -> Dict:
    """Total command -> yaw-rate lag by cross-correlation of the kinematic yaw-rate demand v tan(d_applied) / L with the
    gyro, on moving stretches, both high-passed (1 s mean removed) so the slow trend does not dominate. Sub-sample
    peak by a parabola. This lumps the pure delay, the servo lag, tyre build-up and yaw inertia: it is what the
    replay models must reproduce as a sum, not any one of their terms."""
    mv = moving_mask(S, TH.DELAY_MIN_V, sel)
    rk = S["v_wheel"] * np.tan(S["steer"]) / L_WHEELBASE
    r = S["r"]
    hp = lambda x: x - smooth(x, int(HZ))
    a, b = hp(rk), hp(r)
    lags = np.arange(0, int(max_lag_s * HZ) + 1)
    segs = _runs(mv, int(TH.DELAY_MIN_SEG_S * HZ))
    if not segs:
        return dict(value=float("nan"), n=0, seconds=float(mv.sum() / HZ),
                    how=f"no moving stretch of >= {TH.DELAY_MIN_SEG_S} s (moving {mv.sum() / HZ:.1f} s in all)")
    cc = np.zeros(len(lags))
    per_seg = []
    for s0, s1 in segs:
        c_seg = np.array([np.sum(a[s0:s1 - L_] * b[s0 + L_:s1]) for L_ in lags])
        cc += c_seg
        per_seg.append(c_seg)
    i = int(np.argmax(cc))

    def peak(c):
        j = int(np.argmax(c))
        if 0 < j < len(c) - 1:
            d = c[j - 1] - 2 * c[j] + c[j + 1]
            return (j + (0.5 * (c[j - 1] - c[j + 1]) / d if d != 0 else 0.0)) / HZ
        return j / HZ
    val = peak(cc)
    # bootstrap over stretches
    rng = np.random.default_rng(0)
    reps = []
    P = np.array(per_seg)
    for _ in range(200):
        pick = rng.integers(0, len(P), len(P))
        reps.append(peak(P[pick].sum(0)))
    return dict(value=float(val), ci=(float(np.percentile(reps, 5)), float(np.percentile(reps, 95))),
                n=len(segs), seconds=float(sum(b - a for a, b in segs) / HZ),
                how="xcorr of v*tan(steer_applied)/L with gyro z (1 s high-pass), moving stretches >= 3 s")


def trim(S, k_gain: float = 1.0, sel=None) -> Dict:
    """The applied steering command that drives straight: zero crossing of yaw rate per speed against the command
    near straight (|d| < 0.08 rad, v > 1.5, steady: |dr/dt| small). trim b = -k * d0 in the replay models' terms."""
    mv = moving_mask(S, TH.TRIM_MIN_V, sel)
    d = S["steer"]
    rv = S["r"] / np.maximum(S["v_wheel"], 0.5)
    rd = np.abs(deriv(S["r"], 5))
    m = mv & (np.abs(d) < TH.TRIM_MAX_STEER) & (rd < TH.TRIM_MAX_DR) & np.isfinite(rd)
    if m.sum() < TH.TRIM_MIN_SAMPLES:
        return dict(value=float("nan"), n=int(m.sum()), how="too little near-straight driving")
    x, y = d[m], rv[m]
    A = np.stack([x, np.ones_like(x)], 1)
    g, o = np.linalg.lstsq(A, y, rcond=None)[0]
    d0 = -o / g
    blocks = (S["t"][m] // 10).astype(int)
    reps = []
    rng = np.random.default_rng(0)
    ub = np.unique(blocks)
    for _ in range(200):
        pick = rng.choice(ub, len(ub))
        ii = np.concatenate([np.flatnonzero(blocks == u) for u in pick])
        gg, oo = np.linalg.lstsq(A[ii], y[ii], rcond=None)[0]
        reps.append(-oo / gg)
    return dict(value=float(d0), ci=(float(np.percentile(reps, 5)), float(np.percentile(reps, 95))),
                trim_b=float(-k_gain * d0), n=int(m.sum()), slope_per_rad=float(g),
                how="zero crossing of r/v vs applied steering command, |cmd| < 0.08 rad, v > 1.5 m/s")


# ---- longitudinal: current-limited acceleration, regen-limited braking ---------------------------------------------

def accel_brake(S, sel=None) -> Dict:
    """Acceleration while the motor current sits at its limit, braking while it sits at the regen limit.
    Body acceleration = median of the IMU x (bias removed) and the wheel-speed derivative (the wheel alone lies
    under spin / lock). Binned by speed. Without /sensors/core there is no current and nothing to report."""
    out: Dict = {}
    if "i_motor" not in S or not np.isfinite(S["i_motor"]).any():
        return dict(note="no motor current (/sensors/core missing)")
    i = S["i_motor"]
    v = smooth(S["v_wheel"], 5)
    # light smoothing (50 ms) and only the INTERIOR of each current-limited run (eroded by the filter half-widths):
    # the limited stretches last 0.1-0.5 s, so a 0.1 s filter reaching across their edges halves what they show
    a_w = deriv(smooth(S["v_wheel"], 5), 3)
    a_i = smooth(S["ax"], 5)
    ok = np.isfinite(a_w) & (v > 0.3)
    if "cmd_valid" in S:
        ok &= S["cmd_valid"]
    if sel is not None:
        ok &= sel
    i_hi, i_lo = float(np.nanpercentile(i, 99.9)), float(np.nanpercentile(i, 0.1))
    out["current_max_A"] = i_hi
    out["current_min_A"] = i_lo
    out["current_max_fit"] = current_plateau(i, S["t"], "max")
    out["current_min_fit"] = current_plateau(i, S["t"], "min")
    for side, sel_fn, key in (("drive", lambda: i >= TH.CUR_LIMIT_FRAC * i_hi, "a_max"), ("brake", lambda: i <= TH.CUR_LIMIT_FRAC * i_lo, "a_brake")):
        m = ok & sel_fn()
        trim_n = int(TH.ACCEL_TRIM_S * HZ)
        runs = [(a + trim_n, b - trim_n) for a, b in _runs(m, int(TH.ACCEL_MIN_RUN_S * HZ))]   # >= 0.2 s at the limit, 50 ms trimmed off each end
        if not runs:
            out[key] = dict(value=float("nan"), n=0, seconds=0.0, by_speed=[],
                            how=f"no {side} stretch of >= {TH.ACCEL_MIN_RUN_S} s at the current limit")
            continue
        idx = np.concatenate([np.arange(a, b) for a, b in runs])
        a_body = np.median(np.stack([a_w[idx], a_i[idx]]), 0)
        a_body = a_body if side == "drive" else -a_body
        bins = [(0.3, 3), (3, 5), (5, 7), (7, 9), (9, 20)]
        tab = []
        for lo, hi in bins:
            mm = (v[idx] >= lo) & (v[idx] < hi)
            if mm.sum() >= TH.ACCEL_BIN_MIN_SAMPLES:
                tab.append(dict(v=(lo, hi), median=float(np.median(a_body[mm])), n=int(mm.sum()),
                                wheel=float(np.median(a_w[idx][mm]) * (1 if side == "drive" else -1)),
                                imu=float(np.median(a_i[idx][mm]) * (1 if side == "drive" else -1))))
        blocks = (S["t"][idx] // 5).astype(int)
        ci = _block_ci(a_body, blocks)
        out[key] = dict(value=float(np.median(a_body)), ci=ci, n=int(len(idx)), stretches=len(runs), by_speed=tab,
                        seconds=float(len(idx) / HZ),
                        how=f"median of IMU a_x and wheel dv/dt while current {'>=' if side == 'drive' else '<='} "
                            f"90 % of its p{'99.9' if side == 'drive' else '0.1'} ({i_hi if side == 'drive' else i_lo:.0f} A)")
    # fit a_max * min(1, v_switch / v) to the drive table (the simulator's form)
    tab = out.get("a_max", {}).get("by_speed", [])
    if len(tab) >= 2:
        vm = np.array([0.5 * (t["v"][0] + min(t["v"][1], 10.0)) for t in tab])
        am = np.array([t["median"] for t in tab])
        best = None
        for vs in np.arange(3.0, 12.01, 0.25):
            f = np.minimum(1.0, vs / vm)
            a0 = float(np.sum(am * f) / np.sum(f * f))
            e = float(np.sum((am - a0 * f) ** 2))
            if best is None or e < best[0]:
                best = (e, a0, vs)
        out["a_max_vswitch_fit"] = dict(a_max=best[1], v_switch=best[2], rss=best[0])
    return out


# ---- lateral grip: the largest lateral acceleration the car held ---------------------------------------------------

def lateral_limit(S, hold_s: float = TH.LAT_HOLD_S, slide_deg: float = TH.SLIDE_DEG, sel=None) -> Dict:
    """Largest |a_y| held for `hold_s` (rolling minimum of |a_y| over the hold, IMU, 0.1 s mean) while the car did not
    slide (real body-slip change over the hold < 5 deg from beta_dot = a_y / v - r). A LOWER BOUND on the grip:
    the car was at this acceleration and did not lose it; whether it had more is not in the data unless it slid."""
    v = np.maximum(S["v_wheel"], 0.3)
    ay = S["ay_s"]
    n = int(hold_s * HZ)
    mv = moving_mask(S, 1.0, sel)
    absay = np.where(mv, np.abs(ay), 0.0)
    # rolling min over the hold window
    from numpy.lib.stride_tricks import sliding_window_view as swv
    if len(absay) <= n:
        return dict(value=float("nan"), how="bag too short")
    rm = swv(absay, n).min(1)
    bd = ay / v - S["r"]
    slip = np.abs(np.cumsum(bd) / HZ)
    dslip = swv(slip, n)
    slip_change = dslip.max(1) - dslip.min(1)
    ok = slip_change < np.deg2rad(TH.LAT_NOSLIP_DEG)
    vals = np.where(ok, rm, 0.0)
    j = int(np.argmax(vals))
    vr = smooth(S["v_wheel"] * S["r"], 10)
    # the next-highest independent holds (>= 2 s apart): how often the car was there
    order = np.argsort(vals)[::-1]
    picks = []
    for k in order:
        if vals[k] <= 0:
            break
        if all(abs(k - p) > 2 * HZ for p in picks):
            picks.append(k)
        if len(picks) >= 10:
            break
    tops = [float(vals[k]) for k in picks]
    # slides: |slip change| > slide_deg within 1 s while moving -- the evidence for an UPPER bound. 30 deg: race
    # driving that held its corners reaches 15-25 deg of this integral (IMU scale and bias drift, legitimate turn-in
    # slip); the one real spin (map16x07) passes 40
    n1 = int(TH.SLIDE_WINDOW_S * HZ)
    dsl = swv(slip, n1)
    big = (dsl.max(1) - dsl.min(1)) > np.deg2rad(slide_deg)
    big &= swv(np.where(mv, S["v_wheel"], 0.0), n1).min(1) > TH.SLIDE_MIN_V
    # an impact (0.05 s mean |a| > 15 m/s^2) puts a spike on a_y that the slip integral reads as a slide
    imp = smooth(np.hypot(S["ax"], S["ay"]), 5) > TH.IMPACT_ACC
    big &= ~(swv(imp, n1).any(1))
    slide_t = [float(S["t"][k]) for k in np.flatnonzero(big)[::int(HZ)]][:20]
    return dict(value=float(vals[j]), t=float(S["t"][j]), v_at=float(S["v_wheel"][j]), vr_at=float(vr[j]),
                top10=tops, mu_lower=float(vals[j] / G), slides_t=slide_t, n_slide_s=float(big.sum() / HZ),
                how=f"max over the marked/auto stretches of |a_y| (IMU, 0.1 s mean) held >= {hold_s} s with body-slip change < {TH.LAT_NOSLIP_DEG} deg")


# ---- LiDAR ----------------------------------------------------------------------------------------------------------

def lidar_stats(t_scan, ranges, meta, S, sentinel_tol=0.05) -> Dict:
    """Valid-beam fraction, short-gap dropouts by range (map-free: a gap of <= 5 beams between two returns that agree
    within 10 % is a miss on a continuous surface at their range), and static noise per range from standstill."""
    rmin, rmax = float(meta[5]), float(meta[6])
    R = ranges.astype(np.float32)
    valid = np.isfinite(R) & (R > rmin) & (R < rmax * 0.999)
    out = dict(beams=int(R.shape[1]), fov_deg=float(np.degrees(meta[1] - meta[0])),
               rate_hz=float(1.0 / np.median(np.diff(t_scan))) if len(t_scan) > 2 else float("nan"),
               range_max_msg=rmax, scans=int(len(t_scan)), valid_frac=float(valid.mean()))
    vals_inv = R[~valid & np.isfinite(R)]
    if len(vals_inv):
        u, c = np.unique(np.round(vals_inv, 3), return_counts=True)
        out["invalid_value_mode"] = float(u[np.argmax(c)])
    B_ = R.shape[1]
    idx = np.arange(B_)[None, :].repeat(len(R), 0)
    prev = np.where(valid, idx, -1)
    prev = np.maximum.accumulate(prev, 1)
    nxt = np.where(valid, idx, B_)
    nxt = np.minimum.accumulate(nxt[:, ::-1], 1)[:, ::-1]
    gap = nxt - prev - 1
    inv = ~valid & (prev >= 0) & (nxt < B_) & (gap <= TH.LIDAR_GAP_BEAMS)
    rows = np.arange(len(R))[:, None].repeat(B_, 1)
    rp = np.where(inv, R[rows, np.clip(prev, 0, B_ - 1)], np.nan)
    rn = np.where(inv, R[rows, np.clip(nxt, 0, B_ - 1)], np.nan)
    agree = inv & (np.abs(rp - rn) < 0.1 * 0.5 * (rp + rn))
    r_exp = 0.5 * (rp + rn)
    edges = [0, 2, 4, 6, 8, 10, 30]
    tab = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        miss = int(np.sum(agree & (r_exp >= lo) & (r_exp < hi)))
        hit = int(np.sum(valid & (R >= lo) & (R < hi)))
        tab.append(dict(r=(lo, hi), miss_frac=float(miss / max(miss + hit, 1)), n=miss + hit))
    out["dropout_by_range"] = tab
    # static noise
    v_s = np.interp(t_scan, S["t"], S["v_wheel"])
    st = np.interp(t_scan, S["t"], S["still"].astype(float)) > 0.99
    st &= np.abs(v_s) < TH.LIDAR_STILL_V
    st &= (t_scan >= S["t"][0]) & (t_scan <= S["t"][-1])      # np.interp clamps outside the grid: not 'still'
    # genuinely still: the wheels not turning is not enough (a car being carried or nudged has still wheels)
    gmag = np.sqrt(S["gx"] ** 2 + S["gy"] ** 2 + S["gz"] ** 2)
    st &= np.interp(t_scan, S["t"], smooth(gmag, 20)) < TH.LIDAR_STILL_GYRO
    rate = 1.0 / np.median(np.diff(t_scan)) if len(t_scan) > 2 else 40.0
    nblk = max(int(round(rate * TH.LIDAR_STILL_BLOCK_S)), 10)      # 1 s blocks
    runs = []
    for a, b in _runs(st, nblk):
        runs += [(k, k + nblk) for k in range(a, b - nblk + 1, nblk)]
    sig_tab = []
    if runs:
        sigs, rmed = [], []
        for a, b in runs:
            blk = R[a:b]
            vb = valid[a:b]
            okb = vb.mean(0) > 0.9
            if okb.sum() == 0:
                continue
            med = np.nanmedian(np.where(vb, blk, np.nan), 0)
            mad = np.nanmedian(np.abs(np.where(vb, blk, np.nan) - med), 0)
            sigs.append(1.4826 * mad[okb]); rmed.append(med[okb])
        if sigs:
            sg, rm = np.concatenate(sigs), np.concatenate(rmed)
            for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 10)):
                mm = (rm >= lo) & (rm < hi)
                if mm.sum() > 20:
                    sig_tab.append(dict(r=(lo, hi), sigma_mm=float(np.median(sg[mm]) * 1000), beams=int(mm.sum())))
        out["still_seconds"] = float(sum(b - a for a, b in runs) / max(rate, 1.0))
    out["still_noise"] = sig_tab
    return out
