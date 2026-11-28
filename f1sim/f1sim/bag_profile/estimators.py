"""Two estimators that need a particular kind of recording (FIELD_PROTOCOL T4, T5): coast-down and steering response.

NOT VALIDATED ON THE REAL CAR: no recording of either test exists yet. `tests/test_bag_profile_estimators.py` feeds
them synthetic signals with known values plus noise and checks that the values come back; that is all the evidence
there is. Both say so in their `how` text, and the summary repeats it.

Both return plain dicts with a `status`:
  coastdown       measured | not_identified
  steer_response  separated (pure delay and servo lag told apart) | sum_only (only their sum) | not_identified
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from . import thresholds as TH
from .signals import HZ, L_WHEELBASE, deriv, smooth

NOT_VALIDATED = "실차 미검증 (합성 신호로만 시험함)"


def _runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    m = np.concatenate([[False], mask, [False]])
    d = np.diff(m.astype(int))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(st, en) if b - a >= min_len]


# ---- coast-down ---------------------------------------------------------------------------------------------------

def coastdown(S, sel: Optional[np.ndarray] = None) -> Dict:
    """Rolling + air resistance from decelerating stretches with the motor current ~ 0: decel = c0 + c2 v^2.

    A stretch: |motor current| <= COAST_MAX_CURRENT, v >= COAST_MIN_V, the speed command below the wheel speed (when
    there is one), for at least COAST_MIN_RUN_S; 0.1 s is cut off each end (current transients) and the stretch has to
    lose speed overall. Least squares of -dv/dt on v^2 over all samples; 5-95 % interval from a bootstrap over the
    stretches. 'measured' needs COAST_MIN_TOTAL_S of data in COAST_MIN_SEGMENTS stretches spanning COAST_MIN_V_SPAN of
    speed (without the speed spread c0 and c2 cannot be told apart); otherwise 'not_identified', with the numbers."""
    base = dict(status="not_identified", c0=float("nan"), c2=float("nan"), seconds=0.0, segments=0, v_span=0.0,
                need=dict(seconds=TH.COAST_MIN_TOTAL_S, segments=TH.COAST_MIN_SEGMENTS, v_span=TH.COAST_MIN_V_SPAN),
                validated=False, caveat=NOT_VALIDATED)
    if "i_motor" not in S or not np.isfinite(S["i_motor"]).any():
        return dict(base, how="모터 전류(/sensors/core)가 없어 전류 0 구간을 못 찾음")
    v = smooth(S["v_wheel"], 5)
    cur = np.abs(smooth(S["i_motor"], 5))
    m = np.isfinite(cur) & (cur <= TH.COAST_MAX_CURRENT) & (v >= TH.COAST_MIN_V)
    if "speed_cmd" in S:
        m &= S["speed_cmd"] < v
    if sel is not None:
        m &= sel
    a = -deriv(smooth(S["v_wheel"], 10), 5)
    cut = int(0.1 * HZ)
    xs, ys, ids = [], [], []
    k = 0
    for s0, s1 in _runs(m, int(TH.COAST_MIN_RUN_S * HZ)):
        s0, s1 = s0 + cut, s1 - cut
        if s1 - s0 < 10 or v[s0] - v[s1 - 1] < 0.2:
            continue
        sl = slice(s0, s1)
        ok = np.isfinite(a[sl])
        xs.append(v[sl][ok] ** 2); ys.append(a[sl][ok]); ids.append(np.full(ok.sum(), k)); k += 1
    if not xs:
        why = (f"전류 {TH.COAST_MAX_CURRENT} A 이하로 {TH.COAST_MIN_RUN_S} s 이상 속도가 줄어든 구간이 없음 "
               f"(후보 샘플 {m.sum() / HZ:.1f} s, 구간 합계 필요 {TH.COAST_MIN_TOTAL_S} s)")
        return dict(base, lacking=[why], how="식별 안 됨: " + why)
    x, y, g = np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)
    vs = np.sqrt(x)
    out = dict(base, seconds=float(len(x) / HZ), segments=int(k), v_span=float(np.percentile(vs, 95) - np.percentile(vs, 5)),
               v_range=[float(vs.min()), float(vs.max())])

    def lsq(xx, yy):
        A = np.stack([np.ones_like(xx), xx], 1)
        return np.linalg.lstsq(A, yy, rcond=None)[0]
    c0, c2 = lsq(x, y)
    out.update(c0=float(c0), c2=float(c2))
    lacking = []
    if out["seconds"] < TH.COAST_MIN_TOTAL_S:
        lacking.append(f"구간 합계 {out['seconds']:.1f} s (필요 {TH.COAST_MIN_TOTAL_S} s)")
    if k < TH.COAST_MIN_SEGMENTS:
        lacking.append(f"구간 {k}개 (필요 {TH.COAST_MIN_SEGMENTS}개)")
    if out["v_span"] < TH.COAST_MIN_V_SPAN:
        lacking.append(f"속도 폭 {out['v_span']:.1f} m/s (필요 {TH.COAST_MIN_V_SPAN} m/s)")
    if k >= 2:
        rng = np.random.default_rng(0)
        reps = []
        for _ in range(200):
            pick = rng.integers(0, k, k)
            idx = np.concatenate([np.flatnonzero(g == p) for p in pick])
            reps.append(lsq(x[idx], y[idx]))
        reps = np.array(reps)
        out["c0_ci"] = [float(np.percentile(reps[:, 0], 5)), float(np.percentile(reps[:, 0], 95))]
        out["c2_ci"] = [float(np.percentile(reps[:, 1], 5)), float(np.percentile(reps[:, 1], 95))]
    if lacking:
        out["lacking"] = lacking
        out["how"] = "식별 안 됨: " + ", ".join(lacking)
        return out
    out["status"] = "measured"
    out["how"] = (f"-dv/dt = c0 + c2 v^2, 전류 <= {TH.COAST_MAX_CURRENT} A 감속 구간 {k}개 {out['seconds']:.1f} s, "
                  f"속도 {out['v_range'][0]:.1f}~{out['v_range'][1]:.1f} m/s, 오차는 구간 단위 부트스트랩 5~95 %. " + NOT_VALIDATED)
    return out


# ---- steering step / chirp ----------------------------------------------------------------------------------------

def _segment_spectra(x: np.ndarray, y: np.ndarray, nper: int = 256, step: int = 128):
    w = np.hanning(nper)
    out = []
    for i in range(0, len(x) - nper + 1, step):
        xs = (x[i:i + nper] - x[i:i + nper].mean()) * w
        ys = (y[i:i + nper] - y[i:i + nper].mean()) * w
        X, Y = np.fft.rfft(xs), np.fft.rfft(ys)
        out.append((np.abs(X) ** 2, np.conj(X) * Y, np.abs(Y) ** 2))
    return out


def _frf_model(f, g, td, ts):
    return g * np.exp(-2j * np.pi * f * td) / (1 + 2j * np.pi * f * ts)


def _fit_frf(f, H, w):
    sc = np.median(np.abs(H)) + 1e-12

    def res(p):
        e = (_frf_model(f, *p) - H) / sc * w
        return np.concatenate([e.real, e.imag])
    best = None
    for td0 in (0.02, 0.06, 0.12):
        for ts0 in (0.02, 0.08):
            r = least_squares(res, [np.median(np.abs(H)), td0, ts0], bounds=([0.05, 0.0, 0.001], [5.0, 0.3, 0.3]),
                              x_scale=[0.5, 0.05, 0.05])
            if best is None or r.cost < best.cost:
                best = r
    return best.x


def _chirp_runs(S, sel: Optional[np.ndarray]) -> Tuple[List[Tuple[int, int]], str]:
    """Stretches to analyse: the marked T5 segments if `sel` is given, else windows of CHIRP_WIN_S with the speed above
    CHIRP_MIN_V all through and steering above 1 Hz with rms >= CHIRP_HP_RMS, merged."""
    ok = (S["v_wheel"] > TH.CHIRP_MIN_V) & ~S["still"]
    if "cmd_valid" in S:
        ok &= S["cmd_valid"]
    if sel is not None:
        return _runs(ok & sel, int(4 * HZ)), "표시 토픽 구간"
    hp = S["steer"] - smooth(S["steer"], int(HZ))
    n = int(TH.CHIRP_WIN_S * HZ)
    keep = np.zeros(len(ok), bool)
    for i0 in range(0, len(ok) - n, int(HZ)):
        sl = slice(i0, i0 + n)
        if ok[sl].all() and np.sqrt(np.mean(hp[sl] ** 2)) >= TH.CHIRP_HP_RMS:
            keep[sl] = True
    return _runs(keep, int(4 * HZ)), "자동 인식"


def steer_response(S, sel: Optional[np.ndarray] = None, n_boot: int = 100) -> Dict:
    """Pure command delay (tau_d) and servo/first-order lag (tau_s) from the frequency response between the kinematic
    yaw-rate demand u = v tan(steer_applied) / L and the gyro yaw rate: H(f) = Sxy / Sxx on 2.56 s Hann segments
    (50 % overlap), fitted with g exp(-j 2 pi f tau_d) / (1 + j 2 pi f tau_s) over CHIRP_BAND_LO_HZ..CHIRP_BAND_HI_HZ on
    the bins with coherence >= CHIRP_MIN_COHERENCE. A delay is a phase that keeps growing with frequency, a lag one
    that levels off at 90 deg, so the two come apart only if the input reaches high enough: 'separated' needs input
    power up to CHIRP_MIN_BANDWIDTH_HZ (>= CHIRP_BW_FRAC of the band peak) AND a 5-95 % interval of tau_d (bootstrap over
    segments) narrower than CHIRP_SEP_MAX_CI_S. Otherwise only the sum tau_d + tau_s is reported ('sum_only'); with
    less than CHIRP_MIN_TOTAL_S of data or fewer than CHIRP_MIN_BINS usable bins nothing is ('not_identified').
    tau_s here includes the tyre and yaw build-up that the gyro sees (it is the lag of the whole path)."""
    base = dict(status="not_identified", seconds=0.0, need_seconds=TH.CHIRP_MIN_TOTAL_S, validated=False,
                caveat=NOT_VALIDATED)
    runs, how_sel = _chirp_runs(S, sel)
    secs = sum(b - a for a, b in runs) / HZ
    base.update(seconds=float(secs), runs=len(runs), selection=how_sel)
    if secs < TH.CHIRP_MIN_TOTAL_S:
        return dict(base, how=f"식별 안 됨: 조향 변화가 있는 직진 구간 합계 {secs:.1f} s (필요 {TH.CHIRP_MIN_TOTAL_S} s, {how_sel})")
    u_all = S["v_wheel"] * np.tan(S["steer"]) / L_WHEELBASE
    y_all = S["gz"]
    segs = []
    for a, b in runs:
        segs += _segment_spectra(u_all[a:b], y_all[a:b])
    if len(segs) < 4:
        return dict(base, how="식별 안 됨: 스펙트럼 구간이 4개 미만")
    freqs = np.fft.rfftfreq(256, 1 / HZ)
    band = (freqs >= TH.CHIRP_BAND_LO_HZ) & (freqs <= TH.CHIRP_BAND_HI_HZ)

    def average(idx):
        sxx = np.mean([segs[i][0] for i in idx], 0)
        sxy = np.mean([segs[i][1] for i in idx], 0)
        syy = np.mean([segs[i][2] for i in idx], 0)
        return sxx, sxy, syy

    def fit_from(idx):
        sxx, sxy, syy = average(idx)
        coh = np.abs(sxy) ** 2 / (sxx * syy + 1e-30)
        use = band & (coh >= TH.CHIRP_MIN_COHERENCE)
        if use.sum() < TH.CHIRP_MIN_BINS:
            return None, coh, use, sxx
        H = sxy / (sxx + 1e-30)
        p = _fit_frf(freqs[use], H[use], np.sqrt(coh[use]))
        return p, coh, use, sxx
    allidx = np.arange(len(segs))
    p, coh, use, sxx = fit_from(allidx)
    if p is None:
        return dict(base, bins=int(use.sum()), coherence_mean=float(np.mean(coh[band])),
                    how=(f"식별 안 됨: 일관성 {TH.CHIRP_MIN_COHERENCE} 이상인 주파수 점 {int(use.sum())}개 "
                         f"(필요 {TH.CHIRP_MIN_BINS}개), 대역 평균 일관성 {np.mean(coh[band]):.2f}"))
    g, td, ts = (float(q) for q in p)
    bandpeak = float(sxx[band].max())
    f_bw = float(freqs[band & (sxx >= TH.CHIRP_BW_FRAC * bandpeak)].max())
    rng = np.random.default_rng(0)
    reps = []
    for _ in range(n_boot):
        q, _, _, _ = fit_from(rng.integers(0, len(segs), len(segs)))
        if q is not None:
            reps.append(q)
    reps = np.array(reps) if reps else np.zeros((0, 3))
    out = dict(base, gain=g, bins=int(use.sum()), coherence_mean=float(np.mean(coh[use])), f_bw_hz=f_bw,
               band_hz=[TH.CHIRP_BAND_LO_HZ, TH.CHIRP_BAND_HI_HZ], segments_spectra=len(segs),
               tau_sum=td + ts, tau_d=td, tau_s=ts)
    if len(reps) >= 20:
        out["tau_d_ci"] = [float(np.percentile(reps[:, 1], 5)), float(np.percentile(reps[:, 1], 95))]
        out["tau_s_ci"] = [float(np.percentile(reps[:, 2], 5)), float(np.percentile(reps[:, 2], 95))]
        sm = reps[:, 1] + reps[:, 2]
        out["tau_sum_ci"] = [float(np.percentile(sm, 5)), float(np.percentile(sm, 95))]
        ci_w = out["tau_d_ci"][1] - out["tau_d_ci"][0]
    else:
        ci_w = float("inf")
    out["tau_d_ci_width"] = ci_w
    why = []
    if f_bw < TH.CHIRP_MIN_BANDWIDTH_HZ:
        why.append(f"입력 조향 성분이 {f_bw:.2f} Hz 까지만 있음 (필요 {TH.CHIRP_MIN_BANDWIDTH_HZ} Hz)")
    if not ci_w <= TH.CHIRP_SEP_MAX_CI_S:
        why.append(f"순수 지연 90 % 구간 폭 {ci_w * 1000:.0f} ms (필요 {TH.CHIRP_SEP_MAX_CI_S * 1000:.0f} ms 이하)")
    if why:
        out["status"] = "sum_only"
        out["lacking"] = why
        out["how"] = ("분리 식별 안 됨, 합만 냄: " + "; ".join(why) + f". 합 tau_d+tau_s = {td + ts:.3f} s. " + NOT_VALIDATED)
    else:
        out["status"] = "separated"
        out["how"] = (f"주파수 응답 맞춤 {TH.CHIRP_BAND_LO_HZ}~{TH.CHIRP_BAND_HI_HZ} Hz, 점 {int(use.sum())}개, 입력 대역 {f_bw:.1f} Hz, "
                      f"tau_d 구간 폭 {ci_w * 1000:.0f} ms. " + NOT_VALIDATED)
    return out
