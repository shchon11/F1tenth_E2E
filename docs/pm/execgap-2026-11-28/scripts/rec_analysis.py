"""실행 격차 분해: 기록 분석 (CPU). stage_run.py 기록(<이름>.npz + <이름>_xg.npz)마다
 A. 연속한 두 plan(0.1 s 간격)의 어긋남: 같은 시각(시간 맞춤)과 같은 길 위치(경로 맞춤)에서 옆 방향 차이, 상황별
 B. 주파수 성분: plan이 요구하는 조향·요 각속도 신호의 스펙트럼과 차가 따라갈 수 있는 대역(지연 + 서보 1차 + 선형 단일 트랙)
 C. 한계: plan대로 가려면 필요한 횡가속(타이어 표 대비), 조향 속도(sv_max 대비), 종가속(차마다 a_max·출력 한계, a_brake 대비),
    그리고 지연·서보·속도 루프만으로 plan을 맹목적으로 따라갔을 때 0.5 s 뒤 옆 어긋남(열린 고리)
 D. 세게 닿은 사건(벽·상자) 직전 2 s: 추종 오차와 plan 품질 -> 네 갈래 분류
 E. 한 결정(0.1 s) 동안의 추종 오차 (plan의 0.1 s 점 대비 차의 실제 자리)
인자: 출력json 기록이름 [기록이름 ...]  (기록 폴더 /data/pm-execgap-2026-11-28/rec, 이름 = R0_ideal 등)
기준 분포(사건 분류의 문턱)는 이름이 R0_ideal인 기록에서 잼 -- 첫 인자로 R0_ideal을 줄 것."""
import sys, json, os
import numpy as np
from scipy.signal import welch
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/research/plan-output-2026-11-01/tools")
sys.path.insert(0, "/home/shchon11/F1tenth_E2E/docs/pm/audit-2026-11-28/C_scripts")
from rec_lib import Rec
from hit_split import poly_dist, HW
from matplotlib.path import Path as MPath


def body_pts(x, y, psi, hl, hw):
    """(n * 8, 2) the car's corners and side midpoints at poses (x, y, psi) (n,)"""
    q = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, 0], [-hl, 0], [0, hw], [0, -hw]])
    c, s_ = np.cos(psi)[:, None], np.sin(psi)[:, None]
    return np.stack([x[:, None] + q[:, 0] * c - q[:, 1] * s_, y[:, None] + q[:, 0] * s_ + q[:, 1] * c], -1).reshape(-1, 2)

O = os.environ.get("XG_REC", "/data/pm-execgap-2026-11-28/rec")
LIMN = np.load("/data/pm-execgap-2026-11-28/limits_nominal.npz")
VG, KT = LIMN["v_grid"], LIMN["k_tyre"]
L, LR, SV = float(LIMN["L"]), float(LIMN["lr"]), float(LIMN["sv_max"])
H = 0.025
THROUGH = 0.02                                                      # [m] the plan's body within 2 cm of / inside the obstacle
JUMP_THR = [0.184]                                                  # R0's p99 of the 0.5 s intent change (set in main)


def a_lat_max(v):
    return np.interp(np.abs(v), VG, KT) * v ** 2


def world(xg):
    st = xg["d_st"]                                         # (T, B, 8)
    c, s = np.cos(st[..., 2])[..., None], np.sin(st[..., 2])[..., None]
    x, y = xg["d_x"], xg["d_y"]
    X = st[..., 0:1] + x * c - y * s
    Y = st[..., 1:2] + x * s + y * c
    PSI = st[..., 2:3] + xg["d_psi"]
    return X, Y, PSI


def wrap(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


def seg_dist_many(px, py, qx, qy):
    """distance of points (.., J) to the polyline (.., M) [m]: per row, nearest segment"""
    ax, ay = qx[..., :-1, None], qy[..., :-1, None]               # (.., M-1, 1)
    bx, by = qx[..., 1:, None], qy[..., 1:, None]
    ex, ey = bx - ax, by - ay
    pxx, pyy = px[..., None, :], py[..., None, :]                   # (.., 1, J)
    t = np.clip(((pxx - ax) * ex + (pyy - ay) * ey) / np.maximum(ex * ex + ey * ey, 1e-12), 0, 1)
    dx, dy = pxx - (ax + t * ex), pyy - (ay + t * ey)
    return np.sqrt(dx * dx + dy * dy).min(-2)                       # (.., J)


def pct(x, qs=(50, 90, 99)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(np.percentile(x, q)) if len(x) else float("nan") for q in qs] + [int(len(x))]


# ---------------------------------------------------------------- car bandwidth (linear single track + servo + delay)
def yaw_tf(f, v, d=0.03, tau=0.04, P=None):
    """command -> yaw-rate frequency response, normalised to its steady-state gain (complex, (len(f),))"""
    P = P or {k[4:]: float(LIMN[k]) for k in LIMN.files if k.startswith("nom_")}
    m, Iz, lf, lr = P["m"], P["Iz"], P["lf"], P["lr"]; Lw = lf + lr; g = 9.81
    Fzf, Fzr = m * g * lr / Lw, m * g * lf / Lw
    Cf = P["mu"] * P["mu_f_scale"] * Fzf * P["B_f"] * P["C_f"]; Cr = P["mu"] * Fzr * P["B_r"] * P["C_r"]
    A = np.array([[-(Cf + Cr) / (m * v), -v - (Cf * lf - Cr * lr) / (m * v)],
                  [-(Cf * lf - Cr * lr) / (Iz * v), -(Cf * lf ** 2 + Cr * lr ** 2) / (Iz * v)]])
    Bv = np.array([Cf / m, Cf * lf / Iz])
    out = []
    for fi in np.r_[0.0, f]:
        s = 2j * np.pi * fi
        G = np.linalg.solve(s * np.eye(2) - A, Bv)[1]
        out.append(G * np.exp(-s * d) / (1 + s * tau))
    out = np.array(out)
    return out[1:] / out[0]


def bandwidth(v, d, tau):
    f = np.linspace(0.05, 20, 800)
    Hf = yaw_tf(f, v, d, tau)
    err = np.abs(1 - Hf)                                            # relative error of following a sine at f
    ph = -np.unwrap(np.angle(Hf))
    f50 = float(f[np.argmax(err >= 0.5)]) if (err >= 0.5).any() else np.inf
    f30 = float(f[np.argmax(err >= 0.3)]) if (err >= 0.3).any() else np.inf
    f45 = float(f[np.argmax(ph >= np.pi / 4)]) if (ph >= np.pi / 4).any() else np.inf
    f3 = float(f[np.argmax(np.abs(Hf) <= 0.707)]) if (np.abs(Hf) <= 0.707).any() else np.inf
    return dict(v=v, d=d, tau=tau, f_err30=f30, f_err50=f50, f_phase45=f45, f_3db=f3)


# ---------------------------------------------------------------- per record
def analyse(name, ref=None):
    R = Rec(f"{O}/{name}.npz"); xg = np.load(f"{O}/{name}_xg.npz")
    T, B = xg["d_x"].shape[:2]
    X, Y, PSI = world(xg)
    V = xg["d_v"].astype(np.float32); DEL = xg["d_delta"].astype(np.float32)
    st = xg["d_st"]
    rs = R.rs[:T]
    d = R.d[:T]                                                     # (T, B, 6) onsets wall/box/car, hard wall/box/car
    hard = (d[..., 3] + d[..., 4]) > 0
    any_c = (d[..., 0] + d[..., 1]) > 0
    vcar = st[..., 3]
    out = {"name": name, "stage": str(xg["stage"]), "T": int(T), "B": int(B)}

    # ---- situations per decision
    kap = np.tan(DEL) / L
    kmean = np.abs(kap[..., :21]).mean(-1)                          # first 0.5 s
    near_box = np.zeros((T, B), bool)
    for t in range(0, T):
        p = R.pose[t]; sid = R.sid[t]                               # (B, C, 3)
        dx = p[..., 0] - st[t, :, None, 0]; dy = p[..., 1] - st[t, :, None, 1]
        ahead = dx * np.cos(st[t, :, None, 2]) + dy * np.sin(st[t, :, None, 2])
        dist = np.hypot(dx, dy)
        near_box[t] = ((sid >= 0) & (dist < 3.0) & (ahead > -0.3)).any(-1)
    pre = np.zeros((T, B), bool)
    for t, b in zip(*np.nonzero(hard)):
        pre[max(0, t - 10):t, b] = True
    bad = rs | any_c | (vcar < 1.0)
    sit = {"전체": np.ones((T, B), bool), "직선(|곡률|<0.1)": kmean < 0.1, "코너(|곡률|>0.4)": kmean > 0.4,
           "상자 3 m 안": near_box, "세게 닿기 직전 1 s": pre}

    # ---- A. consecutive plans
    J = 37                                                          # tau = 0 .. 0.9 s on the 0.025 s grid
    xa, ya, pa = X[:-1, :, 4:4 + J], Y[:-1, :, 4:4 + J], PSI[:-1, :, 4:4 + J]   # plan k at t + 0.1 + tau
    xb, yb = X[1:, :, :J], Y[1:, :, :J]                                         # plan k+1 at t + 0.1 + tau
    lat = -(xb - xa) * np.sin(pa) + (yb - ya) * np.cos(pa)          # + = new plan left of the old one
    lat_rel = lat - lat[..., :1]                                    # the start offset (tracking) removed
    ok = ~(bad[:-1] | bad[1:])
    # path-matched: distance of the new plan's points to the old plan's polyline (its whole 1 s)
    arc = np.full((T - 1, B, 3), np.nan, np.float32)
    for t in range(T - 1):
        dd = seg_dist_many(X[t + 1, :, [12, 20, 36]].T, Y[t + 1, :, [12, 20, 36]].T, X[t], Y[t])  # (B, 3)
        arc[t] = dd
    # shape change: plan k re-anchored at its own 0.1 s pose (where the car was meant to be) vs plan k+1 in the car's frame --
    # the planner's change of intent, with where the car actually ended up (tracking) taken out
    dx0, dy0, dp0 = xg["d_x"], xg["d_y"], xg["d_psi"]
    c4, s4 = np.cos(dp0[:-1, :, 4:5]), np.sin(dp0[:-1, :, 4:5])
    ux = dx0[:-1, :, 4:4 + J] - dx0[:-1, :, 4:5]; uy = dy0[:-1, :, 4:4 + J] - dy0[:-1, :, 4:5]
    sx = ux * c4 + uy * s4; sy = -ux * s4 + uy * c4                 # plan k's remainder, its 0.1 s pose = origin
    shape_t = dy0[1:, :, :J] - sy                                   # time matched, lateral (+ left)
    shape_p = np.full((T - 1, B, 3), np.nan, np.float32)
    for t in range(T - 1):
        shape_p[t] = seg_dist_many(dx0[t + 1, :, [12, 20, 36]].T, dy0[t + 1, :, [12, 20, 36]].T, sx[t], sy[t])
    A = {}
    for sname, m in sit.items():
        mm = ok & m[:-1]
        A[sname] = {f"time_lat_{tau}s": pct(np.abs(lat[..., j])[mm]) for tau, j in (("0.3", 12), ("0.5", 20), ("0.9", 36))}
        A[sname].update({f"time_latrel_{tau}s": pct(np.abs(lat_rel[..., j])[mm]) for tau, j in (("0.3", 12), ("0.5", 20), ("0.9", 36))})
        A[sname].update({f"path_{tau}s": pct(arc[..., i][mm]) for i, tau in enumerate(("0.3", "0.5", "0.9"))})
        A[sname].update({f"shape_time_{tau}s": pct(np.abs(shape_t[..., j])[mm]) for tau, j in (("0.3", 12), ("0.5", 20), ("0.9", 36))})
        A[sname].update({f"shape_path_{tau}s": pct(shape_p[..., i][mm]) for i, tau in enumerate(("0.3", "0.5", "0.9"))})
        A[sname]["share"] = float(mm.sum() / max(ok.sum(), 1))
    out["A"] = A
    jump5 = np.where(ok, shape_p[..., 1], np.nan)                   # (T-1, B), for events: the intent change at 0.5 s

    # ---- E. one decision's tracking error: car at t+1 vs plan t's 0.1 s point
    ex = st[1:, :, 0] - X[:-1, :, 4]; ey = st[1:, :, 1] - Y[:-1, :, 4]
    p4 = PSI[:-1, :, 4]
    e1_lat = -ex * np.sin(p4) + ey * np.cos(p4); e1_lon = ex * np.cos(p4) + ey * np.sin(p4)
    e1_head = wrap(st[1:, :, 2] - p4); e1_v = st[1:, :, 3] - V[:-1, :, 4]
    # 0.5 s ahead: car at t+5 vs plan t's 0.5 s point
    ex5 = st[5:, :, 0] - X[:-5, :, 20]; ey5 = st[5:, :, 1] - Y[:-5, :, 20]; p20 = PSI[:-5, :, 20]
    e5_lat = -ex5 * np.sin(p20) + ey5 * np.cos(p20)
    e5_head = wrap(st[5:, :, 2] - p20); e5_v = st[5:, :, 3] - V[:-5, :, 20]
    ok5 = ~(bad[:-5] | bad[1:-4] | bad[2:-3] | bad[3:-2] | bad[4:-1] | bad[5:])
    out["E"] = {"e1_lat": pct(np.abs(e1_lat)[ok]), "e1_lon": pct(np.abs(e1_lon)[ok]), "e1_head_deg": pct(np.degrees(np.abs(e1_head))[ok]),
                "e1_v": pct(np.abs(e1_v)[ok]), "e5_lat": pct(np.abs(e5_lat)[ok5]), "e5_head_deg": pct(np.degrees(np.abs(e5_head))[ok5]),
                "e5_v": pct(np.abs(e5_v)[ok5])}
    # the body's slip angle on every control step (car moving > 2 m/s): sliding / spinning
    cs = xg["c_st"]; mv = cs[..., 3] > 2.0
    beta = np.degrees(np.abs(np.arctan2(cs[..., 4], np.maximum(cs[..., 3], 0.3))))[mv]
    out["E"]["slip_deg"] = pct(beta); out["E"]["slip_gt10"] = float(np.mean(beta > 10)); out["E"]["slip_gt20"] = float(np.mean(beta > 20))
    # sessions with at least one 20-degree slide
    out["E"]["sess_slide20"] = float(np.mean([(np.degrees(np.abs(np.arctan2(cs[:, b, 4], np.maximum(cs[:, b, 3], 0.3))))[cs[:, b, 3] > 2.0] > 20).any()
                                               for b in range(cs.shape[1])]))
    # signed lateral bias in corners (+ = outside of the turn)
    turn = np.sign(PSI[:-1, :, 20] - PSI[:-1, :, 0])
    cor = ok & (kmean[:-1] > 0.4)
    out["E"]["e1_lat_outside_corner_mean_cm"] = float(100 * np.mean((-e1_lat * turn)[cor])) if cor.any() else float("nan")

    # ---- B. spectra of the requested steering / yaw rate: stitched first 0.1 s of every plan (40 Hz)
    d_req = DEL[:, :, 1:5].transpose(1, 0, 2).reshape(B, -1)       # (B, 4T)
    v_req = V[:, :, 1:5].transpose(1, 0, 2).reshape(B, -1)
    r_req = v_req * np.tan(d_req) / L
    r_car = xg["c_st"][..., 5].T                                     # (B, 4T) the car's yaw rate
    d_car = xg["c_st"][..., 6].T
    good_rows = ~(rs.any(0))
    def psd(sig):
        f, p = welch(sig[good_rows], fs=40.0, nperseg=256, detrend="linear", axis=-1)
        return f, p.mean(0)
    fB, P_d = psd(d_req); _, P_r = psd(r_req); _, P_rc = psd(r_car); _, P_dc = psd(d_car)
    out["B"] = {"f": fB.tolist(), "psd_steer_req": P_d.tolist(), "psd_yaw_req": P_r.tolist(), "psd_yaw_car": P_rc.tolist(),
                "psd_steer_car": P_dc.tolist()}
    def frac_above(P, fc):
        return float(P[fB >= fc].sum() / P.sum())
    out["B"]["frac_above"] = {f"{fc}": {"steer_req": frac_above(P_d, fc), "yaw_req": frac_above(P_r, fc),
                                        "yaw_car": frac_above(P_rc, fc)} for fc in (1.0, 2.0, 3.0, 4.0, 5.0, 8.0)}
    # rms of the requested steering above / below a cut (rad) via FFT high-pass on each session
    def hp_rms(sig, fc):
        F = np.fft.rfft(sig[good_rows] - sig[good_rows].mean(1, keepdims=True), axis=1)
        fr = np.fft.rfftfreq(sig.shape[1], 1 / 40.0)
        F[:, fr < fc] = 0
        return float(np.sqrt(np.mean(np.fft.irfft(F, n=sig.shape[1], axis=1) ** 2)))
    out["B"]["rms_steer_req_rad"] = float(np.sqrt(np.mean((d_req[good_rows] - d_req[good_rows].mean(1, keepdims=True)) ** 2)))
    out["B"]["rms_steer_req_above"] = {f"{fc}": hp_rms(d_req, fc) for fc in (2.0, 3.0, 5.0)}
    out["B"]["rms_yaw_req_above"] = {f"{fc}": hp_rms(r_req, fc) for fc in (2.0, 3.0, 5.0)}
    # the plan's own steering change from decision to decision: kink at the 0.1 s boundary (rate change) [rad/s]
    rate_first = (DEL[:, :, 4] - DEL[:, :, 0]) / 0.1
    kink = np.abs(rate_first[1:] - rate_first[:-1])
    out["B"]["kink_steer_rate"] = pct(kink[ok])
    out["B"]["sign_flip_rate_share"] = float(((np.sign(rate_first[1:]) * np.sign(rate_first[:-1]) < 0)
                                              & (np.abs(rate_first[1:]) > 0.1) & (np.abs(rate_first[:-1]) > 0.1))[ok].mean())

    # ---- C. limits the plan asks for (whole plan 1 s and first 0.1 s)
    vm = 0.5 * (V[..., 1:] + V[..., :-1]); dm = 0.5 * (DEL[..., 1:] + DEL[..., :-1])
    alat = vm ** 2 * np.tan(dm) / L
    r_lat = np.abs(alat) / np.maximum(a_lat_max(vm), 0.3)
    r_sv = np.abs(np.diff(DEL, axis=-1)) / H / SV
    acc = np.diff(V, axis=-1) / H
    amax = xg["P_a_max"][None, :, None] * np.minimum(1.0, xg["P_v_switch"][None, :, None] / np.maximum(np.abs(vm), 1e-3))
    r_acc = np.where(acc >= 0, acc / amax, -acc / xg["P_a_brake"][None, :, None])
    r_comb = np.sqrt(alat ** 2 + acc ** 2) / np.maximum(a_lat_max(vm), 0.3)
    okp = ~bad
    C = {}
    for part, sl in (("first0.1s", slice(0, 4)), ("whole1s", slice(0, 40))):
        C[part] = {}
        for k, rr in (("lat", r_lat), ("steer_rate", r_sv), ("long", r_acc), ("drive", np.where(acc >= 0, r_acc, 0.0)),
                      ("brake", np.where(acc < 0, r_acc, 0.0)), ("combined", r_comb)):
            mx = rr[..., sl].max(-1)[okp]
            C[part][k] = {"p50": float(np.median(mx)), "p90": float(np.percentile(mx, 90)), "p99": float(np.percentile(mx, 99)),
                          "share_ge_0.9": float((mx >= 0.9).mean()), "share_gt_1.0": float((mx > 1.0 + 1e-3).mean())}
    # lags only, open loop: the plan's command through this car's delay, servo (1st order + rate) and speed loop, a kinematic
    # bicycle (no tyres) -> lateral / speed miss at 0.5 s and 1.0 s
    sub = np.nonzero(okp.reshape(-1))[0]
    rng = np.random.default_rng(0)
    sub = rng.choice(sub, min(10000, len(sub)), replace=False)
    tt, bb = np.unravel_index(sub, (T, B))
    cmd_d = dm[tt, bb]; cmd_v = vm[tt, bb]                         # (n, 40) the command held each 0.025 s
    n = len(sub); dt_ = 0.001
    lagd = {}
    for var in ("none", "delay", "delay+servo", "delay+servo+motor"):
        dl = xg["P_cmd_delay"][bb] if var != "none" else np.zeros(n)
        tau = xg["P_servo_tau"][bb]; mt = xg["P_motor_tau"][bb]
        x = np.zeros(n); y = np.zeros(n); ps = np.zeros(n); vv = V[tt, bb, 0].astype(float).copy(); dd = DEL[tt, bb, 0].astype(float).copy()
        res = {}
        for k in range(1000):
            tnow = k * dt_
            j_eff = np.floor((tnow - dl + 1e-9) / H).astype(int)       # the command in effect after the delay
            cd = np.where(j_eff >= 0, cmd_d[np.arange(n), np.clip(j_eff, 0, 39)], DEL[tt, bb, 0])
            cv = np.where(j_eff >= 0, cmd_v[np.arange(n), np.clip(j_eff, 0, 39)], V[tt, bb, 0])
            if "servo" in var:
                rate = np.clip((cd - dd) / tau, -SV, SV); dd = dd + rate * dt_
            else:
                dd = cd
            if "motor" in var:
                a_ = (cv - vv) / mt
                a_ = np.minimum(a_, xg["P_a_max"][bb] * np.minimum(1.0, xg["P_v_switch"][bb] / np.maximum(np.abs(vv), 1e-3)))
                a_ = np.maximum(a_, -xg["P_a_brake"][bb]); vv = vv + a_ * dt_
            else:
                vv = cv
            td = np.tan(dd); beta = np.arctan(LR / L * td); r = vv * np.cos(beta) * td / L
            mid = ps + 0.5 * r * dt_
            x += (vv * np.cos(mid) - vv * np.tan(beta) * np.sin(mid)) * dt_
            y += (vv * np.sin(mid) + vv * np.tan(beta) * np.cos(mid)) * dt_
            ps += r * dt_
            if k + 1 in (200, 500, 1000):
                j = (k + 1) // 25
                px_, py_, pp_ = xg["d_x"][tt, bb, j], xg["d_y"][tt, bb, j], xg["d_psi"][tt, bb, j]
                lat_ = -(x - px_) * np.sin(pp_) + (y - py_) * np.cos(pp_)
                res[f"{(k + 1) / 1000:.1f}s"] = {"lat_abs": pct(np.abs(lat_)), "v_abs": pct(np.abs(vv - V[tt, bb, j])),
                                                 "pos": pct(np.hypot(x - px_, y - py_))}
        lagd[var] = res
    C["lags_open_loop"] = lagd
    out["C"] = C
    out["_ok"] = ok; out["_jump5"] = jump5; out["_e5"] = (e5_lat, e5_head, e5_v, ok5); out["_e1"] = (e1_lat, e1_v, ok)
    out["_rlat"] = r_lat; out["_rsv"] = r_sv; out["_racc"] = r_acc

    # ---- D. events (hard wall / box onsets), the 2 s before
    ev = []
    for kind, col in (("wall", 3), ("box", 4)):
        for t, b in zip(*np.nonzero(d[:, :, col] > 0)):
            if t < 15 or rs[max(0, t - 20):t + 1, b].any():
                continue
            e = {"kind": kind, "t": int(t), "b": int(b), "v": float(st[t, b, 3])}
            polys = R.boxes(t, b) if kind == "box" else None
            if kind == "box" and not polys:
                continue
            def clear(k0, upto, center=False):
                """[m] the plan k0's body (corners + side midpoints along its dense poses up to `upto` s) to the obstacle:
                wall = the smallest wall distance of those points (<= 0: the body overlaps a wall); box = their distance to
                the hit box polygon (negative inside). center=True: audit C's measure (centre path - half width)"""
                j1 = int(min(40, max(4, round(upto / H))))
                j0 = 0 if center else 4                                     # the body: from the plan's 0.1 s pose on
                cx, cy, cp = X[k0, b, j0:j1 + 1].astype(float), Y[k0, b, j0:j1 + 1].astype(float), PSI[k0, b, j0:j1 + 1].astype(float)
                if center:
                    pts = np.stack([cx, cy], 1)
                else:
                    pts = body_pts(cx, cy, cp, R.hl, R.hw)
                if kind == "wall":
                    return float(R.wall_dist(int(R.tid[t, b]), pts).min()) - (HW if center else 0.0)
                car = st[min(t + 1, T - 1), b, :2]
                kk = int(np.argmin([poly_dist(car[None], p)[0] for p in polys]))
                dd = poly_dist(pts, polys[kk])
                if not center:
                    dd = np.where(MPath(polys[kk]).contains_points(pts), -dd, dd)
                return float(dd.min()) - (HW if center else 0.0)
            e["clear_t5"] = clear(t - 5, 0.5 + 0.15)
            e["clear_t10"] = clear(t - 10, 1.0)
            e["clear_t1"] = clear(t - 1, 0.25)
            e["clear_c_t5"] = clear(t - 5, 0.5 + 0.15, center=True)
            # which side the obstacle is on, seen from plan t-5's point at the contact time (the contact point the sim recorded,
            # else the nearest obstacle point); the car's deviation from that plan toward it [m]
            hx, hy = float(R.z["held"][t, b, 1]), float(R.z["held"][t, b, 2])
            if not (np.isfinite(hx) and (abs(hx) + abs(hy)) > 0):
                hx, hy = float(st[min(t + 1, T - 1), b, 0]), float(st[min(t + 1, T - 1), b, 1])
            pp = PSI[t - 5, b, 20]
            side = np.sign(-(hx - X[t - 5, b, 20]) * np.sin(pp) + (hy - Y[t - 5, b, 20]) * np.cos(pp))
            e["dev_toward"] = float(e5_lat[t - 5, b] * side)
            # the part of it that execution made: the one-decision misses (car vs each plan's own 0.1 s point) of the last 0.5 s,
            # summed, toward the obstacle -- re-planning drift is not in it (it is 0 under exact execution)
            e["exec_toward"] = float(np.sum(e1_lat[t - 5:t, b]) * side)
            e["plan_touch"] = {f"t-{d_}": bool(clear(t - d_, d_ * 0.1 + 0.15) < THROUGH) for d_ in (10, 7, 4, 1)}
            e["start_touch_t5"] = bool(float(R.wall_dist(int(R.tid[t, b]), body_pts(np.array([st[t - 5, b, 0]]), np.array([st[t - 5, b, 1]]),
                                                                                         np.array([st[t - 5, b, 2]]), R.hl, R.hw)).min()) < THROUGH) if kind == "wall" else False
            e["jump_max"] = float(np.nanmax(jump5[t - 5:t, b])) if np.isfinite(jump5[t - 5:t, b]).any() else float("nan")   # last 0.5 s
            _jw = jump5[t - 10:t, b]
            e["w_jump"] = float(np.mean(_jw[np.isfinite(_jw)] > JUMP_THR[0])) if np.isfinite(_jw).any() else float("nan")
            e["w_follow"] = float(np.mean(np.abs(e1_lat[t - 10:t, b]) > 0.10))
            _brk = np.where(np.diff(V[t - 10:t, b, :21], axis=-1) < 0, r_acc[t - 10:t, b, :20], 0.0).max(-1)
            e["w_demand"] = float(np.mean((r_lat[t - 10:t, b, :20].max(-1) > 1.0 + 1e-3) | (r_sv[t - 10:t, b, :20].max(-1) > 1.0 + 1e-3)
                                          | (_brk > 1.15)))
            e["e5_lat"] = float(e5_lat[t - 5, b]); e["e5_head_deg"] = float(np.degrees(e5_head[t - 5, b])); e["e5_v"] = float(e5_v[t - 5, b])
            e["e1_lat_mean2s"] = float(np.abs(e1_lat[t - 20:t, b]).mean()) if t >= 20 else float(np.abs(e1_lat[:t, b]).mean())
            e["e1_lat_max1s"] = float(np.abs(e1_lat[t - 10:t, b]).max())
            e["rlat_max"] = float(r_lat[t - 10:t, b, :20].max()); e["rsv_max"] = float(r_sv[t - 10:t, b, :20].max())
            e["racc_max"] = float(r_acc[t - 10:t, b, :20].max())
            e["rbrake_max"] = float(np.where(np.diff(V[t - 10:t, b, :21], axis=-1) < 0, r_acc[t - 10:t, b, :20], 0.0).max())
            # what showed first in the 2 s before (decision index k = t-20 .. t): a large one-decision miss (car vs the plan's
            # 0.1 s point > 10 cm), a jump of intent (plan k vs plan k-1 re-anchored, path at 0.5 s > R0's p99), the plan k
            # through the obstacle (path to the contact time + 0.15 s within half-width + 5 cm), a plan beyond the limits
            # (tyre table, servo rate, or braking > 1.15 x this car's in its first 0.5 s)
            ks = np.arange(max(1, t - 20), t + 1)
            first = {}
            for k in ks:
                if "follow" not in first and k < t and abs(e1_lat[k, b]) > 0.10:
                    first["follow"] = k
                if "jump" not in first and np.isfinite(jump5[k - 1, b]) and jump5[k - 1, b] > JUMP_THR[0]:
                    first["jump"] = k
                if "through" not in first and clear(k, (t - k) * 0.1 + 0.15) < THROUGH:
                    first["through"] = k
                if "demand" not in first:
                    brk = np.where(np.diff(V[k, b, :21]) < 0, r_acc[k, b, :20], 0.0).max()
                    if r_lat[k, b, :20].max() > 1.0 + 1e-3 or r_sv[k, b, :20].max() > 1.0 + 1e-3 or brk > 1.15:
                        first["demand"] = k
            e["first"] = {kk: float((t - vv) * 0.1) for kk, vv in first.items()}     # [s] before the contact
            ev.append(e)
    out["events"] = ev
    # base rates per decision away from contacts (not in the 2 s before a hard contact, no restart / contact / slow): how often
    # each sign shows in ordinary driving -> the chance of at least one in a 1 s / 2 s window
    pre2 = np.zeros((T, B), bool)
    for t, b in zip(*np.nonzero(hard)):
        pre2[max(0, t - 20):t + 1, b] = True
    calm = ~bad & ~pre2
    calm[0] = False
    _j = jump5[calm[1:]]; _j = _j[np.isfinite(_j)]
    base = {"jump": float(np.mean(_j > JUMP_THR[0])),
            "follow": float(np.mean(np.abs(e1_lat)[calm[:-1]] > 0.10)),
            "demand": float(np.mean(((r_lat[..., :20].max(-1) > 1.0 + 1e-3) | (r_sv[..., :20].max(-1) > 1.0 + 1e-3)
                                     | (np.where(np.diff(V[..., :21], axis=-1) < 0, r_acc[..., :20], 0.0).max(-1) > 1.15))[calm]))}
    # the plan's body touching a wall within its 1 s (a sample of calm decisions)
    idx = np.argwhere(calm)
    idx = idx[np.random.default_rng(1).choice(len(idx), min(3000, len(idx)), replace=False)]
    wt = []
    for t, b in idx:
        pts = body_pts(X[t, b].astype(float), Y[t, b].astype(float), PSI[t, b].astype(float), R.hl, R.hw)
        wt.append(float(R.wall_dist(int(R.tid[t, b]), pts).min()))
    wt = np.array(wt)
    base["through_wall"] = float(np.mean(wt < THROUGH)); base["wall_within_5cm"] = float(np.mean(wt < 0.05))
    out["D_base"] = base
    return out


def first_sign(ev):
    """the category that showed first (ties: through > jump > demand > follow); none within 2 s -> other"""
    out = []
    for e in ev:
        f = e["first"]
        if not f:
            out.append("기타(2 s 안에 신호 없음)"); continue
        m = max(f.values())
        for k, nm in (("through", "계획이 장애물을 지나감"), ("jump", "계획이 직전에 튐"), ("demand", "계획이 못 따라갈 것을 요구"),
                      ("follow", "계획은 괜찮았는데 못 따라감")):
            if k in f and f[k] >= m - 1e-9:
                out.append(nm); break
    return out


def classify(ev, thr):
    """exclusive, in this order:
    1 plan made 1 s before already put the body into the obstacle (from its 0.1 s pose on) -> the plan chose it
    2 execution moved the car > 10 cm toward the obstacle in the last 0.5 s (sum of the one-decision misses, each against its own
      plan's 0.1 s point; 0 under exact execution) -> it did not follow;
      with a plan beyond the car's limits in the last 1 s (tyre table, servo rate, braking > 1.15 x this car's) -> 'demanded'
    3 a later plan (0.7 / 0.4 / 0.1 s before) put the body into it after a large intent change (> R0's p99) -> 'jumped'
    4 a later plan put the body into it without one -> the plan chose it (late)
    5 none of these"""
    lab = []
    for e in ev:
        early = e["plan_touch"]["t-10"]
        late = any(e["plan_touch"][k] for k in ("t-7", "t-4", "t-1"))
        dev = e["exec_toward"] > 0.10
        demand = (e["rlat_max"] > 1.0 + 1e-3) or (e["rsv_max"] > 1.0 + 1e-3) or (e["rbrake_max"] > 1.15)
        jump = np.isfinite(e["jump_max"]) and e["jump_max"] > thr["jump"]
        if early:
            c = "계획이 장애물을 지나감"
        elif dev and demand:
            c = "계획이 못 따라갈 것을 요구"
        elif dev:
            c = "계획은 괜찮았는데 못 따라감"
        elif late and jump:
            c = "계획이 직전에 튐"
        elif late:
            c = "계획이 장애물을 지나감"
        else:
            c = "기타(계획 여유 2 cm 이상·추종 오차 작음)"
        lab.append((c, dict(through=bool(early or late), through_early=bool(early), jump=bool(jump), demand=bool(demand),
                            follow=bool(dev), brake_short=bool(e["rbrake_max"] > 1.0 + 1e-3), lat_over=bool(e["rlat_max"] > 1.0 + 1e-3),
                            start_touch=bool(e["start_touch_t5"]))))
    return lab


if __name__ == "__main__":
    outj = sys.argv[1]; names = sys.argv[2:]
    res = {}
    ref = None
    for nm in names:
        if not os.path.exists(f"{O}/{nm}_xg.npz"):
            print(f"skip {nm} (no record)"); continue
        r = analyse(nm)
        if nm.startswith("R0") or ref is None and nm == names[0]:
            ok = r["_ok"]
            e5_lat, _, _, ok5 = r["_e5"]; e1_lat, _, ok1 = r["_e1"]
            ref = {"jump": float(np.nanpercentile(r["_jump5"][ok], 99)),
                   "e5_lat": float(np.percentile(np.abs(e5_lat)[ok5], 95)), "e1_lat": 0.10}
            print(f"[ref from {nm}] {ref}")
            JUMP_THR[0] = ref["jump"]
        lab = classify(r["events"], ref)
        cnt = {}
        for (c, fl), e in zip(lab, r["events"]):
            cnt.setdefault(e["kind"], {}).setdefault(c, 0); cnt[e["kind"]][c] += 1
            e["class"] = c; e.update({"f_" + k: v for k, v in fl.items()})
        r["D_counts"] = cnt; r["D_ref"] = ref
        fs = first_sign(r["events"]); cnt2 = {}
        for c, e in zip(fs, r["events"]):
            e["first_class"] = c
            cnt2.setdefault(e["kind"], {}).setdefault(c, 0); cnt2[e["kind"]][c] += 1
        r["D_first"] = cnt2
        fl_tot = {k: {f: int(sum(e["f_" + f] for e in r["events"] if e["kind"] == k)) for f in ("through", "through_early", "jump", "demand", "follow", "brake_short", "lat_over", "start_touch")}
                  for k in ("wall", "box")}
        r["D_flags"] = fl_tot
        r["D_n"] = {k: int(sum(e["kind"] == k for e in r["events"])) for k in ("wall", "box")}
        for k in list(r):
            if k.startswith("_"):
                r.pop(k)
        res[nm] = r
        A0 = r["A"]["전체"]; C0 = r["C"]; E0 = r["E"]
        print(f"== {nm} ({r['stage']}): A time|lat| 0.5 s p50/p90/p99 {np.round(A0['time_lat_0.5s'][:3], 3)} rel {np.round(A0['time_latrel_0.5s'][:3], 3)} "
              f"path 0.5 s {np.round(A0['path_0.5s'][:3], 3)} shape(t) {np.round(A0['shape_time_0.5s'][:3], 3)} shape(p) {np.round(A0['shape_path_0.5s'][:3], 3)} | E e1 lat {np.round(E0['e1_lat'][:3], 3)} e5 lat {np.round(E0['e5_lat'][:3], 3)} slip p50/90/99 {np.round(E0['slip_deg'][:3], 1)} >20deg {E0['slip_gt20']:.4f} sess {E0['sess_slide20']:.2f} "
              f"| B >3 Hz steer {r['B']['frac_above']['3.0']['steer_req']:.3f} yaw {r['B']['frac_above']['3.0']['yaw_req']:.3f} "
              f"| C lat p99 {C0['whole1s']['lat']['p99']:.2f} >1 {C0['whole1s']['lat']['share_gt_1.0']:.3f} long>1 {C0['whole1s']['long']['share_gt_1.0']:.3f} "
              f"sv>1 {C0['whole1s']['steer_rate']['share_gt_1.0']:.3f} | lags 0.5 s lat p50/p90 " + " ".join(f"{k_}:{np.round(v_['0.5s']['lat_abs'][:2], 3)}" for k_, v_ in C0['lags_open_loop'].items()), flush=True)
        print(f"   D n {r['D_n']} flags {r['D_flags']} classes {r['D_counts']}", flush=True)
        print(f"   D first sign {r['D_first']}", flush=True)
        print(f"   D base rates per decision {r.get('D_base')}", flush=True)
        for kind in ("wall", "box"):
            E = [e for e in r["events"] if e["kind"] == kind]
            if E:
                print(f"   {kind}: per-decision rate in the last 1 s " + " | ".join(
                    f"{k} {np.nanmean([e['w_' + k] for e in E]):.3f}" for k in ("jump", "follow", "demand")), flush=True)
    res["_bandwidth"] = [bandwidth(v, dl, tau) for v in (3.0, 5.0, 7.0) for dl, tau in ((0.03, 0.04), (0.02, 0.025), (0.05, 0.07))]
    for bw in res["_bandwidth"]:
        print("  bandwidth", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in bw.items()})
    json.dump(res, open(outj, "w"), default=float, ensure_ascii=False)
