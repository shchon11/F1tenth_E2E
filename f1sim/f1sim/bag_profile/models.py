"""Lateral vehicle models for open-loop replay of recorded commands, simplest first.

  persistence   state held (a scale for the errors)
  lin           (a) the audit's 4-value model: r = v tan(k d(t - tau) + b) / (L + K v^2); algebraic, never spins
  st            (b) dynamic single track, saturating tyres per axle (slope s per unit load, peak mu, shape C of
                    sin(C atan(B a)) with B = s / (mu C)), servo first-order lag + pure delay, static axle loads
  st + lt       (d) the same with longitudinal load transfer (h a_x / L), CoG height fixed at the spec sheet's
  sim           (c) the simulator's own physics chain (f1sim.dynamics / actuators), any parameter overridable

(a), (b), (d) share one longitudinal model -- a first-order speed loop on the delayed command, accelerations clipped
-- so they differ only laterally. (c) runs the simulator's own VESC loop and wheel model.

Everything is per unit mass: with tyre forces written per unit normal load the mass cancels, and only Iz/m (the yaw
radius of gyration squared), lf / lr and h remain as fixed geometry the bags cannot identify.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .signals import HZ, L_WHEELBASE as L
from .windows import WARM, HOR

G = 9.81
LF, LR = 0.15875, 0.17145          # spec sheet (racecar_f110.ini); wheelbase matches the car, split unmeasured
IZ_M = 0.04712 / 3.74              # yaw inertia per unit mass [m^2] (f1tenth_gym values; never measured)
H_COG = 0.074                      # [m] spec sheet

SPEED_DEFAULT = dict(tau_m=0.15, tau_v=0.10, a_a=4.0, a_b=4.0, v_s=10.0)
LIN_DEFAULT = dict(k=1.0, tau=0.07, b=0.0, kus=0.0)
ST_DEFAULT = dict(k=1.0, b=0.0, tau_d=0.05, tau_s=0.04, sf=3.5, mf=1.0, Cf=1.3, sr=5.5, mr=1.1, Cr=1.3,
                  iz_m=IZ_M, lf=LF, lr=LR, h=H_COG)


def _sel(W, idx):
    return np.arange(len(W["r"])) if idx is None else np.asarray(idx)


def _at(x, f):
    """x (N, T) sampled at fractional column index f (scalar), clamped, linear."""
    T = x.shape[1]
    f = min(max(f, 0.0), T - 1.0)
    i = int(f)
    a = f - i
    if a < 1e-9 or i >= T - 1:
        return x[:, i]
    return x[:, i] * (1 - a) + x[:, i + 1] * a


def speed_traj(W, p, idx=None, hor=HOR, warm=WARM):
    """(N, warm+hor+1) speed: measured through warm-up and t0, the first-order loop after. Also its derivative."""
    sel = _sel(W, idx)
    cv, vm = W["speed_cmd"][sel], W["v"][sel]
    q = {**SPEED_DEFAULT, **{k: p[k] for k in SPEED_DEFAULT if k in p}}
    dt = 1.0 / HZ
    V = np.empty((len(sel), warm + hor + 1))
    A = np.empty_like(V)
    V[:, :warm + 1] = vm[:, :warm + 1]
    # warm-up acceleration from the IMU (0.1 s mean): the wheel-speed derivative is ERPM-quantised noise at 100 Hz
    A[:, :warm + 1] = W["ax"][sel, :warm + 1] if "ax" in W else np.gradient(vm[:, :warm + 1], dt, axis=1)
    v = vm[:, warm].copy()
    # whole samples: linear interpolation at a fractional delay low-passes the 40 Hz sample-and-hold command by an
    # amount that depends on the fraction, which puts artificial dips in the objective between grid points (11-28);
    # the delay is therefore fitted on a grid (fit.fit_grid), not by finite differences
    tv = int(round(q["tau_v"] * HZ))
    nc = cv.shape[1]
    for i in range(warm, warm + hor):
        a = (cv[:, min(max(i - tv, 0), nc - 1)] - v) / q["tau_m"]
        a = np.clip(a, -q["a_b"], q["a_a"] * np.minimum(1.0, q["v_s"] / np.maximum(np.abs(v), 1e-3)))
        v = np.maximum(v + a * dt, 0.0)
        V[:, i + 1] = v
        A[:, i] = a
    A[:, warm + hor] = A[:, warm + hor - 1]
    return V, A


def lin_rollout(W, p, idx=None, steer_key="steer", hor=HOR, warm=WARM):
    """(a) r = v tan(k d(t - tau) + b) / (L + K v^2); heading and base_link position integrated."""
    sel = _sel(W, idx)
    q = {**LIN_DEFAULT, **p}
    cd = W[steer_key][sel]
    V, _ = speed_traj(W, q, sel, hor, warm)
    N = len(sel)
    td = int(round(q["tau"] * HZ))                  # whole samples, see speed_traj; fit tau with fit.fit_grid
    k, b, kus = q["k"], q["b"], q["kus"]
    nc = cd.shape[1]
    x, y, yaw = (W["px"][sel, warm].copy(), W["py"][sel, warm].copy(), W["pyaw"][sel, warm].copy())
    R = np.empty((N, hor + 1)); X = np.empty_like(R); Y = np.empty_like(R); YW = np.empty_like(R)

    def rate(i, v):
        return v * np.tan(k * cd[:, min(max(i - td, 0), nc - 1)] + b) / (L + kus * v * v)

    dt = 1.0 / HZ
    for j in range(hor + 1):
        i = warm + j
        v = V[:, i]
        r = rate(i, v)
        R[:, j] = r; X[:, j] = x; Y[:, j] = y; YW[:, j] = yaw
        if j < hor:
            vn = V[:, i + 1]
            rn = rate(i + 1, vn)
            yaw = yaw + 0.5 * (r + rn) * dt
            x = x + vn * np.cos(yaw) * dt
            y = y + vn * np.sin(yaw) * dt
    R[:, 0] = W["r"][sel, warm]
    Vh = V[:, warm:]
    return dict(r=R, v=Vh, ay=Vh * R, bx=X, by=Y, yaw=YW, beta=np.zeros_like(R))


def tyre(alpha, B, C):
    return np.sin(C * np.arctan(B * alpha))


def tyre_curve(p, axle, alpha):
    """normalised force F/Fz of axle 'f' or 'r' at slip angle alpha [rad]."""
    s, mu, C = (p["sf"], p["mf"], p["Cf"]) if axle == "f" else (p["sr"], p["mr"], p["Cr"])
    return mu * tyre(alpha, s / (mu * C), C)


def tyre_peak(p, axle):
    """(peak F/Fz, slip angle of the peak [rad]) of sin(C atan(B a)) * mu: the peak is mu at B a = tan(pi / 2C)
    when C > 1; for C <= 1 the curve never turns over and saturates at mu sin(C pi / 2)."""
    s, mu, C = (p["sf"], p["mf"], p["Cf"]) if axle == "f" else (p["sr"], p["mr"], p["Cr"])
    B = s / (mu * C)
    if C > 1.0:
        return mu, float(np.tan(np.pi / (2 * C)) / B)
    return mu * float(np.sin(C * np.pi / 2)), float("inf")


def st_rollout(W, p, idx=None, steer_key="steer", hor=HOR, warm=WARM, sub=5, lt=False, v_blend=(0.5, 1.5),
               sv_max=None):
    """(b) / (d) dynamic single track, per unit mass. Warm-up: speed and yaw rate held at the measurement every
    10 ms, lateral velocity and servo angle the model's own. Returns base_link pose like the PF's."""
    sel = _sel(W, idx)
    q = {**ST_DEFAULT, **p}
    cd = W[steer_key][sel]
    rm = W["r"][sel]
    V, A = speed_traj(W, q, sel, hor, warm)
    N = len(sel)
    k, b = q["k"], q["b"]
    td = q["tau_d"] * HZ
    lf, lr, iz, h = q["lf"], q["lr"], q["iz_m"], q["h"]
    LL = lf + lr
    mf, mr = q["mf"], q["mr"]
    Bf, Br = q["sf"] / (mf * q["Cf"]), q["sr"] / (mr * q["Cr"])
    Cf, Cr = q["Cf"], q["Cr"]
    dt = 1.0 / HZ / sub
    a_s = 1.0 - np.exp(-dt / max(q["tau_s"], 1e-4))
    fzf0, fzr0 = G * lr / LL, G * lf / LL
    v0, v1 = v_blend
    kk = lr / LL
    # initial state at the start of the warm-up
    delta = k * cd[:, 0] + b
    vx = V[:, 0]
    r = rm[:, 0].copy()
    vy = vx * kk * np.tan(delta)
    x = W["px"][sel, warm] + lr * np.cos(W["pyaw"][sel, warm])
    y = W["py"][sel, warm] + lr * np.sin(W["pyaw"][sel, warm])
    yaw = W["pyaw"][sel, warm].copy()
    out = {kname: np.empty((N, hor + 1)) for kname in ("r", "v", "ay", "x", "y", "yaw", "beta", "delta", "vy")}
    ay_acc = np.zeros(N)

    def rec(j, ay):
        out["r"][:, j] = r; out["v"][:, j] = vx; out["ay"][:, j] = ay
        out["x"][:, j] = x; out["y"][:, j] = y; out["yaw"][:, j] = yaw
        out["beta"][:, j] = np.arctan2(vy, np.maximum(vx, 0.1))
        out["delta"][:, j] = delta; out["vy"][:, j] = vy

    for j in range(warm + hor):
        if j <= warm:
            r = rm[:, j].copy()
        if j == warm:
            rec(0, np.full(N, np.nan))
        ay_acc[:] = 0.0
        for s in range(sub):
            f = j + s / sub
            vx = V[:, j] + (V[:, j + 1] - V[:, j]) * (s / sub)
            tgt = k * _at(cd, f - td) + b
            if sv_max is None:
                delta = delta + (tgt - delta) * a_s
            else:
                delta = delta + np.clip((tgt - delta) * a_s, -sv_max * dt, sv_max * dt)
            vxs = np.maximum(vx, 0.3)
            af = delta - np.arctan2(vy + lf * r, vxs)
            ar = -np.arctan2(vy - lr * r, vxs)
            if lt:
                ax_ = A[:, j]
                fzf = np.maximum(fzf0 - h * ax_ / LL, 0.0)
                fzr = np.maximum(fzr0 + h * ax_ / LL, 0.0)
            else:
                fzf, fzr = fzf0, fzr0
            cdl = np.cos(delta)
            fyf = mf * fzf * np.sin(Cf * np.arctan(Bf * af))
            fyr = mr * fzr * np.sin(Cr * np.arctan(Br * ar))
            vyd_dyn = fyf * cdl + fyr - vx * r
            rd_dyn = (lf * fyf * cdl - lr * fyr) / iz
            w = np.clip((vx - v0) / (v1 - v0), 0.0, 1.0)
            if np.all(w >= 1.0):
                vyd, rd = vyd_dyn, rd_dyn
            else:
                td_ = np.tan(delta)
                beta_k = np.arctan(kk * td_)
                r_kin = vx * np.cos(beta_k) * td_ / LL
                vy_kin = vx * np.tan(beta_k)
                vyd = w * vyd_dyn + (1 - w) * 20.0 * (vy_kin - vy)
                rd = w * rd_dyn + (1 - w) * 20.0 * (r_kin - r)
            vy = vy + vyd * dt
            r = r + rd * dt
            ay_acc += vyd + vx * r
            if j >= warm:
                c, sn = np.cos(yaw), np.sin(yaw)
                x = x + (vx * c - vy * sn) * dt
                y = y + (vx * sn + vy * c) * dt
                yaw = yaw + r * dt
        if j >= warm:
            vx = V[:, j + 1]
            rec(j - warm + 1, ay_acc / sub)
    out["ay"][:, 0] = out["ay"][:, 1]
    out["r"][:, 0] = rm[:, warm]
    out["bx"] = out["x"] - lr * np.cos(out["yaw"])
    out["by"] = out["y"] - lr * np.sin(out["yaw"])
    return out


def persistence(W, idx=None, hor=HOR, warm=WARM):
    sel = _sel(W, idx)
    r0, v0, ay0 = W["r"][sel, warm], W["v"][sel, warm], W["ay"][sel, warm]
    n = hor + 1
    tt = np.arange(n) / HZ
    yaw = W["pyaw"][sel, warm][:, None] + r0[:, None] * tt[None]
    dx = np.cumsum(v0[:, None] * np.cos(yaw) / HZ, 1) - v0[:, None] * np.cos(yaw[:, :1]) / HZ
    dy = np.cumsum(v0[:, None] * np.sin(yaw) / HZ, 1) - v0[:, None] * np.sin(yaw[:, :1]) / HZ
    return dict(r=np.repeat(r0[:, None], n, 1), v=np.repeat(v0[:, None], n, 1), ay=np.repeat(ay0[:, None], n, 1),
                bx=W["px"][sel, warm][:, None] + dx, by=W["py"][sel, warm][:, None] + dy, yaw=yaw,
                beta=np.zeros((len(sel), n)))


# ---- the simulator's own chain (torch) -------------------------------------------------------------------------------

def sim_nominal(cfg=None):
    import torch
    from ..params import Config
    from ..randomization import ParamSet
    cfg = cfg or Config()
    ps = ParamSet(cfg, 1, torch.device("cpu"), torch.Generator())
    return dict(ps.nominal), bool(cfg.vehicle.wheel_model), float(cfg.sim.physics_dt)


def sim_rollout(W, over=None, idx=None, steer_key="steer", hor=HOR, warm=WARM, cfg=None, wheel=None, threads=None):
    """The simulator's actuator + dynamics chain (`sim._roll_physics` without contacts / IMU), driven by the recorded
    commands at its 1 ms substep. Ported from the 11-28 audit (`D_tools/models.py`), with `cfg` so a profile's
    Config can be replayed. Warm-up as for the other models; returns base_link pose."""
    import torch
    from .. import dynamics as dyn
    from ..actuators import servo_target, vesc_accel
    if threads:
        torch.set_num_threads(threads)
    P0, wheel0, dt = sim_nominal(cfg)
    wheel = wheel0 if wheel is None else wheel
    sel = _sel(W, idx)
    N = len(sel)
    with torch.no_grad():
        P = {k: torch.full((N,), float(v), dtype=torch.float64) for k, v in P0.items()}
        for k, v in (over or {}).items():
            P[k] = (torch.as_tensor(v, dtype=torch.float64).expand(N).clone() if np.ndim(v)
                    else torch.full((N,), float(v), dtype=torch.float64))
        t = lambda a: torch.as_tensor(a[sel], dtype=torch.float64)
        cd, cv, rm, vm, axm = t(W[steer_key]), t(W["speed_cmd"]), t(W["r"]), t(W["v"]), t(W["ax"])
        sub = int(round(1.0 / HZ / dt))
        dsteps = torch.round(P["cmd_delay"] / dt).long()
        msteps = torch.round(P.get("motor_delay", torch.zeros(N, dtype=torch.float64)) / dt).long()
        ar = torch.arange(N)
        ncmd = cd.shape[1]

        def cmd(kk):
            k1 = ((kk - dsteps).clamp(min=0) // sub).clamp(max=ncmd - 1)
            km = ((kk - dsteps - msteps).clamp(min=0) // sub).clamp(max=ncmd - 1)
            return cd[ar, k1], cv[ar, km]

        st = torch.zeros(N, dyn.STATE_DIM, dtype=torch.float64)
        st[:, dyn.IVX] = vm[:, 0]; st[:, dyn.IR] = rm[:, 0]
        st[:, dyn.ISTEER] = servo_target(cd[:, 0], P)
        st[:, dyn.IOMEGA] = vm[:, 0] / P["r_w"]
        ax = axm[:, 0].clone()
        lr = P["lr"]
        out = {k: [] for k in ("r", "v", "ay", "x", "y", "yaw", "vy", "vx")}

        def rec(ay):
            out["r"].append(st[:, dyn.IR].clone()); out["v"].append(st[:, dyn.IOMEGA] * P["r_w"])
            out["ay"].append(ay.clone()); out["x"].append(st[:, dyn.IX].clone()); out["y"].append(st[:, dyn.IY].clone())
            out["yaw"].append(st[:, dyn.IYAW].clone()); out["vy"].append(st[:, dyn.IVY].clone())
            out["vx"].append(st[:, dyn.IVX].clone())

        for j in range(warm + hor):
            if j <= warm:
                st[:, dyn.IVX] = vm[:, j]; st[:, dyn.IR] = rm[:, j]; st[:, dyn.IOMEGA] = vm[:, j] / P["r_w"]
            if j == warm:
                yaw0 = t(W["pyaw"])[:, warm]
                st[:, dyn.IX] = t(W["px"])[:, warm] + lr * torch.cos(yaw0)
                st[:, dyn.IY] = t(W["py"])[:, warm] + lr * torch.sin(yaw0)
                st[:, dyn.IYAW] = yaw0
                rec(torch.full((N,), float("nan"), dtype=torch.float64))
            ay_acc = torch.zeros(N, dtype=torch.float64)
            for s in range(sub):
                kk = j * sub + s
                c_d, c_v = cmd(torch.full((N,), kk))
                tgt = servo_target(c_d, P)
                v_fb = st[:, dyn.IOMEGA] * P["r_w"] if wheel else st[:, dyn.IVX]
                a_cmd = vesc_accel(c_v, v_fb, P)
                st, ax, ay = dyn.step_dynamics(st, tgt, a_cmd, ax, P, P["servo_tau"], dt, wheel=wheel)
                ay_acc += ay
            if j >= warm:
                rec(ay_acc / sub)
        o = {k: torch.stack(v, 1).numpy() for k, v in out.items()}
    o["ay"][:, 0] = o["ay"][:, 1]
    lrn = float(P0["lr"]) if "lr" not in (over or {}) else float(np.mean(over["lr"]))
    # the yaw is wrapped by the simulator; unwrap along time so the heading error is not a 2 pi jump
    o["yaw"] = np.unwrap(o["yaw"], axis=1)
    o["bx"] = o["x"] - lrn * np.cos(o["yaw"]); o["by"] = o["y"] - lrn * np.sin(o["yaw"])
    o["beta"] = np.arctan2(o["vy"], np.maximum(o["vx"], 0.1))
    return o


# ---- errors -------------------------------------------------------------------------------------------------------------

def errors(W, pr, idx=None, horizons=(50, 100, 200), warm=WARM):
    """Per window, per horizon h: yaw-rate RMS over (0, h], |heading error| at h, |speed error| at h, lateral
    acceleration RMS over (0, h], base_link position error at h (vs the localisation pose; NaN without one)."""
    sel = _sel(W, idx)
    rm, vm, aym = W["r"][sel, warm:], W["v"][sel, warm:], W["ay"][sel, warm:]
    out = {}
    for h in horizons:
        er = pr["r"][:, 1:h + 1] - rm[:, 1:h + 1]
        e = dict(yaw=np.sqrt(np.mean(er ** 2, 1)), head=np.abs(er.sum(1) / HZ),
                 v=np.abs(pr["v"][:, h] - vm[:, h]),
                 ay=np.sqrt(np.mean((pr["ay"][:, 1:h + 1] - aym[:, 1:h + 1]) ** 2, 1)))
        pe = np.hypot(pr["bx"][:, h] - W["px"][sel, warm + h], pr["by"][:, h] - W["py"][sel, warm + h])
        pe[~W["has_pf"][sel]] = np.nan
        e["pos"] = pe
        out[h] = e
    return out


def spin_flags(pr, beta_deg=20.0):
    """windows where the model's body sideslip exceeds `beta_deg` anywhere in the horizon."""
    return np.nanmax(np.abs(pr["beta"]), 1) > np.deg2rad(beta_deg)


def speed_rollout(W, p, idx=None, hor=HOR, warm=WARM, steer_key=None):
    """the shared longitudinal model alone (for fitting it): dict(v=(N, hor+1))."""
    V, A = speed_traj(W, p, idx, hor, warm)
    return dict(v=V[:, warm:], a=A[:, warm:])


def st_lt_rollout(W, p, idx=None, steer_key="steer", hor=HOR, warm=WARM):
    """(d): (b) with longitudinal load transfer."""
    return st_rollout(W, p, idx=idx, steer_key=steer_key, hor=hor, warm=warm, lt=True)
