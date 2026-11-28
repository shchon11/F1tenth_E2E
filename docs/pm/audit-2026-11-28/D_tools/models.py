"""Models for the open-loop audit replay: the simulator's own physics chain, a delayed kinematic bicycle, persistence."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/shchon11/F1tenth_E2E/f1sim")
from f1sim import dynamics as dyn                      # noqa: E402
from f1sim.actuators import servo_target, vesc_accel   # noqa: E402
from f1sim.params import Config                        # noqa: E402
from f1sim.randomization import ParamSet               # noqa: E402

from windows import WARM, HOR, LR                      # noqa: E402

HZ = 100
L = 0.3302


def nominal_params():
    cfg = Config()
    ps = ParamSet(cfg, 1, torch.device("cpu"), torch.Generator())
    return dict(ps.nominal), bool(cfg.vehicle.wheel_model), float(cfg.sim.physics_dt)


@torch.no_grad()
def sim_rollout(W, over=None, idx=None, wheel=None, steer_key="dcmd", hor=HOR):
    """The simulator's actuator + dynamics chain (sim._roll_physics without contacts/IMU), driven by the
    recorded commands at 1 ms substeps. Warm-up: the 0.5 s before t0 with vx / wheel / yaw rate nudged to the
    measurement every 10 ms (so vy, servo angle and load transfer are the model's own); then free run 2 s.
    Returns dict of (N, HOR+1) arrays: r, v (wheel surface speed), ay (specific force), bx, by (base_link), yaw."""
    P0, wheel0, dt = nominal_params()
    wheel = wheel0 if wheel is None else wheel
    sel = np.arange(len(W["r"])) if idx is None else idx
    N = len(sel)
    P = {k: torch.full((N,), float(v), dtype=torch.float64) for k, v in P0.items()}
    for k, v in (over or {}).items():
        P[k] = torch.as_tensor(v, dtype=torch.float64).expand(N).clone() if np.ndim(v) else torch.full((N,), float(v), dtype=torch.float64)
    t = lambda a: torch.as_tensor(a[sel], dtype=torch.float64)
    cd, cv, rm, vm, axm = t(W[steer_key]), t(W["vcmd"]), t(W["r"]), t(W["v"]), t(W["ax"])
    sub = int(round(1.0 / HZ / dt))
    dsteps = torch.round(P["cmd_delay"] / dt).long()
    msteps = torch.round(P.get("motor_delay", torch.zeros(N, dtype=torch.float64)) / dt).long()
    ar = torch.arange(N)
    ncmd = cd.shape[1]

    def cmd(k):
        kk = ((k - dsteps).clamp(min=0) // sub).clamp(max=ncmd - 1)
        km = ((k - dsteps - msteps).clamp(min=0) // sub).clamp(max=ncmd - 1)
        return cd[ar, kk], cv[ar, km]

    st = torch.zeros(N, dyn.STATE_DIM, dtype=torch.float64)
    st[:, dyn.IVX] = vm[:, 0]; st[:, dyn.IR] = rm[:, 0]
    st[:, dyn.ISTEER] = servo_target(cd[:, 0], P)
    st[:, dyn.IOMEGA] = vm[:, 0] / P["r_w"]
    ax = axm[:, 0].clone()
    out = {k: [] for k in ("r", "v", "ay", "x", "y", "yaw", "vy")}
    for j in range(WARM + hor):
        if j <= WARM:                                  # nudge measured states (warm-up and t0)
            st[:, dyn.IVX] = vm[:, j]; st[:, dyn.IR] = rm[:, j]; st[:, dyn.IOMEGA] = vm[:, j] / P["r_w"]
        if j == WARM:
            if "px" in W:
                yaw0 = t(W["pyaw"])[:, WARM]
                st[:, dyn.IX] = t(W["px"])[:, WARM] + LR * torch.cos(yaw0)
                st[:, dyn.IY] = t(W["py"])[:, WARM] + LR * torch.sin(yaw0)
                st[:, dyn.IYAW] = yaw0
            out_rec(out, st, P, torch.zeros(N, dtype=torch.float64) + float("nan"))
        ay_acc = torch.zeros(N, dtype=torch.float64)
        for s in range(sub):
            k = j * sub + s
            c_d, c_v = cmd(torch.full((N,), k))
            tgt = servo_target(c_d, P)
            v_fb = st[:, dyn.IOMEGA] * P["r_w"] if wheel else st[:, dyn.IVX]
            a_cmd = vesc_accel(c_v, v_fb, P)
            st, ax, ay = dyn.step_dynamics(st, tgt, a_cmd, ax, P, P["servo_tau"], dt, wheel=wheel)
            ay_acc += ay
        if j >= WARM:
            out_rec(out, st, P, ay_acc / sub)
    o = {k: torch.stack(v, 1).numpy() for k, v in out.items()}
    o["ay"][:, 0] = o["ay"][:, 1]
    o["bx"] = o["x"] - LR * np.cos(o["yaw"]); o["by"] = o["y"] - LR * np.sin(o["yaw"])
    return o


def out_rec(out, st, P, ay):
    out["r"].append(st[:, dyn.IR].clone()); out["v"].append(st[:, dyn.IOMEGA] * P["r_w"])
    out["ay"].append(ay.clone()); out["x"].append(st[:, dyn.IX].clone()); out["y"].append(st[:, dyn.IY].clone())
    out["yaw"].append(st[:, dyn.IYAW].clone()); out["vy"].append(st[:, dyn.IVY].clone())


def kb_rollout(W, p, idx=None, steer_key="dcmd"):
    """Kinematic bicycle at the rear axle (= base_link), steering delayed by p['tau'] (one pure delay, no servo
    lag): r = v tan(k d(t - tau) + b) / (L + kus v^2). Speed: first-order loop on the command delayed tau_v,
    accel clipped to [-a_b, a_a * min(1, v_s / v)]. 100 Hz explicit."""
    sel = np.arange(len(W["r"])) if idx is None else idx
    cd, cv, rm, vm = W[steer_key][sel], W["vcmd"][sel], W["r"][sel], W["v"][sel]
    N = len(sel)
    k, b, kus = p.get("k", 1.0), p.get("b", 0.0), p.get("kus", 0.0)
    td = int(round(p["tau"] * HZ)); tv = int(round(p.get("tau_v", 0.03) * HZ))
    tm, aa, ab, vs = p.get("tau_m", 0.2), p.get("a_a", 5.0), p.get("a_b", 5.0), p.get("v_s", 7.0)
    dt = 1.0 / HZ
    v = vm[:, WARM].copy()
    if "px" in W:
        x, y, yaw = W["px"][sel, WARM].copy(), W["py"][sel, WARM].copy(), W["pyaw"][sel, WARM].copy()
    else:
        x = y = yaw = np.zeros(N)
    R, V, AY, X, Y, YW = [], [], [], [], [], []

    def rate(i, v):
        d = k * cd[:, min(max(i - td, 0), cd.shape[1] - 1)] + b
        return v * np.tan(d) / (L + kus * v * v)

    for i in range(WARM, WARM + HOR + 1):
        r = rate(i, v)
        R.append(r); V.append(v.copy()); AY.append(v * r); X.append(x.copy()); Y.append(y.copy()); YW.append(yaw.copy())
        a = (cv[:, min(max(i - tv, 0), cv.shape[1] - 1)] - v) / tm
        a = np.clip(a, -ab, aa * np.minimum(1.0, vs / np.maximum(np.abs(v), 1e-3)))
        vn = np.maximum(v + a * dt, 0.0)
        rn = rate(i + 1, vn)
        yaw = yaw + 0.5 * (r + rn) * dt
        x = x + vn * np.cos(yaw) * dt; y = y + vn * np.sin(yaw) * dt
        v = vn
    st = lambda a: np.stack(a, 1)
    R = st(R); R[:, 0] = rm[:, WARM]               # t0 sample is the measured state for every model
    return dict(r=R, v=st(V), ay=st(AY), bx=st(X), by=st(Y), yaw=st(YW))


def persistence(W, idx=None):
    sel = np.arange(len(W["r"])) if idx is None else idx
    r0, v0, ay0 = W["r"][sel, WARM], W["v"][sel, WARM], W["ay"][sel, WARM]
    n = HOR + 1
    tt = np.arange(n) / HZ
    yaw = W["pyaw"][sel, WARM][:, None] + r0[:, None] * tt[None]
    # constant turn rate and speed at the rear axle
    dx = np.cumsum(v0[:, None] * np.cos(yaw) / HZ, 1) - v0[:, None] * np.cos(yaw[:, :1]) / HZ
    dy = np.cumsum(v0[:, None] * np.sin(yaw) / HZ, 1) - v0[:, None] * np.sin(yaw[:, :1]) / HZ
    return dict(r=np.repeat(r0[:, None], n, 1), v=np.repeat(v0[:, None], n, 1), ay=np.repeat(ay0[:, None], n, 1),
                bx=W["px"][sel, WARM][:, None] + dx, by=W["py"][sel, WARM][:, None] + dy, yaw=yaw)


def errors(W, pr, idx=None, horizons=(50, 100, 200)):
    """per-window errors at each horizon: yaw-rate RMS over (0,h], heading |int err| at h, |v err| at h,
    a_y RMS over (0,h], position |err| at h (base_link vs particle filter)."""
    sel = np.arange(len(W["r"])) if idx is None else idx
    rm, vm, aym = W["r"][sel, WARM:], W["v"][sel, WARM:], W["ay"][sel, WARM:]
    out = {}
    for h in horizons:
        er = pr["r"][:, 1:h + 1] - rm[:, 1:h + 1]
        e = dict(yaw=np.sqrt(np.mean(er ** 2, 1)), head=np.abs(er.sum(1) / HZ),
                 v=np.abs(pr["v"][:, h] - vm[:, h]),
                 ay=np.sqrt(np.mean((pr["ay"][:, 1:h + 1] - aym[:, 1:h + 1]) ** 2, 1)))
        if "px" in W:
            pe = np.hypot(pr["bx"][:, h] - W["px"][sel, WARM + h], pr["by"][:, h] - W["py"][sel, WARM + h])
            pe[~W["has_pf"][sel]] = np.nan
            e["pos"] = pe
        out[h] = e
    return out
