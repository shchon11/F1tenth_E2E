"""Short-trajectory + MPC action space for the e2e policy.

The policy does not output steer/speed. It outputs a short local *plan* in the car's own frame:
    4 curvature values at 0, 1/3, 2/3 and 1 of the plan length L_p (L_p = 0.7 s of travel,
    1.5-6 m), interpolated linearly along arc length and integrated into a path that leaves the
    car straight ahead (curvature, not y(x): a hairpin is just a large curvature, whereas a
    polynomial in x cannot bend back), and
    2 speeds: the target speed 0.15 s from now and at the end of the plan (linear in between).
A receding-horizon tracker (iLQR on a kinematic bicycle with understeer, 12 x 50 ms) turns the
plan into (steer, speed) every control step. Nothing here needs a map or a pose estimate: the
plan is re-issued every step in the body frame, so the real car runs exactly the same code.

Why: the plan is a smooth, actuator-independent action space (easier exploration, the tracker
absorbs servo calibration / latency), the tracker's model makes the sim-to-real step explicit,
and the plan itself is a safety hook (it can be checked against the scan before execution)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

N_KNOTS = 4
ACT_DIM = N_KNOTS + 2
XI = torch.linspace(0.0, 1.0, N_KNOTS)


@dataclass
class PlanSpec:
    kappa_max: float = 1.6         # [1/m] curvature range of the knots (the car's full-lock radius is ~0.74 m)
    horizon_s: float = 0.7         # plan length = horizon_s * v, clamped to [len_min, len_max]
    len_min: float = 1.5
    len_max: float = 6.0
    v_cmd_lead: float = 0.15       # [s] the first speed target refers to this far ahead
    N: int = 12                    # MPC steps
    dt: float = 0.05               # [s] MPC step
    delay: float = 0.06            # [s] nominal command latency the tracker predicts over
    k_us: float = 0.003            # [s^2/m] understeer: effective wheelbase L + k_us v^2
    a_max: float = 6.0             # [m/s^2] tracker acceleration bound
    q: Tuple[float, float, float, float] = (1.0, 6.0, 1.0, 0.4)    # x, y, heading, speed tracking weights
    qf: Tuple[float, float, float, float] = (2.0, 10.0, 2.0, 0.4)
    r: Tuple[float, float] = (0.3, 0.02)                            # steer, accel effort
    rd: Tuple[float, float] = (6.0, 0.05)                           # steer, accel rate
    iters: int = 2


def plan_length(v: torch.Tensor, spec: PlanSpec) -> torch.Tensor:
    return (spec.horizon_s * v.abs()).clamp(spec.len_min, spec.len_max)


def decode(action: torch.Tensor, v_meas: torch.Tensor, v_max: float, speed_cap: torch.Tensor, spec: PlanSpec):
    """normalized action (B,6) -> (kappa knots (B,4) [1/m], L_p (B,), v_start (B,), v_end (B,))"""
    a = action.clamp(-1.0, 1.0)
    Lp = plan_length(v_meas, spec)
    k = a[:, :N_KNOTS] * spec.kappa_max
    v0 = torch.minimum((a[:, N_KNOTS] + 1.0) * 0.5 * v_max, speed_cap)
    v1 = torch.minimum((a[:, N_KNOTS + 1] + 1.0) * 0.5 * v_max, speed_cap)
    return k, Lp, v0, v1


def encode(kappas: torch.Tensor, v_start: torch.Tensor, v_end: torch.Tensor, v_max: float, spec: PlanSpec) -> torch.Tensor:
    """(B,4) curvature knots [1/m], speeds [m/s] -> normalized action (B,6)"""
    return torch.cat([(kappas / spec.kappa_max).clamp(-1, 1), (v_start / v_max * 2 - 1).clamp(-1, 1)[:, None],
                      (v_end / v_max * 2 - 1).clamp(-1, 1)[:, None]], 1)


def path_points(k: torch.Tensor, Lp: torch.Tensor, n: int = 25):
    """Dense samples of the plan in the body frame: x, y, heading (B,n) and arc length (B,n).
    Curvature is linear between the knots; heading is its integral, the path the integral of that."""
    xi = torch.linspace(0.0, 1.0, n, device=k.device, dtype=k.dtype)[None]          # (1,n)
    pos = xi * (N_KNOTS - 1)
    i0 = pos.floor().clamp(max=N_KNOTS - 2).long(); w = pos - i0.to(k.dtype)
    i0 = i0.expand(k.shape[0], n)
    kap = k.gather(1, i0) * (1 - w) + k.gather(1, i0 + 1) * w                        # (B,n)
    ds = (Lp / (n - 1))[:, None]
    psi = torch.cumsum(0.5 * (kap[:, 1:] + kap[:, :-1]) * ds, 1)
    psi = torch.cat([torch.zeros_like(psi[:, :1]), psi], 1)
    cx, sy = torch.cos(psi), torch.sin(psi)
    x = torch.cat([torch.zeros_like(psi[:, :1]), torch.cumsum(0.5 * (cx[:, 1:] + cx[:, :-1]) * ds, 1)], 1)
    y = torch.cat([torch.zeros_like(psi[:, :1]), torch.cumsum(0.5 * (sy[:, 1:] + sy[:, :-1]) * ds, 1)], 1)
    s = xi * Lp[:, None]
    return x, y, psi, s


def reference(k, Lp, v0, v1, spec: PlanSpec, v_now: Optional[torch.Tensor] = None):
    """Time-indexed reference (B, N+1, 4) = x, y, heading, speed along the plan. The speed *target*
    is the plan's profile (v0 at the start, linear in arc length to v1 at the end); the position
    reference is walked with the speed the car can actually have: from v_now (measured) towards
    the profile within a_max, so a car that is slower than its plan is not chased by a reference
    that has run ahead along the curve (time-free tracking, like pure pursuit)."""
    x, y, psi, s = path_points(k, Lp)
    S = s[:, -1].clamp_min(1e-3)
    dev = k.device; T = spec.N + 1
    st = torch.zeros(k.shape[0], T, device=dev, dtype=k.dtype); v = torch.zeros_like(st)
    vw = (v0 if v_now is None else v_now.abs()).clamp_min(0.3)
    for t_ in range(T):
        frac = (st[:, t_] / S).clamp(0.0, 1.0)
        v[:, t_] = v0 + (v1 - v0) * frac                                   # target profile at that point of the path
        if t_ < T - 1:
            dv = (v[:, t_] - vw).clamp(-spec.a_max * spec.dt, spec.a_max * spec.dt)
            vw = (vw + dv).clamp_min(0.3)
            st[:, t_ + 1] = st[:, t_] + vw * spec.dt
    idx = torch.searchsorted(s.contiguous(), st.contiguous()).clamp(1, s.shape[1] - 1)   # (B,T)
    s_lo, s_hi = s.gather(1, idx - 1), s.gather(1, idx)
    w = ((st - s_lo) / (s_hi - s_lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    g_ = lambda arr: arr.gather(1, idx - 1) * (1 - w) + arr.gather(1, idx) * w
    xr, yr, pr = g_(x), g_(y), g_(psi)
    over = (st - S[:, None]).clamp_min(0.0)                               # past the end: straight on
    pe = psi[:, -1:]
    xr = xr + over * torch.cos(pe); yr = yr + over * torch.sin(pe)
    pr = torch.where(st > S[:, None], pe.expand_as(pr), pr)
    return torch.stack([xr, yr, pr, v], 2)


def _dyn(z, u, spec: PlanSpec, wb: float):
    x, y, psi, v = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    d, a = u[:, 0], u[:, 1]
    Le = wb + spec.k_us * v * v
    nz = torch.stack([x + v * torch.cos(psi) * spec.dt, y + v * torch.sin(psi) * spec.dt,
                      psi + v * torch.tan(d) / Le * spec.dt, (v + a * spec.dt).clamp_min(0.0), d, a], 1)
    return nz


def _jac(z, u, spec: PlanSpec, wb: float):
    B = z.shape[0]; dev = z.device; dt = spec.dt
    psi, v, d = z[:, 2], z[:, 3], u[:, 0]
    Le = wb + spec.k_us * v * v
    A = torch.zeros(B, 6, 6, device=dev); Bm = torch.zeros(B, 6, 2, device=dev)
    for i in range(4): A[:, i, i] = 1.0
    A[:, 0, 2] = -v * torch.sin(psi) * dt; A[:, 0, 3] = torch.cos(psi) * dt
    A[:, 1, 2] = v * torch.cos(psi) * dt; A[:, 1, 3] = torch.sin(psi) * dt
    A[:, 2, 3] = torch.tan(d) * (wb - spec.k_us * v * v) / Le ** 2 * dt
    Bm[:, 2, 0] = v / Le * (1 + torch.tan(d) ** 2) * dt; Bm[:, 3, 1] = dt
    Bm[:, 4, 0] = 1.0; Bm[:, 5, 1] = 1.0
    return A, Bm


def ilqr(z0: torch.Tensor, ref: torch.Tensor, u_warm: torch.Tensor, spec: PlanSpec, wb: float, s_max: float, v_max: float):
    """z0 (B,6) [x,y,psi,v,steer_prev,accel_prev], ref (B,N+1,4), u_warm (B,N,2) -> u (B,N,2), z (B,N+1,6)."""
    B, N, dev = z0.shape[0], spec.N, z0.device
    Q = torch.zeros(6, 6, device=dev); Q[:4, :4] = torch.diag(torch.tensor(spec.q, device=dev))
    Qf = torch.zeros(6, 6, device=dev); Qf[:4, :4] = torch.diag(torch.tensor(spec.qf, device=dev))
    R = torch.diag(torch.tensor(spec.r, device=dev)); Rd = torch.diag(torch.tensor(spec.rd, device=dev))
    Q[4:, 4:] = Rd                                                     # (u - u_prev)^T Rd (u - u_prev): u_prev lives in the state
    lo = torch.tensor([-s_max, -spec.a_max], device=dev); hi = torch.tensor([s_max, spec.a_max], device=dev)
    E = torch.zeros(6, 2, device=dev); E[4, 0] = 1.0; E[5, 1] = 1.0      # picks u_prev out of z

    def rollout(u):
        z = [z0]
        for t in range(N):
            z.append(_dyn(z[-1], u[:, t], spec, wb))
        return torch.stack(z, 1)

    u = u_warm.clone(); z = rollout(u)
    for _ in range(spec.iters):
        # cost derivatives around (z, u): quadratic cost, exact
        e = z.clone(); e[:, :, :4] -= ref                                   # tracking error (u_prev slots kept)
        Vz = 2 * (e[:, N] @ Qf); Vzz = (2 * Qf)[None].expand(B, 6, 6).clone()
        ks, Ks = [None] * N, [None] * N
        for t in reversed(range(N)):
            A, Bm = _jac(z[:, t], u[:, t], spec, wb)
            du = u[:, t] - z[:, t, 4:6]
            lz = 2 * (e[:, t] @ Q); lz[:, 4:6] = -2 * (du @ Rd)
            lu = 2 * (u[:, t] @ R) + 2 * (du @ Rd)
            lzz = 2 * Q; luu = 2 * (R + Rd); luz = (-2 * Rd) @ E.T            # (2,6)
            Qz = lz + (A.transpose(1, 2) @ Vz[..., None]).squeeze(-1)
            Qu = lu + (Bm.transpose(1, 2) @ Vz[..., None]).squeeze(-1)
            Qzz = lzz + A.transpose(1, 2) @ Vzz @ A
            Quu = luu + Bm.transpose(1, 2) @ Vzz @ Bm + 1e-3 * torch.eye(2, device=dev)
            Quz = luz + Bm.transpose(1, 2) @ Vzz @ A
            det = Quu[:, 0, 0] * Quu[:, 1, 1] - Quu[:, 0, 1] * Quu[:, 1, 0]
            inv = torch.stack([torch.stack([Quu[:, 1, 1], -Quu[:, 0, 1]], 1), torch.stack([-Quu[:, 1, 0], Quu[:, 0, 0]], 1)], 1) / det[:, None, None]
            k = -(inv @ Qu[..., None]).squeeze(-1); K = -(inv @ Quz)
            ks[t], Ks[t] = k, K
            Vz = Qz + (K.transpose(1, 2) @ (Quu @ k[..., None])).squeeze(-1) + (K.transpose(1, 2) @ Qu[..., None]).squeeze(-1) + (Quz.transpose(1, 2) @ k[..., None]).squeeze(-1)
            Vzz = Qzz + K.transpose(1, 2) @ Quu @ K + K.transpose(1, 2) @ Quz + Quz.transpose(1, 2) @ K
            Vzz = 0.5 * (Vzz + Vzz.transpose(1, 2))
        # forward pass with clamped inputs
        zn = z0; un = torch.zeros_like(u); zs = [z0]
        for t in range(N):
            ut = u[:, t] + ks[t] + (Ks[t] @ (zn - z[:, t])[..., None]).squeeze(-1)
            ut = torch.maximum(torch.minimum(ut, hi), lo)
            ut[:, 1] = torch.minimum(ut[:, 1], (v_max - zn[:, 3]) / spec.dt)   # never command past v_max
            un[:, t] = ut; zn = _dyn(zn, ut, spec, wb); zs.append(zn)
        u, z = un, torch.stack(zs, 1)
    return u, z


def solve(action, v_meas, speed_cap, yaw_rate, delay, u_prev, warm, spec: PlanSpec, wb: float, s_max: float, v_max: float):
    """Whole tracker step as one function (compiled into a single CUDA graph on the GPU): decode the plan,
    build the reference, predict over the latency, run the iLQR. Returns (u (B,N,2), z (B,N+1,6), ref)."""
    k, Lp, v0, v1 = decode(action, v_meas, v_max, speed_cap, spec)
    ref = reference(k, Lp, v0, v1, spec, v_meas)
    v = v_meas.abs()
    Le = wb + spec.k_us * v * v
    steer_now = torch.atan(yaw_rate * Le / v.clamp_min(0.5)).clamp(-s_max, s_max)
    psi0 = yaw_rate * delay
    z0 = torch.stack([v * delay, 0.5 * v * psi0 * delay, psi0, (v + u_prev[:, 1] * delay).clamp_min(0.0),
                      torch.where(v > 0.5, steer_now, u_prev[:, 0]), u_prev[:, 1]], 1)
    u, z = ilqr(z0, ref, warm, spec, wb, s_max, v_max)
    return u, z, ref


_solve_compiled = {}


def solve_fast(*args, **kw):
    """solve() through torch.compile(mode="reduce-overhead") on CUDA (one graph per (batch, spec))."""
    if not args[0].is_cuda:
        return solve(*args, **kw)
    key = (args[0].shape[0], id(args[7]))
    f = _solve_compiled.get(key)
    if f is None:
        try:
            f = torch.compile(solve, dynamic=False, mode="reduce-overhead")
        except Exception:
            f = solve
        _solve_compiled[key] = f
    try:
        return f(*args, **kw)
    except Exception:
        _solve_compiled[key] = solve
        return solve(*args, **kw)


class PlanTracker:
    """Per-env state for the plan -> (steer, speed) tracker. Call reset(ids) when envs restart."""

    def __init__(self, num_envs: int, device, wheelbase: float, s_max: float, v_max: float, spec: Optional[PlanSpec] = None):
        self.B, self.device, self.wb, self.s_max, self.v_max = num_envs, torch.device(device), wheelbase, s_max, v_max
        self.spec = spec or PlanSpec()
        self.u_prev = torch.zeros(num_envs, 2, device=self.device)                 # last applied steer, accel
        self.u_seq = torch.zeros(num_envs, self.spec.N, 2, device=self.device)     # warm start
        self.last_ref = None                                                       # (B,N+1,4) body frame plan, for viewers
        self.last_pred = None                                                      # (B,N+1,4) the tracker's predicted motion

    def reset(self, ids: torch.Tensor):
        self.u_prev[ids] = 0.0; self.u_seq[ids] = 0.0

    @torch.no_grad()
    def __call__(self, action: torch.Tensor, v_meas: torch.Tensor, speed_cap: torch.Tensor,
                 yaw_rate: Optional[torch.Tensor] = None, delay=None) -> torch.Tensor:
        """normalized plan (B,6), measured speed (B,), speed cap (B,), measured yaw rate (B,) (IMU gyro z,
        optional), delay: calibrated command latency [s] (float or (B,), default spec.delay)
        -> (steer [rad], speed cmd [m/s]) (B,2)"""
        sp = self.spec
        v = v_meas.abs()
        if yaw_rate is None:                                       # no IMU: assume the last command took
            yaw_rate = v * torch.tan(self.u_prev[:, 0]) / (self.wb + sp.k_us * v * v)
        if delay is None:
            delay = torch.full_like(v, sp.delay)
        elif not torch.is_tensor(delay):
            delay = torch.full_like(v, float(delay))
        warm = torch.cat([self.u_seq[:, 1:], self.u_seq[:, -1:]], 1)
        u, z, ref = solve_fast(action, v_meas, speed_cap, yaw_rate, delay, self.u_prev, warm, sp, self.wb, self.s_max, self.v_max)
        u, z, ref = u.clone(), z.clone(), ref.clone()             # CUDA-graph outputs are reused by the next run
        self.u_seq = u; self.u_prev = u[:, 0].clone(); self.last_ref = ref; self.last_pred = z[:, :, :4]
        k_ = max(1, int(round(sp.v_cmd_lead / sp.dt)))
        v_cmd = torch.minimum(z[:, k_, 3], speed_cap)
        return torch.stack([u[:, 0, 0], v_cmd], 1)
