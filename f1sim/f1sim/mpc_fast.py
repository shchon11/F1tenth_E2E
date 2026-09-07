"""Deployment-side plan tracker: the same math as mpc.solve() for one car, compiled with Numba.

The training tracker (mpc.py) is batched torch and runs as one CUDA graph; for a single car on
the Jetson's CPU the same code is ~19,000 tiny tensor ops (about 10 ms on a laptop core), all
launch overhead. This module is that computation written as scalar loops, compiled once with
Numba into a few microseconds. tests/test_mpc_fast.py checks it against mpc.solve() in float64
to 1e-6, so the car runs exactly what the policy was trained with.

Nothing here imports torch: the ROS node can run the policy on the GPU and the tracker here."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    from numba import njit
except Exception:                                                   # pragma: no cover - numba missing: plain python (slow but correct)
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

from .mpc import ACT_DIM, N_KNOTS, PlanSpec

NP = 25                                                             # dense path samples (mpc.path_points n)


def spec_array(spec: PlanSpec) -> np.ndarray:
    """PlanSpec -> flat float64 vector the compiled kernels read (numba cannot take the dataclass)."""
    return np.array([spec.kappa_max, spec.horizon_s, spec.len_min, spec.len_max, spec.v_cmd_lead, spec.N, spec.dt,
                     spec.delay, spec.k_us, spec.a_max, *spec.q, *spec.qf, *spec.r, *spec.rd, spec.iters], dtype=np.float64)


@njit(cache=True, fastmath=False)
def _path_points(k, Lp, x, y, psi, s):
    """mpc.path_points for one car: curvature linear between the 4 knots, heading its integral, path the integral of that."""
    n = x.shape[0]
    ds = Lp / (n - 1)
    kap_prev = 0.0
    psi[0] = 0.0; x[0] = 0.0; y[0] = 0.0; s[0] = 0.0
    for i in range(n):
        xi = i / (n - 1)
        pos = xi * (N_KNOTS - 1)
        i0 = int(math.floor(pos))
        if i0 > N_KNOTS - 2:
            i0 = N_KNOTS - 2
        w = pos - i0
        kap = k[i0] * (1.0 - w) + k[i0 + 1] * w
        s[i] = xi * Lp
        if i > 0:
            psi[i] = psi[i - 1] + 0.5 * (kap + kap_prev) * ds
            x[i] = x[i - 1] + 0.5 * (math.cos(psi[i]) + math.cos(psi[i - 1])) * ds
            y[i] = y[i - 1] + 0.5 * (math.sin(psi[i]) + math.sin(psi[i - 1])) * ds
        kap_prev = kap


@njit(cache=True, fastmath=False)
def _reference(k, Lp, v0, v1, v_now, a_max, dt, N, ref):
    """mpc.reference for one car -> ref (N+1, 4) = x, y, heading, target speed."""
    x = np.empty(NP); y = np.empty(NP); psi = np.empty(NP); s = np.empty(NP)
    _path_points(k, Lp, x, y, psi, s)
    S = s[NP - 1]
    if S < 1e-3:
        S = 1e-3
    T = N + 1
    st = 0.0
    vw = abs(v_now)
    if vw < 0.3:
        vw = 0.3
    pe = psi[NP - 1]
    for t in range(T):
        frac = st / S
        if frac < 0.0:
            frac = 0.0
        if frac > 1.0:
            frac = 1.0
        vt = v0 + (v1 - v0) * frac
        # searchsorted (left): first index with s[idx] >= st, clamped to [1, NP-1]
        idx = 0
        while idx < NP and s[idx] < st:
            idx += 1
        if idx < 1:
            idx = 1
        if idx > NP - 1:
            idx = NP - 1
        s_lo = s[idx - 1]; s_hi = s[idx]
        den = s_hi - s_lo
        if den < 1e-6:
            den = 1e-6
        w = (st - s_lo) / den
        if w < 0.0:
            w = 0.0
        if w > 1.0:
            w = 1.0
        xr = x[idx - 1] * (1.0 - w) + x[idx] * w
        yr = y[idx - 1] * (1.0 - w) + y[idx] * w
        pr = psi[idx - 1] * (1.0 - w) + psi[idx] * w
        over = st - S
        if over < 0.0:
            over = 0.0
        xr += over * math.cos(pe); yr += over * math.sin(pe)
        if st > S:
            pr = pe
        ref[t, 0] = xr; ref[t, 1] = yr; ref[t, 2] = pr; ref[t, 3] = vt
        if t < T - 1:
            dv = vt - vw
            lim = a_max * dt
            if dv < -lim:
                dv = -lim
            if dv > lim:
                dv = lim
            vw = vw + dv
            if vw < 0.3:
                vw = 0.3
            st = st + vw * dt


@njit(cache=True, fastmath=False)
def _dyn(z, d, a, k_us, wb, dt, out):
    v = z[3]
    Le = wb + k_us * v * v
    out[0] = z[0] + v * math.cos(z[2]) * dt
    out[1] = z[1] + v * math.sin(z[2]) * dt
    out[2] = z[2] + v * math.tan(d) / Le * dt
    vn = v + a * dt
    out[3] = vn if vn > 0.0 else 0.0
    out[4] = d
    out[5] = a


@njit(cache=True, fastmath=False)
def _ilqr(z0, ref, u_warm, sp, wb, s_max, v_max, u_out, z_out):
    """mpc.ilqr for one car. sp = spec_array(). u_out (N,2), z_out (N+1,6)."""
    N = int(sp[5]); dt = sp[6]; k_us = sp[8]; a_max = sp[9]
    q = sp[10:14]; qf = sp[14:18]; r = sp[18:20]; rd = sp[20:22]; iters = int(sp[22])
    u = u_warm.copy()
    z = np.empty((N + 1, 6))
    z[0, :] = z0
    for t in range(N):
        _dyn(z[t], u[t, 0], u[t, 1], k_us, wb, dt, z[t + 1])
    A = np.zeros((6, 6)); Bm = np.zeros((6, 2))
    Vz = np.empty(6); Vzz = np.empty((6, 6))
    Qz = np.empty(6); Qu = np.empty(2); Qzz = np.empty((6, 6)); Quu = np.empty((2, 2)); Quz = np.empty((2, 6))
    tmp6 = np.empty((6, 6)); tmp2 = np.empty((2, 6)); AtV = np.empty(6); BtV = np.empty(2)
    ks = np.empty((N, 2)); Ks = np.empty((N, 2, 6))
    un = np.empty((N, 2)); zn = np.empty((N + 1, 6))
    lo0 = -s_max; hi0 = s_max; lo1 = -a_max; hi1 = a_max
    for _ in range(iters):
        # terminal cost derivatives: Vz = 2 Qf e_N (Qf only weights x,y,psi,v), Vzz = 2 Qf
        for i in range(6):
            Vz[i] = 0.0
            for j in range(6):
                Vzz[i, j] = 0.0
        for i in range(4):
            Vz[i] = 2.0 * qf[i] * (z[N, i] - ref[N, i])
            Vzz[i, i] = 2.0 * qf[i]
        for t in range(N - 1, -1, -1):
            # jacobians
            psi = z[t, 2]; v = z[t, 3]; d = u[t, 0]
            Le = wb + k_us * v * v
            td = math.tan(d)
            for i in range(6):
                for j in range(6):
                    A[i, j] = 0.0
            for i in range(4):
                A[i, i] = 1.0
            A[0, 2] = -v * math.sin(psi) * dt; A[0, 3] = math.cos(psi) * dt
            A[1, 2] = v * math.cos(psi) * dt; A[1, 3] = math.sin(psi) * dt
            A[2, 3] = td * (wb - k_us * v * v) / (Le * Le) * dt
            for i in range(6):
                Bm[i, 0] = 0.0; Bm[i, 1] = 0.0
            Bm[2, 0] = v / Le * (1.0 + td * td) * dt; Bm[3, 1] = dt
            Bm[4, 0] = 1.0; Bm[5, 1] = 1.0
            # cost derivatives (quadratic, exact)
            du0 = u[t, 0] - z[t, 4]; du1 = u[t, 1] - z[t, 5]
            # A^T Vz, B^T Vz
            for i in range(6):
                acc = 0.0
                for m in range(6):
                    acc += A[m, i] * Vz[m]
                AtV[i] = acc
            for i in range(2):
                acc = 0.0
                for m in range(6):
                    acc += Bm[m, i] * Vz[m]
                BtV[i] = acc
            for i in range(4):
                Qz[i] = 2.0 * q[i] * (z[t, i] - ref[t, i]) + AtV[i]
            Qz[4] = -2.0 * rd[0] * du0 + AtV[4]
            Qz[5] = -2.0 * rd[1] * du1 + AtV[5]
            Qu[0] = 2.0 * r[0] * u[t, 0] + 2.0 * rd[0] * du0 + BtV[0]
            Qu[1] = 2.0 * r[1] * u[t, 1] + 2.0 * rd[1] * du1 + BtV[1]
            # tmp6 = Vzz A ; Qzz = lzz + A^T tmp6
            for i in range(6):
                for j in range(6):
                    acc = 0.0
                    for m in range(6):
                        acc += Vzz[i, m] * A[m, j]
                    tmp6[i, j] = acc
            for i in range(6):
                for j in range(6):
                    acc = 0.0
                    for m in range(6):
                        acc += A[m, i] * tmp6[m, j]
                    Qzz[i, j] = acc
            for i in range(4):
                Qzz[i, i] += 2.0 * q[i]
            Qzz[4, 4] += 2.0 * rd[0]; Qzz[5, 5] += 2.0 * rd[1]
            # Quz = luz + B^T Vzz A = luz + B^T tmp6
            for i in range(2):
                for j in range(6):
                    acc = 0.0
                    for m in range(6):
                        acc += Bm[m, i] * tmp6[m, j]
                    Quz[i, j] = acc
            Quz[0, 4] += -2.0 * rd[0]; Quz[1, 5] += -2.0 * rd[1]
            # Quu = luu + B^T Vzz B + 1e-3 I ; tmp2 = B^T Vzz
            for i in range(2):
                for j in range(6):
                    acc = 0.0
                    for m in range(6):
                        acc += Bm[m, i] * Vzz[m, j]
                    tmp2[i, j] = acc
            for i in range(2):
                for j in range(2):
                    acc = 0.0
                    for m in range(6):
                        acc += tmp2[i, m] * Bm[m, j]
                    Quu[i, j] = acc
            Quu[0, 0] += 2.0 * (r[0] + rd[0]) + 1e-3; Quu[1, 1] += 2.0 * (r[1] + rd[1]) + 1e-3
            det = Quu[0, 0] * Quu[1, 1] - Quu[0, 1] * Quu[1, 0]
            i00 = Quu[1, 1] / det; i01 = -Quu[0, 1] / det; i10 = -Quu[1, 0] / det; i11 = Quu[0, 0] / det
            k0 = -(i00 * Qu[0] + i01 * Qu[1]); k1 = -(i10 * Qu[0] + i11 * Qu[1])
            ks[t, 0] = k0; ks[t, 1] = k1
            for j in range(6):
                Ks[t, 0, j] = -(i00 * Quz[0, j] + i01 * Quz[1, j])
                Ks[t, 1, j] = -(i10 * Quz[0, j] + i11 * Quz[1, j])
            # Vz = Qz + K^T (Quu k + Qu) + Quz^T k
            qk0 = Quu[0, 0] * k0 + Quu[0, 1] * k1 + Qu[0]
            qk1 = Quu[1, 0] * k0 + Quu[1, 1] * k1 + Qu[1]
            for i in range(6):
                Vz[i] = Qz[i] + Ks[t, 0, i] * qk0 + Ks[t, 1, i] * qk1 + Quz[0, i] * k0 + Quz[1, i] * k1
            # Vzz = Qzz + K^T Quu K + K^T Quz + Quz^T K, symmetrized ; tmp2 = Quu K
            for i in range(2):
                for j in range(6):
                    tmp2[i, j] = Quu[i, 0] * Ks[t, 0, j] + Quu[i, 1] * Ks[t, 1, j]
            for i in range(6):
                for j in range(6):
                    tmp6[i, j] = (Qzz[i, j] + Ks[t, 0, i] * tmp2[0, j] + Ks[t, 1, i] * tmp2[1, j]
                                  + Ks[t, 0, i] * Quz[0, j] + Ks[t, 1, i] * Quz[1, j]
                                  + Quz[0, i] * Ks[t, 0, j] + Quz[1, i] * Ks[t, 1, j])
            for i in range(6):
                for j in range(6):
                    Vzz[i, j] = 0.5 * (tmp6[i, j] + tmp6[j, i])
        # forward pass with clamped inputs
        zn[0, :] = z0
        for t in range(N):
            dz0 = zn[t, 0] - z[t, 0]; dz1 = zn[t, 1] - z[t, 1]; dz2 = zn[t, 2] - z[t, 2]
            dz3 = zn[t, 3] - z[t, 3]; dz4 = zn[t, 4] - z[t, 4]; dz5 = zn[t, 5] - z[t, 5]
            u0 = u[t, 0] + ks[t, 0] + (Ks[t, 0, 0] * dz0 + Ks[t, 0, 1] * dz1 + Ks[t, 0, 2] * dz2 + Ks[t, 0, 3] * dz3 + Ks[t, 0, 4] * dz4 + Ks[t, 0, 5] * dz5)
            u1 = u[t, 1] + ks[t, 1] + (Ks[t, 1, 0] * dz0 + Ks[t, 1, 1] * dz1 + Ks[t, 1, 2] * dz2 + Ks[t, 1, 3] * dz3 + Ks[t, 1, 4] * dz4 + Ks[t, 1, 5] * dz5)
            if u0 > hi0:
                u0 = hi0
            if u0 < lo0:
                u0 = lo0
            if u1 > hi1:
                u1 = hi1
            if u1 < lo1:
                u1 = lo1
            cap = (v_max - zn[t, 3]) / dt                       # never command past v_max
            if u1 > cap:
                u1 = cap
            un[t, 0] = u0; un[t, 1] = u1
            _dyn(zn[t], u0, u1, k_us, wb, dt, zn[t + 1])
        for t in range(N):
            u[t, 0] = un[t, 0]; u[t, 1] = un[t, 1]
        for t in range(N + 1):
            for i in range(6):
                z[t, i] = zn[t, i]
    for t in range(N):
        u_out[t, 0] = u[t, 0]; u_out[t, 1] = u[t, 1]
    for t in range(N + 1):
        for i in range(6):
            z_out[t, i] = z[t, i]


@njit(cache=True, fastmath=False)
def _solve(action, v_meas, speed_cap, yaw_rate, delay, u_prev, warm, sp, wb, s_max, v_max, u_out, z_out, ref_out):
    """mpc.solve for one car: decode the plan, build the reference, predict over the latency, run the iLQR."""
    kappa_max = sp[0]; horizon_s = sp[1]; len_min = sp[2]; len_max = sp[3]; N = int(sp[5]); dt = sp[6]; k_us = sp[8]; a_max = sp[9]
    a = np.empty(ACT_DIM)
    for i in range(ACT_DIM):
        ai = action[i]
        if ai > 1.0:
            ai = 1.0
        if ai < -1.0:
            ai = -1.0
        a[i] = ai
    v = abs(v_meas)
    Lp = horizon_s * v
    if Lp < len_min:
        Lp = len_min
    if Lp > len_max:
        Lp = len_max
    k = np.empty(N_KNOTS)
    for i in range(N_KNOTS):
        k[i] = a[i] * kappa_max
    v0 = (a[N_KNOTS] + 1.0) * 0.5 * v_max
    if v0 > speed_cap:
        v0 = speed_cap
    v1 = (a[N_KNOTS + 1] + 1.0) * 0.5 * v_max
    if v1 > speed_cap:
        v1 = speed_cap
    _reference(k, Lp, v0, v1, v_meas, a_max, dt, N, ref_out)
    Le = wb + k_us * v * v
    vc = v if v > 0.5 else 0.5
    steer_now = math.atan(yaw_rate * Le / vc)
    if steer_now > s_max:
        steer_now = s_max
    if steer_now < -s_max:
        steer_now = -s_max
    psi0 = yaw_rate * delay
    z0 = np.empty(6)
    z0[0] = v * delay; z0[1] = 0.5 * v * psi0 * delay; z0[2] = psi0
    vd = v + u_prev[1] * delay
    z0[3] = vd if vd > 0.0 else 0.0
    z0[4] = steer_now if v > 0.5 else u_prev[0]
    z0[5] = u_prev[1]
    _ilqr(z0, ref_out, warm, sp, wb, s_max, v_max, u_out, z_out)


def solve(action, v_meas: float, speed_cap: float, yaw_rate: float, delay: float, u_prev, warm, spec: PlanSpec,
          wb: float, s_max: float, v_max: float):
    """numpy front end: action (6,), u_prev (2,), warm (N,2) -> (u (N,2), z (N+1,6), ref (N+1,4)) float64."""
    sp = spec_array(spec); N = spec.N
    u = np.empty((N, 2)); z = np.empty((N + 1, 6)); ref = np.empty((N + 1, 4))
    _solve(np.asarray(action, dtype=np.float64).reshape(-1), float(v_meas), float(speed_cap), float(yaw_rate), float(delay),
           np.asarray(u_prev, dtype=np.float64).reshape(-1), np.ascontiguousarray(warm, dtype=np.float64), sp,
           float(wb), float(s_max), float(v_max), u, z, ref)
    return u, z, ref


class PlanTrackerFast:
    """Single-car mpc.PlanTracker on numpy: same state (last command, warm start), same outputs.
    __call__(action (6,), v_meas, speed_cap, yaw_rate=None, delay=None) -> (steer [rad], speed cmd [m/s])."""

    def __init__(self, wheelbase: float, s_max: float, v_max: float, spec: Optional[PlanSpec] = None):
        self.wb, self.s_max, self.v_max = float(wheelbase), float(s_max), float(v_max)
        self.spec = spec or PlanSpec(); self.sp = spec_array(self.spec)
        N = self.spec.N
        self.u_prev = np.zeros(2); self.u_seq = np.zeros((N, 2))
        self.u = np.empty((N, 2)); self.z = np.empty((N + 1, 6)); self.ref = np.empty((N + 1, 4))
        self.last_ref = None; self.last_pred = None
        self.warmup()

    def warmup(self):
        """Trigger (or load from cache) the numba compilation so the first control step is not slow."""
        self(np.zeros(ACT_DIM), 0.0, 1.0); self.reset()

    def reset(self):
        self.u_prev[:] = 0.0; self.u_seq[:] = 0.0

    def __call__(self, action, v_meas: float, speed_cap: float, yaw_rate: Optional[float] = None, delay: Optional[float] = None) -> Tuple[float, float]:
        sp = self.spec
        v = abs(float(v_meas))
        if yaw_rate is None:                                       # no IMU: assume the last command took
            yaw_rate = v * math.tan(self.u_prev[0]) / (self.wb + sp.k_us * v * v)
        if delay is None:
            delay = sp.delay
        warm = np.empty_like(self.u_seq); warm[:-1] = self.u_seq[1:]; warm[-1] = self.u_seq[-1]
        _solve(np.asarray(action, dtype=np.float64).reshape(-1), float(v_meas), float(speed_cap), float(yaw_rate), float(delay),
               self.u_prev, warm, self.sp, self.wb, self.s_max, self.v_max, self.u, self.z, self.ref)
        self.u_seq[:] = self.u; self.u_prev[:] = self.u[0]
        self.last_ref = self.ref.copy(); self.last_pred = self.z[:, :4].copy()
        k_ = max(1, int(round(sp.v_cmd_lead / sp.dt)))
        return float(self.u[0, 0]), float(min(self.z[k_, 3], speed_cap))
