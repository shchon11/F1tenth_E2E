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

N_KNOTS = 6                       # 6 knots over a 15 m plan is one every 3 m; 4 would be every 5 m
ACT_DIM = N_KNOTS + 2
XI = torch.linspace(0.0, 1.0, N_KNOTS)

#: How the plan says how fast. "linear" is the action space every checkpoint was trained in and the
#: default; the other two are opt-in (`PlanSpec.speed_mode`) and change what the speed dimensions mean.
#:
#:   linear    2 numbers: the target speed 0.15 s ahead and at the end of the plan, linear between.
#:             A straight line has its minimum at an end, so an apex *inside* the plan cannot be said:
#:             measured against the teacher's own profile over the 0.6 s the tracker consumes, the
#:             line asks for > 0.5 m/s more than the profile allows in 49-60 % of windows (p90 1.1-1.8
#:             m/s), and the faster the teacher the worse it gets.
#:   envelope  2 numbers: `a_hat`, the lateral acceleration the policy believes it can use, and
#:             `v_end`, the speed to arrive at the end of the plan with. The profile is
#:             v(s) = sqrt(a_hat / |kappa(s)|) on the plan's *own* curvature, braked backwards from
#:             v_end and ramped forwards from the measured speed -- so it follows the corner by
#:             construction, and one scalar carries "how slippery is it". Same test: 2-4 %, p90 0.2-0.35.
#:   knots     N_KNOTS numbers: the speed at each curvature knot, linear between. Same test: 0.3-0.5 %.
SPEED_MODES = ("linear", "envelope", "knots")


def act_dim(speed_mode: str = "linear") -> int:
    if speed_mode not in SPEED_MODES:
        raise ValueError(f"speed_mode must be one of {SPEED_MODES}, got {speed_mode!r}")
    return N_KNOTS + (N_KNOTS if speed_mode == "knots" else 2)


@dataclass
class PlanSpec:
    kappa_max: float = 1.6         # [1/m] curvature range of the knots (the car's full-lock radius is ~0.74 m)
    horizon_s: float = 1.5         # plan length = horizon_s * v, clamped to [len_min, len_max]
    len_min: float = 2.0
    len_max: float = 15.0          # at 10 m/s the plan must be able to hold a braking manoeuvre:
                                   # slowing 10 -> 4 m/s needs 7 m at 6 m/s^2. The old 0.7 s / 6 m
                                   # plan covered 5.6 m at 8 m/s against a 9.2 m need, so the plan
                                   # could not even represent the manoeuvre the corner required.
    # Backing out of something. The speed dimension maps to [0, v_max] and a checkpoint's action
    # space cannot be widened without re-meaning every plan it has ever emitted, so reverse is not
    # a dimension of its own: it is what "stop" means when the car is *already* stopped and in
    # contact with something. Wedged nose-first into a hose or a crate, forward is the obstacle,
    # and a policy that commands zero speed there is asking to get out.
    #
    # The policy still chooses it -- the speed command and the steering are both its own -- and it
    # cannot be entered any other way, so an ordinary slow corner is untouched. What it cannot
    # choose is how fast to reverse; `v_reverse` is fixed, which for un-wedging is the behaviour
    # anyway. Making it a real dimension would cost the three encoders, the two decoders, the ten
    # places that address the speed knots by position, the tracker, the checkpoint growth path and
    # the plan viewer, and is written down here rather than left as an unstated shortcut.
    v_reverse: float = 0.8         # [m/s] how fast backing out goes
    reverse_v_gate: float = 0.4    # [m/s] only from near standstill -- you cannot reverse at pace
    reverse_cmd_gate: float = 0.15 # [m/s] a commanded speed under this is "stop", i.e. the request
    v_cmd_lead: float = 0.15       # [s] the first speed target refers to this far ahead
    N: int = 12                    # MPC steps
    dt: float = 0.05               # [s] MPC step
    delay: float = 0.035           # [s] nominal command latency the tracker predicts over (delay + half the servo lag)
    k_us: float = 0.003            # [s^2/m] understeer: effective wheelbase L + k_us v^2
    a_max: float = 6.0             # [m/s^2] tracker acceleration bound (drive side)
    a_brake: float = 5.0           # [m/s^2] braking bound. Asymmetric because the car is: braking is
                                   # limited by the VESC regen current at -4.2 to -5.7 m/s^2 while the
                                   # drive side reaches +7. A symmetric +-6 let the tracker plan a
                                   # deceleration 20 % deeper than the car can produce, so it arrived
                                   # at corners faster than its own reference said it would -- the
                                   # error is silent, and it is on the dangerous side.
    q: Tuple[float, float, float, float] = (1.0, 6.0, 1.0, 0.4)    # x, y, heading, speed tracking weights
    qf: Tuple[float, float, float, float] = (2.0, 10.0, 2.0, 0.4)
    r: Tuple[float, float] = (0.3, 0.02)                            # steer, accel effort
    rd: Tuple[float, float] = (6.0, 0.05)                           # steer, accel rate
    iters: int = 2
    speed_mode: str = "linear"     # SPEED_MODES; anything but "linear" re-interprets the speed dimensions
    a_hat_max: float = 12.0        # [m/s^2] envelope: the lateral-acceleration belief spans [a_hat_min, a_hat_max]
    a_hat_min: float = 1.0
    a_brake_profile: float = 4.0   # [m/s^2] envelope: braking the backward pass plans with (the teacher's a_brake)
    n_profile: int = 25            # dense samples of the speed profile, the same grid `path_points` uses
    envelope_forward: bool = True  # envelope: ramp the profile up from the measured speed (ellipse-limited drive)


def plan_length(v: torch.Tensor, spec: PlanSpec) -> torch.Tensor:
    return (spec.horizon_s * v.abs()).clamp(spec.len_min, spec.len_max)


def decode(action: torch.Tensor, v_meas: torch.Tensor, v_max: float, speed_cap: torch.Tensor, spec: PlanSpec):
    """normalized action (B,6) -> (kappa knots (B,4) [1/m], L_p (B,), v_start (B,), v_end (B,))"""
    if spec.speed_mode != "linear":
        # Every runtime layer that reads a plan (grip arms, clearance, viewers) reads it through here
        # and means (v_start, v_end) by the last two numbers. Under another speed mode they are not
        # speeds at all, and the result would be a plausible-looking wrong plan rather than an error.
        raise ValueError(f"mpc.decode reads the 'linear' speed dimensions; this plan is {spec.speed_mode!r} "
                         f"-- use decode_profile")
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


def encode_envelope(kappas: torch.Tensor, a_hat: torch.Tensor, v_end: torch.Tensor, v_max: float, spec: PlanSpec) -> torch.Tensor:
    """curvature knots [1/m], usable lateral acceleration [m/s^2], end speed [m/s] -> normalized action"""
    return torch.cat([(kappas / spec.kappa_max).clamp(-1, 1), (a_hat / spec.a_hat_max * 2 - 1).clamp(-1, 1)[:, None],
                      (v_end / v_max * 2 - 1).clamp(-1, 1)[:, None]], 1)


def encode_knots(kappas: torch.Tensor, v_knots: torch.Tensor, v_max: float, spec: PlanSpec) -> torch.Tensor:
    """curvature knots [1/m], speed at each knot [m/s] -> normalized action (B, 2 * N_KNOTS)"""
    return torch.cat([(kappas / spec.kappa_max).clamp(-1, 1), (v_knots / v_max * 2 - 1).clamp(-1, 1)], 1)


def decode_profile(action: torch.Tensor, v_meas: torch.Tensor, v_max: float, speed_cap: torch.Tensor, spec: PlanSpec):
    """'envelope' / 'knots' action -> (kappa knots (B,K) [1/m], L_p (B,), speed profile (B, n_profile) [m/s]).

    The profile is on `path_points`' own uniform arc grid, which is also the grid `reference`
    interpolates a `v_limit` on -- so it is handed to `reference` as one and nothing else changes."""
    a = action.clamp(-1.0, 1.0)
    Lp = plan_length(v_meas, spec)
    k = a[:, :N_KNOTS] * spec.kappa_max
    n = spec.n_profile
    cap = speed_cap[:, None]
    if spec.speed_mode == "knots":
        vk = torch.minimum((a[:, N_KNOTS:2 * N_KNOTS] + 1.0) * 0.5 * v_max, cap)
        pos = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)[None] * (N_KNOTS - 1)
        i0 = pos.floor().clamp(max=N_KNOTS - 2).long(); w = pos - i0.to(a.dtype)
        i0 = i0.expand(a.shape[0], n)
        return k, Lp, vk.gather(1, i0) * (1 - w) + vk.gather(1, i0 + 1) * w
    if spec.speed_mode != "envelope":
        raise ValueError(f"decode_profile is for the 'envelope' and 'knots' modes, got {spec.speed_mode!r}")
    a_hat = ((a[:, N_KNOTS] + 1.0) * 0.5 * spec.a_hat_max).clamp_min(spec.a_hat_min)[:, None]
    v_end = torch.minimum((a[:, N_KNOTS + 1] + 1.0) * 0.5 * v_max, speed_cap)
    _, _, _, _, kap = path_points(k, Lp, n=n, return_kappa=True)
    g = kap.abs().clamp_min(1e-3) / a_hat                                   # 1 / v_curve^2
    ds = (Lp / (n - 1)).clamp_min(1e-3)
    cols = list(torch.minimum(torch.rsqrt(g), cap).unbind(1))
    cols[-1] = torch.minimum(cols[-1], v_end)
    # The friction ellipse on both passes, with a_hat as the whole budget: what is spent turning is
    # not there to brake or drive with. Python constants, so the loops unroll and stay graph-safe.
    for i in range(n - 2, -1, -1):          # backward: slow in time for what is ahead
        ax = spec.a_brake_profile * torch.sqrt((1.0 - (cols[i + 1] ** 2 * g[:, i + 1]) ** 2).clamp_min(0.0))
        cols[i] = torch.minimum(cols[i], torch.sqrt(cols[i + 1] ** 2 + 2.0 * ax * ds))
    if spec.envelope_forward:
        cols[0] = torch.minimum(cols[0], v_meas.abs().clamp_min(0.5))
        for i in range(1, n):               # forward: no more than can be gained from the speed it has
            ax = spec.a_max * torch.sqrt((1.0 - (cols[i - 1] ** 2 * g[:, i - 1]) ** 2).clamp_min(0.0))
            cols[i] = torch.minimum(cols[i], torch.sqrt(cols[i - 1] ** 2 + 2.0 * ax * ds))
    return k, Lp, torch.stack(cols, 1)


def path_points(k: torch.Tensor, Lp: torch.Tensor, n: int = 25, return_kappa: bool = False):
    """Dense samples of the plan in the body frame: x, y, heading (B,n) and arc length (B,n).
    Curvature is linear between the knots; heading is its integral, the path the integral of that.

    `return_kappa` also hands back the interpolated curvature at those samples. It is computed here
    either way; a speed envelope needs it and would otherwise re-derive the same interpolation."""
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
    if return_kappa:
        return x, y, psi, s, kap
    return x, y, psi, s


def reference(k, Lp, v0, v1, spec: PlanSpec, v_now: Optional[torch.Tensor] = None,
              v_limit: Optional[torch.Tensor] = None, a_walk: Optional[torch.Tensor] = None,
              v_profile: Optional[torch.Tensor] = None):
    """Time-indexed reference (B, N+1, 4) = x, y, heading, speed along the plan. The speed *target*
    is the plan's profile (v0 at the start, linear in arc length to v1 at the end); the position
    reference is walked with the speed the car can actually have: from v_now (measured) towards
    the profile within a_max, so a car that is slower than its plan is not chased by a reference
    that has run ahead along the curve (time-free tracking, like pure pursuit).

    `v_limit` (B, n) caps the speed target at each of `path_points`' samples -- a friction-derived
    envelope, supplied by the caller; it is interpolated onto the walked arc length the same way the
    pose is. `a_walk` (B, 2) replaces `spec.a_brake` / `spec.a_max` in the walk with per-env values.
    ``v_profile`` replaces the speed target (including its measured initial state).
    ``a_walk`` also accepts local spatial budgets (B,n,2), interpolated at the walked arc.
    Optional inputs default to None, preserving the original behaviour exactly."""
    x, y, psi, s = path_points(k, Lp)
    S = s[:, -1].clamp_min(1e-3)
    dev = k.device; T = spec.N + 1
    st = torch.zeros(k.shape[0], T, device=dev, dtype=k.dtype); v = torch.zeros_like(st)
    vw = (v0 if v_now is None else v_now.abs()).clamp_min(0.0 if v_profile is not None else 0.3)
    local_walk = a_walk is not None and a_walk.ndim == 3
    a_brk_w = (-spec.a_brake * spec.dt) if a_walk is None else (-a_walk[..., 0] * spec.dt)
    a_acc_w = (spec.a_max * spec.dt) if a_walk is None else (a_walk[..., 1] * spec.dt)
    for t_ in range(T):
        frac = (st[:, t_] / S).clamp(0.0, 1.0)
        v[:, t_] = v0 + (v1 - v0) * frac                                   # target profile at that point of the path
        if v_profile is not None:
            pos = frac * (v_profile.shape[1] - 1)
            ip = pos.floor().clamp(max=v_profile.shape[1] - 2).long()
            wp = pos - ip.to(v.dtype)
            vp0 = v_profile.gather(1, ip[:, None])[:, 0]
            vp1 = v_profile.gather(1, ip[:, None] + 1)[:, 0]
            # The spatial planner constrains squared speed (v_next^2-v^2=2*a*ds).
            # Its edge acceleration advances time even from rest; a zero profile
            # produces neither fictitious motion nor a launch deadlock.
            v[:, t_] = (vp0.square() * (1 - wp) + vp1.square() * wp).clamp_min(0).sqrt()
            edge_accel = (vp1.square() - vp0.square()) / (2 * S / (v_profile.shape[1] - 1))
            edge_accel = torch.where(st[:, t_] < S, edge_accel, torch.zeros_like(edge_accel))
        if v_limit is not None:
            # the envelope at this arc length, linearly interpolated between the path samples
            pos = frac * (v_limit.shape[1] - 1)
            i_ = pos.floor().clamp(max=v_limit.shape[1] - 2).long()
            wgt = pos - i_.to(v.dtype)
            cap = v_limit.gather(1, i_[:, None])[:, 0] * (1 - wgt) + v_limit.gather(1, i_[:, None] + 1)[:, 0] * wgt
            v[:, t_] = torch.minimum(v[:, t_], cap)
        if t_ < T - 1:
            if local_walk:
                pos = frac * (a_walk.shape[1] - 1)
                iw = pos.floor().clamp(max=a_walk.shape[1] - 2).long()
                ww = pos - iw.to(v.dtype)
                bw = a_walk.gather(1, iw[:, None, None].expand(-1, 1, 2))[:, 0] * (1 - ww[:, None]) + a_walk.gather(1, (iw + 1)[:, None, None].expand(-1, 1, 2))[:, 0] * ww[:, None]
                a_brk_w, a_acc_w = -bw[:, 0] * spec.dt, bw[:, 1] * spec.dt
            demand = v[:, t_] - vw
            if v_profile is not None:
                demand = demand + edge_accel * spec.dt
            dv = torch.clamp(demand, a_brk_w, a_acc_w) if a_walk is None else torch.maximum(torch.minimum(demand, a_acc_w), a_brk_w)
            old_vw = vw
            vw = (vw + dv).clamp_min(0.0 if v_profile is not None else 0.3)
            step_speed = .5 * (old_vw + vw) if v_profile is not None else vw
            st[:, t_ + 1] = st[:, t_] + step_speed * spec.dt
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


def build_ilqr_consts(spec: PlanSpec, s_max: float, device, dtype=torch.float32):
    """`ilqr`'s read-only cost and bound tensors, built once by a caller that wants to reuse them.

    Optional. Passing `consts=None` (the default everywhere) builds them inline exactly as before,
    so training and every existing caller are unchanged.

    It exists because those `torch.tensor(...)` calls are what stops `solve` being CUDA-graph
    capturable: a CUDA tensor built from a Python tuple is a pageable host-to-device copy, and the
    first one -- `spec.q` -- fails capture with `cudaErrorStreamCaptureInvalidated`. The viewer
    builds this tuple once per session and binds it into its own solver; nothing is cached
    globally, so the tensors live and die with the session that made them.

    `dtype` defaults to float32 because that is what the inline branch produces: it writes
    `torch.zeros(6, 6, device=dev)` with no dtype, i.e. torch's default. Passing anything else is
    *not* equivalent to the inline path -- it would carry a different precision through `Vzz` and
    `Quu` -- so the viewer's fast path only opts in for a float32 session and leaves this default.
    """
    dev = torch.device(device)
    Q = torch.zeros(6, 6, device=dev, dtype=dtype); Q[:4, :4] = torch.diag(torch.tensor(spec.q, device=dev, dtype=dtype))
    Qf = torch.zeros(6, 6, device=dev, dtype=dtype); Qf[:4, :4] = torch.diag(torch.tensor(spec.qf, device=dev, dtype=dtype))
    R = torch.diag(torch.tensor(spec.r, device=dev, dtype=dtype)); Rd = torch.diag(torch.tensor(spec.rd, device=dev, dtype=dtype))
    Q[4:, 4:] = Rd
    lo = torch.tensor([-s_max, -spec.a_brake], device=dev, dtype=dtype)
    hi = torch.tensor([s_max, spec.a_max], device=dev, dtype=dtype)
    E = torch.zeros(6, 2, device=dev, dtype=dtype); E[4, 0] = 1.0; E[5, 1] = 1.0
    return (Q, Qf, R, Rd, lo, hi, E, torch.eye(2, device=dev, dtype=dtype))


def ilqr(z0: torch.Tensor, ref: torch.Tensor, u_warm: torch.Tensor, spec: PlanSpec, wb: float, s_max: float, v_max: float,
         consts=None, bounds: Optional[torch.Tensor] = None, projector=None):
    """z0 (B,6), ref (B,N+1,4), warm (B,N,2) -> controls and augmented states.

    Bounds may be per environment (B,2,2) or stage (B,N,2,2). An optional
    projector(state, proposed_control, stage) owns the complete projection and
    runs on initial warm controls and every forward rollout at its actual state.
    """
    B, N, dev = z0.shape[0], spec.N, z0.device
    if consts is None:                                                 # unchanged path: build inline
        Q = torch.zeros(6, 6, device=dev); Q[:4, :4] = torch.diag(torch.tensor(spec.q, device=dev))
        Qf = torch.zeros(6, 6, device=dev); Qf[:4, :4] = torch.diag(torch.tensor(spec.qf, device=dev))
        R = torch.diag(torch.tensor(spec.r, device=dev)); Rd = torch.diag(torch.tensor(spec.rd, device=dev))
        Q[4:, 4:] = Rd                                                 # (u - u_prev)^T Rd (u - u_prev): u_prev lives in the state
        lo = torch.tensor([-s_max, -spec.a_brake], device=dev); hi = torch.tensor([s_max, spec.a_max], device=dev)
        E = torch.zeros(6, 2, device=dev); E[4, 0] = 1.0; E[5, 1] = 1.0  # picks u_prev out of z
        eye2 = torch.eye(2, device=dev)
    else:
        Q, Qf, R, Rd, lo, hi, E, eye2 = consts
    if bounds is not None:
        # Per-env control bounds: (B, 2, 2) as [[steer_lo, accel_lo], [steer_hi, accel_hi]].
        # Limiting only the reference would leave the solver free to command past the budget, so the
        # same numbers have to reach the clamp in the forward rollout below.
        lo, hi = (bounds[:, :, 0], bounds[:, :, 1]) if bounds.ndim == 4 else (bounds[:, 0], bounds[:, 1])

    def project(ut, state, t):
        if projector is not None:
            return projector(state, ut, t)
        lt, ht = (lo[:, t], hi[:, t]) if bounds is not None and bounds.ndim == 4 else (lo, hi)
        ut = torch.maximum(torch.minimum(ut, ht), lt)
        ut[:, 1] = torch.minimum(ut[:, 1], (v_max - state[:, 3]) / spec.dt)
        if bounds is not None:
            ut[:, 1] = torch.maximum(ut[:, 1], lt[:, 1])
        return ut

    def rollout(u):
        z = [z0]
        controls = []
        for t in range(N):
            # Historical warm starts stay unchanged. New local/stage constraints apply before
            # linearization, so the backward pass never linearizes an unchecked warm trajectory.
            ut = project(u[:, t], z[-1], t) if projector is not None or (bounds is not None and bounds.ndim == 4) else u[:, t]
            controls.append(ut)
            z.append(_dyn(z[-1], ut, spec, wb))
        return torch.stack(controls, 1), torch.stack(z, 1)

    u, z = rollout(u_warm.clone())
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
            Quu = luu + Bm.transpose(1, 2) @ Vzz @ Bm + 1e-3 * eye2
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
            ut = project(ut, zn, t)
            un[:, t] = ut; zn = _dyn(zn, ut, spec, wb); zs.append(zn)
        u, z = un, torch.stack(zs, 1)
    return u, z


def solve(action, v_meas, speed_cap, yaw_rate, delay, u_prev, warm, spec: PlanSpec, wb: float, s_max: float, v_max: float,
          consts=None, v_limit: Optional[torch.Tensor] = None, bounds: Optional[torch.Tensor] = None,
          a_walk: Optional[torch.Tensor] = None):
    """Whole tracker step as one function (compiled into a single CUDA graph on the GPU): decode the plan,
    build the reference, predict over the latency, run the iLQR. Returns (u (B,N,2), z (B,N+1,6), ref)."""
    if spec.speed_mode == "linear":
        k, Lp, v0, v1 = decode(action, v_meas, v_max, speed_cap, spec)
        ref = reference(k, Lp, v0, v1, spec, v_meas, v_limit=v_limit, a_walk=a_walk)
    else:
        # The plan carries a whole profile. `reference` already knows how to follow one -- that is
        # what `v_limit` is -- so the linear target is set to the cap and the profile does the rest.
        k, Lp, prof = decode_profile(action, v_meas, v_max, speed_cap, spec)
        if v_limit is not None:
            prof = torch.minimum(prof, v_limit)
        ref = reference(k, Lp, speed_cap, speed_cap, spec, v_meas, v_limit=prof, a_walk=a_walk)
    v = v_meas.abs()
    Le = wb + spec.k_us * v * v
    steer_now = torch.atan(yaw_rate * Le / v.clamp_min(0.5)).clamp(-s_max, s_max)
    psi0 = yaw_rate * delay
    z0 = torch.stack([v * delay, 0.5 * v * psi0 * delay, psi0, (v + u_prev[:, 1] * delay).clamp_min(0.0),
                      torch.where(v > 0.5, steer_now, u_prev[:, 0]), u_prev[:, 1]], 1)
    u, z = ilqr(z0, ref, warm, spec, wb, s_max, v_max, consts, bounds=bounds)
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

    def __init__(self, num_envs: int, device, wheelbase: float, s_max: float, v_max: float, spec: Optional[PlanSpec] = None,
                 compile_solver: bool = True):
        self.B, self.device, self.wb, self.s_max, self.v_max = num_envs, torch.device(device), wheelbase, s_max, v_max
        self.spec = spec or PlanSpec()
        self.compile_solver = compile_solver
        # Per-instance solver override. `None` keeps the module-level selection below, which is what
        # every existing caller gets. The viewer's CUDA-graph fast path binds one here for its own
        # tracker only; nothing global is replaced, so other trackers in the process are unaffected.
        self._solver = None
        #: `ilqr` constants per cost/bound signature, built on first use (see `ilqr_consts`).
        self._consts_cache = {}
        # Optional nominal actuator conversion, installed only by the local automatic controller.
        self._command_hook = None
        self._input_hook = None
        self._reset_hook = None
        # Per-instance hook on the *plan*, run before anything is decoded. `None` is the untouched
        # path: no branch is taken and nothing is allocated, so a legacy tracker is what it was.
        # It exists because a runtime layer that adjusts the plan -- `learn/clearance.py` -- has to
        # act before `solve` builds `last_ref` from the action, which is both what the tracker
        # follows and what `crash_attribution.py` measures. Deliberately a different attribute from
        # `_solver`: the two layers then compose in either installation order.
        self._plan_hook = None
        self.u_prev = torch.zeros(num_envs, 2, device=self.device)                 # last applied steer, accel
        self.u_seq = torch.zeros(num_envs, self.spec.N, 2, device=self.device)     # warm start
        self.last_ref = None                                                       # (B,N+1,4) body frame plan, for viewers
        self.last_pred = None                                                      # (B,N+1,4) the tracker's predicted motion
        #: (B,) whether each car is in contact with something right now. `F1VecEnv` sets it under
        #: soft collision; left False, `reverse_cmd_gate` can never fire and the tracker is the one
        #: it has always been.
        self.contact = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def ilqr_consts(self, spec: PlanSpec):
        """`build_ilqr_consts` for `spec` on this tracker's device, built once and kept here.

        For a caller that runs `solve` on its own -- the interactive teacher's MPC preview -- inside
        a CUDA graph capture. Built inline (`consts=None`), the cost matrices are `torch.tensor` of
        Python tuples, a pageable host-to-device copy that a capture refuses; built here on the
        first, eager call, the capture only reads them. Kept on the tracker because a capture has to
        find every tensor a call reads from the environment (`graph_fastpath._tensor_paths`), and
        `env.tracker` is on that path.
        """
        key = (tuple(spec.q), tuple(spec.qf), tuple(spec.r), tuple(spec.rd),
               float(spec.a_brake), float(spec.a_max), float(self.s_max))
        c = self._consts_cache.get(key)
        if c is None:
            # A list, not the builder's tuple: a capture points each input's path at a private
            # clone for its duration, and an element of a tuple cannot be re-pointed.
            c = self._consts_cache[key] = list(build_ilqr_consts(spec, self.s_max, self.u_prev.device))
        return c

    def reset(self, ids: torch.Tensor):
        self.u_prev[ids] = 0.0; self.u_seq[ids] = 0.0
        if self._reset_hook is not None:
            self._reset_hook(ids)

    @torch.no_grad()
    def __call__(self, action: torch.Tensor, v_meas: torch.Tensor, speed_cap: torch.Tensor,
                 yaw_rate: Optional[torch.Tensor] = None, delay=None) -> torch.Tensor:
        """normalized plan (B,6), measured speed (B,), speed cap (B,), measured yaw rate (B,) (IMU gyro z,
        optional), delay: calibrated command latency [s] (float or (B,), default spec.delay)
        -> (steer [rad], speed cmd [m/s]) (B,2)"""
        sp = self.spec
        if self._input_hook is not None:
            action, v_meas, speed_cap, yaw_rate, delay = self._input_hook(action, v_meas, speed_cap, yaw_rate, delay)
        if self._plan_hook is not None:
            # Before the decode, so every consumer below -- the reference, the solver, `last_ref`,
            # `last_pred` and the command -- sees one plan, the adjusted one.
            action = self._plan_hook(action, v_meas, speed_cap)
        v = v_meas.abs()
        if yaw_rate is None:                                       # no IMU: assume the last command took
            yaw_rate = v * torch.tan(self.u_prev[:, 0]) / (self.wb + sp.k_us * v * v)
        if delay is None:
            delay = torch.full_like(v, sp.delay)
        elif not torch.is_tensor(delay):
            delay = torch.full_like(v, float(delay))
        warm = torch.cat([self.u_seq[:, 1:], self.u_seq[:, -1:]], 1)
        solver = self._solver or (solve_fast if self.compile_solver else solve)
        u, z, ref = solver(action, v_meas, speed_cap, yaw_rate, delay, self.u_prev, warm, sp, self.wb, self.s_max, self.v_max)
        u, z, ref = u.clone(), z.clone(), ref.clone()             # CUDA-graph outputs are reused by the next run
        self.u_seq = u; self.u_prev = u[:, 0].clone(); self.last_ref = ref; self.last_pred = z[:, :, :4]
        if self._command_hook is not None:
            return self._command_hook(u, z, v_meas, speed_cap)
        k_ = max(1, int(round(sp.v_cmd_lead / sp.dt)))
        v_cmd = torch.minimum(z[:, k_, 3], speed_cap)
        # Stopped, asking to stay stopped, and touching something: back out. See `v_reverse`.
        stalled = ((v_meas.abs() < sp.reverse_v_gate) & (v_cmd < sp.reverse_cmd_gate)
                   & self.contact)
        v_cmd = torch.where(stalled, torch.full_like(v_cmd, -sp.v_reverse), v_cmd)
        return torch.stack([u[:, 0, 0], v_cmd], 1)
