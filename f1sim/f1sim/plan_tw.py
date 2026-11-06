"""Time waypoints: the policy's plan as where the car is at t = dt, 2 dt, .., K dt (2026-11-06, the user: "t=n 까지의
waypoint 모음으로 만드는게 정석 아잉교?", and no MPC -- "실차에서는 mpc를 위한 차체 param을 알기가 너무 부담돼서").

Why: the 'wp4' plan is one forward path (curvature knots over 0.6 s of the measured speed) with reverse only as the
first speed below -1, so "back up 1 m, then turn out round the box" took ~20 decisions flipping the speed's sign and the
curvature's at one exact moment -- exploration never found it and the policy learned to stand behind boxes (11-06 box
world: 24-61 % of the cars started behind a box passed it under the mean action, 70 % of the closest ones stood still).
Here one plan says it: each of the K segments has its own signed speed, so a plan can run backwards and then forwards.

The action keeps the 8 numbers of the wire (`mpc.PLAN_WIRE_DIM`) and wp4's column layout, curvatures first:

    a[:, 0:K]   kappa_k = a * KAPPA_MAX       the steering curvature over segment k (+ = wheels left), either direction
    a[:, K:2K]  v_k = a * V_FWD (a >= 0)       the signed speed over segment k; a = 0 is a stop, the middle of the range
                    = a * V_REV (a < 0)

Segment k runs for DT seconds: a constant-curvature arc of signed length v_k * DT from the end of segment k - 1. A
negative length backs along the arc (heading changes by kappa * ds, as a car backing with that steering angle does).

The follower (`map_controller.MAPPlanTracker`) drives the first run -- the segments up to the first change of direction
-- and nothing after it: forward runs go to MAP as a waypoint array (0.1 m apart, the segment speeds, 0 past a change of
direction), backward runs to a pure pursuit on the points behind the car (wheelbase only). The policy plans again every
0.05 s, so by the time the car reaches a change of direction the next plan already starts on the other side of it."""
import math
from typing import Tuple

import torch

TW_K = 4                  # segments (the wire's 8 numbers: K curvatures + K speeds)
TW_DT = 0.375             # [s] per segment: 1.5 s ahead
KAPPA_MAX = 1.6           # [1/m] as wp4 (`mpc.PlanSpec.kappa_max`); the car's full lock is ~1.35
V_FWD = 10.0              # [m/s] a = +1
V_REV = 2.0               # [m/s] a = -1 (`EnvConfig.reverse_speed`)
V_STOP = 0.05             # [m/s] below this a segment is a stop: no direction of its own


def decode(action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B, 2K) action -> (kappa (B, K) [1/m], signed speed (B, K) [m/s])."""
    a = action.clamp(-1.0, 1.0)
    k = a[:, :TW_K] * KAPPA_MAX
    s = a[:, TW_K:2 * TW_K]
    v = torch.where(s >= 0, s * V_FWD, s * V_REV)
    return k, v


def encode(kappa: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The inverse of `decode` (clamped to the action range)."""
    ak = (kappa / KAPPA_MAX).clamp(-1.0, 1.0)
    av = torch.where(v >= 0, v / V_FWD, v / V_REV).clamp(-1.0, 1.0)
    return torch.cat([ak, av], 1)


def _arc(x, y, psi, kappa, ds):
    """End pose of a constant-curvature arc of signed length ds from (x, y, psi)."""
    dpsi = kappa * ds
    small = dpsi.abs() < 1e-6
    # chord of the arc: length 2 sin(dpsi/2) / kappa along psi + dpsi/2
    chord = torch.where(small, ds, 2.0 * torch.sin(0.5 * dpsi) / torch.where(small, torch.ones_like(kappa), kappa))
    mid = psi + 0.5 * dpsi
    return x + chord * torch.cos(mid), y + chord * torch.sin(mid), psi + dpsi


def waypoints(kappa: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """(B, K + 1, 4) body-frame x, y, heading, t at t = 0, DT, .., K DT (the car's pose first)."""
    B = kappa.shape[0]
    x = torch.zeros(B, device=kappa.device, dtype=kappa.dtype)
    y, psi = torch.zeros_like(x), torch.zeros_like(x)
    out = [torch.stack([x, y, psi, torch.zeros_like(x)], 1)]
    for k in range(TW_K):
        x, y, psi = _arc(x, y, psi, kappa[:, k], v[:, k] * TW_DT)
        out.append(torch.stack([x, y, psi, torch.full_like(x, (k + 1) * TW_DT)], 1))
    return torch.stack(out, 1)


def dense(kappa: torch.Tensor, v: torch.Tensor, n_per: int = 12) -> torch.Tensor:
    """(B, K n_per + 1, 5) body-frame x, y, heading, signed speed, t along the whole plan, uniform in time -- what the
    viewers draw (both directions) and the tracking tests compare against."""
    B = kappa.shape[0]
    x = torch.zeros(B, device=kappa.device, dtype=kappa.dtype)
    y, psi = torch.zeros_like(x), torch.zeros_like(x)
    pts = [torch.stack([x, y, psi, v[:, 0], torch.zeros_like(x)], 1)]
    for k in range(TW_K):
        ds = v[:, k] * TW_DT / n_per
        for j in range(n_per):
            x, y, psi = _arc(x, y, psi, kappa[:, k], ds)
            pts.append(torch.stack([x, y, psi, v[:, k], torch.full_like(x, (k + (j + 1) / n_per) * TW_DT)], 1))
    return torch.stack(pts, 1)


def first_run(v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(direction (B,) +1 / -1 / 0, segments (B, K) bool) of each plan's first run: from its first moving segment, the
    segments that move the same way, stops included, up to the first that moves the other way. Direction 0: the plan
    never moves."""
    mv = v.abs() >= V_STOP
    sg = torch.sign(v) * mv
    first = torch.where(mv.any(1), mv.float().argmax(1), torch.full((v.shape[0],), v.shape[1] - 1, device=v.device))
    d = sg.gather(1, first[:, None])[:, 0]
    k_ = torch.arange(v.shape[1], device=v.device)[None, :]
    flip = (sg == -d[:, None]) & (d[:, None] != 0)
    end = torch.where(flip.any(1), flip.float().argmax(1), torch.full_like(first, v.shape[1]))   # first other-way segment
    return d, k_ < end[:, None]


def run_path(kappa: torch.Tensor, v: torch.Tensor, spacing: float, n: int):
    """The first run as a waypoint array for the follower: (B, n, 2) body-frame points `spacing` apart along the run
    (driven forward for a forward run, backward for a backward one), (B, n) speed magnitude, (B, n) curvature, (B,)
    direction, (B,) run length [m]. Past the run's end the array goes straight on along the end heading (MAP's
    lookahead needs points there) at speed 0 when the run ends in a change of direction or a stop, else at the last
    segment's speed."""
    d, seg = first_run(v)
    B = v.shape[0]
    dev, dt_ = v.device, v.dtype
    seg_len = (v.abs() * TW_DT) * seg                                           # (B, K) metres of each run segment
    cum = torch.cumsum(seg_len, 1)
    total = cum[:, -1]
    starts = cum - seg_len
    sg = torch.arange(n, device=dev, dtype=dt_)[None, :] * spacing                   # (1, n) arc positions
    # the segment each arc position falls in (the last run segment past the end)
    kidx = (sg[:, :, None] >= starts[:, None, :]).sum(-1) - 1                       # (B, n)
    nseg = seg.sum(1).clamp_min(1)
    kidx = torch.minimum(kidx.clamp_min(0), (nseg - 1)[:, None])
    # integrate the run's arcs exactly at the waypoints: pose at the start of segment k, then the arc into it
    x0 = torch.zeros(B, TW_K, device=dev, dtype=dt_)
    y0, p0 = torch.zeros_like(x0), torch.zeros_like(x0)
    x = torch.zeros(B, device=dev, dtype=dt_)
    y, psi = torch.zeros_like(x), torch.zeros_like(x)
    dd = d.clamp(min=0) * 2 - 1                                                       # +1 forward / stop, -1 backward
    dd = torch.where(d == 0, torch.ones_like(dd), dd).to(dt_)
    for k in range(TW_K):
        x0[:, k], y0[:, k], p0[:, k] = x, y, psi
        x, y, psi = _arc(x, y, psi, kappa[:, k], dd * seg_len[:, k])
    kk = kappa.gather(1, kidx)
    within = (sg - starts.gather(1, kidx)).clamp_min(0.0)
    over = (sg - total[:, None]).clamp_min(0.0)
    within = torch.minimum(within, seg_len.gather(1, kidx))
    px, py, pp = _arc(x0.gather(1, kidx), y0.gather(1, kidx), p0.gather(1, kidx), kk, dd[:, None] * within)
    # straight on past the end, the way the run was going
    px = px + dd[:, None] * over * torch.cos(pp)
    py = py + dd[:, None] * over * torch.sin(pp)
    speed = v.abs().gather(1, kidx)
    flips = (~seg).any(1)                                                             # the run ends in a change of direction
    last_v = (v.abs() * seg).gather(1, (nseg - 1)[:, None])[:, 0]
    tail_v = torch.where(flips, torch.zeros_like(last_v), last_v)
    speed = torch.where(sg > total[:, None], tail_v[:, None], speed)
    kap = torch.where(sg > total[:, None], torch.zeros_like(kk), kk)
    return torch.stack([px, py], -1), speed, kap, d, total


def pursue_back(path_xy: torch.Tensor, total: torch.Tensor, wheelbase: float, lookahead: torch.Tensor,
                s_max: float, spacing: float) -> torch.Tensor:
    """Steering [rad] (B,) of a pure pursuit driving backwards to the point `lookahead` metres along the backward run
    (its end if it is shorter): the arc through the car's rear-facing axis and that point has curvature 2 y / d^2 --
    the same formula forwards and backwards, since a car backing with steering curvature kappa traces the circle of
    radius 1 / kappa tangent to its heading -- and the steering is atan(kappa x wheelbase)."""
    n = path_xy.shape[1]
    la = torch.minimum(lookahead, total.clamp_min(spacing))
    j = (la / spacing).round().long().clamp(1, n - 1)
    tgt = path_xy.gather(1, j[:, None, None].expand(-1, 1, 2))[:, 0]
    d2 = (tgt ** 2).sum(1).clamp_min(1e-6)
    kappa = 2.0 * tgt[:, 1] / d2
    return torch.atan(kappa * wheelbase).clamp(-s_max, s_max)
