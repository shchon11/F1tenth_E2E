"""Direct, differentiable hindsight costs on the WHOLE plan (11-25, plan_rl --plan-grad-cost; the user: "그려").

Why. With --tail-consistency only the plan's first 0.1 s is PPO's action (noise, a log-probability); the other 18 outputs get
no PPO gradient, they only copy where the car then went. The plan-car cost (`plan_car_cost`) charges a plan's time-aligned
overlap with the other car's real future -- mostly at its 0.2-1.0 s points, which PPO cannot move, it is ~1 % of the lap
reward, and the decisions it fires on are 1-2 % of all. The oracle gate (orc34a: near-true clearances as an input, 75
updates) left the 0.6x overlap at 12.7 -> 12.2 % and the trunk's weights on those inputs at a fifteenth of the others'.
So here the cost is not a reward: the plan the policy's MEAN outputs make (all 20 outputs, decoded exactly as the env decodes
them) is priced against what the rollout recorded, and the price's gradient goes straight into the outputs and the trunk
(an analytic policy gradient on the plan, as in SHAC-style training -- possible because the plan is executed exactly and
the costs are geometric).

The three terms, each at the plan's points t_k = 0.1 k s (k = 1..10), each point weighted w_k (1.0 at 0.1 s down to the
plan-car cost's last weight at 1.0 s), all in metres of soft penetration ``pen(d) = s softplus((m - d) / s)`` (0 where
d > m + CUT s, so 'non-zero' means 'within CUT soft widths of the limit'):

  (a) car   our car's 3-disc cover (the plan pose: its point, its heading channel) against the other car's 3 discs at its
            REAL pose at t + t_k (the rollout's record), every disc pair, limit 2 r - margin (the plan-car cost's
            overlap). Only where the other car was knowable at the decision (within --pred-range, a beam on it in the last
            1 s), only up to a restart of either car or the rollout's end (a prefix of the points), and only where the env's
            fault rule puts a contact on us -- read at the point before (our plan pose there against the other car then,
            the car itself for the first point), as `plan_car_cost` reads it.
  (b) walls the signed distance to the wall surface (`WallField`: the reward's track EDT, made signed, bilinear) at the
            footprint's 4 corners and 4 edge midpoints, limit = a small clearance.
  (c) boxes the boxes the car has seen this episode (in view of the LiDAR now or before, or passed), the nearest few
            within reach: the box's own plane distance (`ProceduralObstacles.clearance`'s measure) at the footprint's 8
            points, and our footprint's plane distance at the box's corners (a box narrower than the gap between two
            footprint points still counts), limit = a small clearance.

Everything here is plain torch on the caller's device; nothing reads the env except `WallField` (built once from the
track tensors) and the caller's stored rollout."""
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .. import plan_steps as PS
from . import plan_car_cost as PCC

CUT = 4.0          # a soft penetration is 0 beyond this many soft widths past its limit


def pen(d: torch.Tensor, m: float, s: float, cut: float = CUT) -> torch.Tensor:
    """soft penetration [m] of a clearance `d` [m] below the limit `m`: s softplus((m - d) / s), 0 where d > m + cut s"""
    z = (m - d) / s
    return torch.where(z > -cut, s * F.softplus(z), torch.zeros_like(d))


def decode(a_raw: torch.Tensor, v0: torch.Tensor, d0: torch.Tensor, st0: torch.Tensor, lim: PS.Limits,
           cap: Optional[torch.Tensor] = None):
    """the env's decode of a plan, differentiable: pre-tanh outputs (n, 20), the plan's start speed v0 (n,) and steering d0
    (n,) (the speed estimate as clamped at the rollout, the last command), the car's state at the decision st0 (n, >= 3) ->
    the 10 points in the world: x, y, heading (n, 10) each. `plan_v2` + `plan_world` exactly (`plan_steps.roll` has no
    in-place step and no no-grad block)"""
    from .plan_policy import plan_world, PT_IDX
    p = PS.roll(torch.tanh(a_raw), v0, d0, lim, cap)
    x, y, psi = plan_world(p, st0)
    ii = PT_IDX.to(a_raw.device)
    return x[:, ii], y[:, ii], psi[:, ii]


def outline8(length: float, width: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """(8, 2) the footprint's corners and edge midpoints about the CoG (the simulator's footprint is symmetric about it)"""
    a, b = 0.5 * length, 0.5 * width
    return torch.tensor([[a, b], [a, 0.0], [a, -b], [0.0, -b], [-a, -b], [-a, 0.0], [-a, b], [0.0, b]], device=device, dtype=dtype)


def place(x: torch.Tensor, y: torch.Tensor, psi: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """poses (...) and body-frame points P (m, 2) -> (..., m, 2) in the world"""
    c, s = torch.cos(psi)[..., None], torch.sin(psi)[..., None]
    return torch.stack([x[..., None] + P[:, 0] * c - P[:, 1] * s, y[..., None] + P[:, 0] * s + P[:, 1] * c], -1)


# ------------------------------------------------------------------------------------------------------------------ (a) car
def fault_gate(me_xy0: torch.Tensor, me_psi0: torch.Tensor, me_rem0: torch.Tensor, pl_xy: torch.Tensor, pl_psi: torch.Tensor,
               pl_rem: torch.Tensor, op_xy_prev: torch.Tensor, op_rem_prev: torch.Tensor, lap: torch.Tensor,
               fault_gap: float = PCC.FAULT_GAP) -> torch.Tensor:
    """(n, K) bool: a contact at plan point k would be ours by the env's rule, read at the point before it -- our pose there
    (the car at the decision for k = 1: me_xy0 (n, 2), me_psi0 / me_rem0 (n,); else the plan's point k - 1: pl_xy (n, K, 2),
    pl_psi / pl_rem (n, K)) against the other car then (op_xy_prev (n, K, 2), op_rem_prev (n, K): at t + t_{k-1}); `lap`
    (n,) the lap's length. `plan_car_cost`'s rule: the other car ahead along the track, or less than a car length behind"""
    pxy = torch.cat([me_xy0[:, None], pl_xy[:, :-1]], 1)
    ppsi = torch.cat([me_psi0[:, None], pl_psi[:, :-1]], 1)
    prem = torch.cat([me_rem0[:, None], pl_rem[:, :-1]], 1)
    g = PCC.along_gap(torch, pxy, ppsi, prem, op_xy_prev, op_rem_prev, lap[:, None])
    return g > fault_gap


def car_term(px: torch.Tensor, py: torch.Tensor, ppsi: torch.Tensor, op_xy: torch.Tensor, op_psi: torch.Tensor,
             mask: torch.Tensor, w: torch.Tensor, length: float, width: float, margin: float, s: float):
    """(a): the plan's points (n, K) x3 against the other car's real poses at the same moments op_xy (n, K, 2), op_psi (n, K);
    `mask` (n, K) the points that count (knowable, compared, at fault); `w` (K,) the points' weights -> (cost (n,) [m],
    any pair within the soft band (n,) bool, any real overlap -- a pair closer than the limit (n,) bool)"""
    me = PCC._discs(torch, torch.stack([px, py], -1), ppsi, length)                     # (n, K, 3, 2)
    ot = PCC._discs(torch, op_xy, op_psi, length)
    dv = me[:, :, :, None, :] - ot[:, :, None, :, :]                                     # (n, K, 3, 3, 2)
    d = torch.sqrt((dv * dv).sum(-1) + 1e-12)
    thr = max(2.0 * PCC.disc_radius(length, width) - margin, 0.0)
    pk = pen(d, thr, s).sum((-1, -2))                                                   # (n, K)
    mf = mask.to(pk.dtype)
    cost = (pk * mf * w).sum(-1)
    near = ((pk > 0) & mask).any(-1)
    over = ((d < thr).flatten(2).any(-1) & mask).any(-1)
    return cost, near, over


# ---------------------------------------------------------------------------------------------------------------- (b) walls
class WallField:
    """The reward's track EDT (`TrackTensors.edt`: the distance from a free cell's centre to the nearest wall cell's centre,
    0 in a wall) made signed, laid out like it: free cells edt - res/2, wall cells -(the distance to the nearest free cell's
    centre - res/2), so the bilinear field crosses 0 on the cell boundary between a free and a wall cell -- where
    `sample_edt` <= 0 starts. Off the grid: -res/2 (a wall, as `sample_edt` has it). Built once per distinct grid on the CPU
    (one grid at a time), kept in the EDT's own dtype on its device."""

    def __init__(self, tt):
        import numpy as np
        from scipy import ndimage
        self.tt = tt
        self.field = torch.empty_like(tt.edt)
        done = set()
        for k, t in enumerate(tt.tracks):
            o = int(tt.t_off[k])
            if o in done:
                continue
            done.add(o)
            res = float(t.resolution)
            edt = np.asarray(t.edt, np.float32)
            wall = edt <= 0.0
            d_in = ndimage.distance_transform_edt(wall).astype(np.float32) * res
            sd = np.where(wall, -(d_in - 0.5 * res), edt - 0.5 * res).astype(np.float32)
            self.field[o:o + sd.size] = torch.from_numpy(sd.reshape(-1)).to(self.field)
            del d_in, sd, wall
        self.n_grids = len(done)

    def sample(self, xy: torch.Tensor, tid: torch.Tensor) -> torch.Tensor:
        """(..., 2) points, (...) track ids -> (...) [m] signed distance to the wall surface (bilinear, differentiable in xy)"""
        tt = self.tt
        tid = torch.broadcast_to(tid, xy.shape[:-1])
        res = tt.t_res[tid]; ox = tt.t_origin[tid, 0]; oy = tt.t_origin[tid, 1]
        H = tt.t_H[tid].long(); W = tt.t_W[tid].long(); off = tt.t_off[tid]
        c = (xy[..., 0] - ox) / res
        r = (xy[..., 1] - oy) / res
        j0 = torch.floor(c.detach()); i0 = torch.floor(r.detach())
        wc = c - j0; wr = r - i0
        j0 = j0.long(); i0 = i0.long()

        def g(i, j):
            inside = (j >= 0) & (j < W) & (i >= 0) & (i < H)
            idx = off + i.clamp(min=0).minimum(H - 1) * W + j.clamp(min=0).minimum(W - 1)
            v = self.field[idx].float()
            return torch.where(inside, v, -0.5 * res)
        return (g(i0, j0) * (1 - wc) * (1 - wr) + g(i0, j0 + 1) * wc * (1 - wr)
                + g(i0 + 1, j0) * (1 - wc) * wr + g(i0 + 1, j0 + 1) * wc * wr)


def wall_term(px: torch.Tensor, py: torch.Tensor, ppsi: torch.Tensor, tid: torch.Tensor, field, P8: torch.Tensor,
              w: torch.Tensor, m: float, s: float):
    """(b): the plan's points (n, K) x3 on track `tid` (n,), the footprint's points P8 (8, 2) -> (cost (n,) [m], any point
    within the soft band (n,), any point inside the limit (n,))"""
    q = place(px, py, ppsi, P8)                                                          # (n, K, 8, 2)
    sd = field.sample(q, tid[:, None, None])
    pk = pen(sd, m, s).sum(-1)                                                           # (n, K)
    return (pk * w).sum(-1), (pk > 0).any(-1), (sd < m).flatten(1).any(-1)


# ---------------------------------------------------------------------------------------------------------------- (c) boxes
def box_term(px: torch.Tensor, py: torch.Tensor, ppsi: torch.Tensor, bn: torch.Tensor, bd: torch.Tensor, bvalid: torch.Tensor,
             bv: torch.Tensor, bmask: torch.Tensor, P8: torch.Tensor, w: torch.Tensor, length: float, width: float, m: float,
             s: float):
    """(c): the plan's points (n, K) x3 against up to KB boxes a decision: their faces in the world, outward normals bn
    (n, KB, F, 2), offsets bd (n, KB, F) (inside: n . x < d on every face), bvalid (n, KB, F) the real faces, their corners
    bv (n, KB, F, 2) (padding repeats a real one), bmask (n, KB) the boxes that count. Clearance of a footprint point to a
    box = its largest plane distance over the box's faces (`ProceduralObstacles.clearance`); of a box corner to our
    footprint = max(|x_body| - L/2, |y_body| - W/2) -> (cost (n,) [m], within the soft band (n,), inside the limit (n,))"""
    n, K = px.shape
    if bn.shape[1] == 0 or not bool(bmask.any()):
        z = px.sum(-1) * 0.0
        return z, torch.zeros(n, dtype=torch.bool, device=px.device), torch.zeros(n, dtype=torch.bool, device=px.device)
    q = place(px, py, ppsi, P8)                                                          # (n, K, 8, 2)
    # our 8 points against each box's faces: (n, K, 8, KB, F)
    pd_ = (q[:, :, :, None, None, :] * bn[:, None, None]).sum(-1) - bd[:, None, None]
    pd_ = torch.where(bvalid[:, None, None], pd_, torch.full_like(pd_, -1e9))
    sd_b = pd_.amax(-1)                                                                  # (n, K, 8, KB)
    # each box's corners in our body frame at every point: (n, K, KB, F)
    c, s_ = torch.cos(ppsi)[:, :, None, None], torch.sin(ppsi)[:, :, None, None]
    dx = bv[:, None, :, :, 0] - px[:, :, None, None]; dy = bv[:, None, :, :, 1] - py[:, :, None, None]
    lx = dx * c + dy * s_; ly = -dx * s_ + dy * c
    sd_c = torch.maximum(lx.abs() - 0.5 * length, ly.abs() - 0.5 * width)
    bm1 = bmask[:, None, None, :]                                                        # (n, 1, 1, KB)
    p1 = pen(sd_b, m, s) * bm1
    p2 = pen(sd_c, m, s) * (bmask[:, None, :, None] & bvalid[:, None, :, :])             # padded corners repeat a real one
    pk = p1.sum((-1, -2)) + p2.sum((-1, -2))                                             # (n, K)
    inside = (((sd_b < m) & bm1).flatten(1).any(-1)
              | ((sd_c < m) & bmask[:, None, :, None] & bvalid[:, None, :, :]).flatten(1).any(-1))
    return (pk * w).sum(-1), (pk > 0).any(-1), inside


def box_record(po, lid: torch.Tensor, seen: torch.Tensor, car_xy: torch.Tensor, kb: int, reach: float):
    """at a decision: the `kb` nearest boxes this car has seen (`seen` (B_l, C) bool, live slots only), centres within `reach`
    [m] of the car (car_xy (B_l, 2)), as `box_term` reads them: (bn (B_l, kb, F, 2), bd (B_l, kb, F), bvalid (B_l, kb, F),
    bv (B_l, kb, F, 2), bmask (B_l, kb))"""
    from ..prop_math import _to_world, _polygon_vertices
    pos = po.p_poses[lid].float()                                                        # (B_l, C, 3)
    Bl, C = pos.shape[:2]
    live = po.p_sid[lid] >= 0
    dd = (pos[..., :2] - car_xy[:, None, :]).norm(dim=-1)
    dd = torch.where(seen & live & (dd <= reach), dd, torch.full_like(dd, float("inf")))
    k = min(int(kb), C)
    dk, idx = torch.topk(dd, k, dim=1, largest=False)                                    # (B_l, k)
    g = lambda a: torch.gather(a, 1, idx.reshape(idx.shape + (1,) * (a.dim() - 2)).expand(-1, -1, *a.shape[2:]))
    pn, pdd, ps = g(po.p_n[lid].float()), g(po.p_d[lid].float()), g(pos)               # (B_l, k, F, 2), (B_l, k, F), (B_l, k, 3)
    F_ = pn.shape[2]
    valid = pn.pow(2).sum(-1) > 0.5
    nw, dw = _to_world(pn.reshape(Bl * k, F_, 2), torch.where(valid, pdd, torch.zeros_like(pdd)).reshape(Bl * k, F_),
                       ps.reshape(Bl * k, 3))
    vv = valid.reshape(Bl * k, F_)
    ver = _polygon_vertices(nw, dw, vv)
    bmask = torch.isfinite(dk) & (valid.sum(-1) >= 3)
    return (nw.view(Bl, k, F_, 2), dw.view(Bl, k, F_), vv.view(Bl, k, F_), ver.view(Bl, k, F_, 2), bmask)


def later_poses(op_xy: torch.Tensor, op_psi: torch.Tensor, op_rem: torch.Tensor, op_ok: torch.Tensor, brk: torch.Tensor,
                know: torch.Tensor, kofs: Sequence[int]):
    """the other car along a rollout, for every decision t and plan point k: its pose at t + t_k and at t + t_{k-1} (t itself
    for the first), its distance to go at t + t_{k-1}, and whether point k is compared (a prefix: inside the rollout, no
    restart / jump of either car in between, the other car there, knowable at t). op_* (T+1, B[, 2]) at each decision's
    start and after the last, brk / know (T, B) -> dict of (T, B, K[, 2])"""
    T = op_psi.shape[0] - 1
    zf = op_psi[:T, :, None].expand(-1, -1, len(kofs)) * 0.0
    (fx, fy, fpsi, frem), ok = PCC._later(torch, zf, [op_xy[..., 0], op_xy[..., 1], op_psi, op_rem], op_ok, brk, kofs)
    ok = (torch.cumsum((~ok).long(), -1) == 0) & know[..., None]
    px_ = PCC._shift(fx, op_xy[:T][..., 0]); py_ = PCC._shift(fy, op_xy[:T][..., 1])
    prem = PCC._shift(frem, op_rem[:T])
    return {"xy": torch.stack([fx, fy], -1), "psi": fpsi, "xy_prev": torch.stack([px_, py_], -1), "rem_prev": prem, "ok": ok}


def gae_zero_baseline(r: torch.Tensor, done: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """(T, B) rewards and restarts -> the GAE of these rewards alone with a zero value everywhere (delta = r): the part of
    the advantages this reward component makes, as if the critic knew nothing of it -- an approximation (the critic's value
    of the component is ignored)"""
    adv = torch.zeros_like(r)
    g = torch.zeros_like(r[0])
    for t in reversed(range(r.shape[0])):
        g = r[t] + gamma * lam * (1.0 - done[t]) * g
        adv[t] = g
    return adv


def flat_norm(gs, params) -> torch.Tensor:
    """(gradients, parameters) -> the flat gradient vector (zeros for unused)"""
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])
