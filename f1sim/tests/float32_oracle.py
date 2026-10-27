"""What ANY correct float32 evaluation of the LiDAR's ray tests may return -- the portable test oracle.

The Triton kernels (`lidar_triton.car_mesh_hits_triton`, `prop_triton.prism_hits_triton`) repeat
the eager torch functions operation for operation, and on the GPUs measured so far (`BIT_EXACT`)
they return the same bits. That is checked where it was measured. What must hold on ANY GPU and
toolchain is weaker and principled, and that is what this module decides:

Every formula is evaluated in float64 from the same float32 inputs (and torch's own float32 cos/sin
of the yaws, which both implementations share), carrying a bound on how far a float32 evaluation of
the same expression can be from it. Each operation of that evaluation rounds at most once (relative
2^-24) -- or not at all, as when a compiler fuses a multiply and an add into an FMA -- so the bound
propagates exactly as written in `Ev`: products |x|e_y + |y|e_x + e_x e_y, sums e_x + e_y,
quotients (e_x + |q| e_y) / (|y| - e_y), then one rounding of the result. A decision (`u >= 0`,
`t > 0`, `den < 0`, the level's rounding, the `|den| < 1e-9` clamp) is *robust* when the whole
interval lies on one side of it, and *ambiguous* otherwise. A beam then has

* `must`: some hit is robust -- every correct evaluation reports a hit, with a range at most the
  robust hit's upper bound and at least the smallest possible hit's lower bound;
* not `may`: no hit is possible -- every correct evaluation reports none;
* otherwise it is ambiguous: it passes within the rounding bound of an outline endpoint or face
  (u or t near a boundary), starts on an outline (t near 0), sits on a level boundary, or runs
  parallel to a segment or face -- and a hit, if reported, is at least the smallest possible one.

Two implementations that both satisfy the band can differ only where the geometry is ambiguous, and
there only by what the rounding allows. The reference (torch eager) is checked against the band as
well as the kernel, so a band that were too tight would fail on the reference first.
"""
from __future__ import annotations

import numpy as np
import torch

EPS = 2.0 ** -24                    # float32 unit roundoff
REL64 = 1e-14                       # the float64 evaluation's own rounding, generously
EPS9 = float(np.float32(1e-9))      # the `|den| < 1e-9` threshold as both implementations compare it
INF = float("inf")

#: (major, minor) compute capabilities on which the kernels were measured bit-identical to the eager
#: reference: RTX 4070 SUPER (sm_89) and RTX 5060 Laptop (sm_120), 2026-10-27.
BIT_EXACT = {(8, 9), (12, 0)}


def bit_exact_here(device="cuda") -> bool:
    return torch.cuda.is_available() and tuple(torch.cuda.get_device_capability(device)) in BIT_EXACT


class Ev:
    """A float64 value `v` and a bound `e` on |float32 evaluation - v|."""
    __slots__ = ("v", "e")

    def __init__(self, v, e):
        self.v, self.e = v, e

    @staticmethod
    def of(t):
        """An exact float32 input."""
        v = t.double()
        return Ev(v, torch.zeros_like(v))

    def __getitem__(self, idx):
        return Ev(self.v[idx], self.e[idx])

    def _r(self, v, e):                                  # one float32 rounding of the result
        return Ev(v, e * (1.0 + EPS) + (EPS + REL64) * v.abs())

    def __add__(a, b):
        b = _ev(b, a)
        return a._r(a.v + b.v, a.e + b.e)

    __radd__ = __add__

    def __sub__(a, b):
        b = _ev(b, a)
        return a._r(a.v - b.v, a.e + b.e)

    def __rsub__(a, b):
        return _ev(b, a) - a

    def __mul__(a, b):
        b = _ev(b, a)
        return a._r(a.v * b.v, a.v.abs() * b.e + b.v.abs() * a.e + a.e * b.e)

    __rmul__ = __mul__

    def __neg__(a):
        return Ev(-a.v, a.e)

    def __truediv__(a, b):
        b = _ev(b, a)
        v = a.v / b.v
        room = b.v.abs() - b.e
        e = torch.where(room > 0, (a.e + v.abs() * b.e) / room.clamp_min(1e-300), torch.full_like(v, INF))
        return a._r(v, e)

    def sqrt(a):
        v = a.v.clamp_min(0.0).sqrt()
        lo = (a.v - a.e).clamp_min(0.0).sqrt()
        return a._r(v, a.e / (lo + v).clamp_min(1e-300))


def _ev(b, like):
    if isinstance(b, Ev):
        return b
    v = torch.full_like(like.v, float(np.float32(b)))   # a Python constant enters float32 math rounded
    return Ev(v, torch.zeros_like(v))


def _range_band(t_rob, t_lo, slen):
    """t bounds -> range bounds: r = fl(t * fl(slen))."""
    r_hi = t_rob * (slen.v + slen.e) * (1.0 + EPS) * (1.0 + 1e-12)
    r_lo = t_lo * (slen.v - slen.e).clamp_min(0.0) * (1.0 - EPS) * (1.0 - 1e-12)
    return r_lo, r_hi


def violations(r, h, band):
    """(M,) bool: where an implementation's (range, hit) leaves the band."""
    must, may, r_lo, r_hi = band
    r = r.double()
    in_band = h & (r >= r_lo) & (r <= r_hi)
    return (must & ~in_band) | (~may & h) | (may & ~must & h & ~(r >= r_lo))


# ---------------------------------------------------------------------------------- car outlines
def mesh_band(origin, dh, k, poses, scale, chunk: int = 512):
    """`lidar._ray_slices_hits_gathered` for M beams (flattened: origin (M,3), dh (M,2), k (M,),
    poses (M,C,3), scale (M,C), float32) -> (must, may, r_lo, r_hi), each (M,)."""
    from f1sim.lidar import car_slices
    z_cpu, segs_c, valid_c = car_slices("cpu")
    dev = k.device
    segs, valid = segs_c.to(dev).double(), valid_c.to(dev)
    L = valid.shape[0]
    z0 = float(z_cpu[0])
    inv_dz = float(torch.ones((), dtype=torch.float32) / (z_cpu[1] - z_cpu[0]))
    cy32, sy32 = torch.cos(poses[..., 2]), torch.sin(poses[..., 2])          # what both implementations use
    M, C = poses.shape[:2]
    outs = []
    for m0 in range(0, M, chunk):
        s = slice(m0, m0 + chunk)
        ox, oy, oz = Ev.of(origin[s, 0]), Ev.of(origin[s, 1]), Ev.of(origin[s, 2])
        hx, hy, kk = Ev.of(dh[s, 0]), Ev.of(dh[s, 1]), Ev.of(k[s])
        slen = (1.0 + kk * kk).sqrt()
        m = kk.v.shape[0]
        t_rob = torch.full((m,), INF, dtype=torch.float64, device=dev)
        t_lo = torch.full((m,), INF, dtype=torch.float64, device=dev)
        for c in range(C):
            px, py = Ev.of(poses[s, c, 0]), Ev.of(poses[s, c, 1])
            cy, sy, sc = Ev.of(cy32[s, c]), Ev.of(sy32[s, c]), Ev.of(scale[s, c])
            rx, ry = ox - px, oy - py
            lx, ly = rx * cy + ry * sy, (-rx) * sy + ry * cy
            dx, dy = hx * cy + hy * sy, (-hx) * sy + hy * cy
            tcr = (-lx) * dx - ly * dy
            tc = Ev(tcr.v.clamp_min(0.0), tcr.e)                                  # |max(a,0) - max(b,0)| <= |a - b|
            lam = ((oz + kk * tc) - z0) * inv_dz
            la, lb = torch.floor(lam.v - lam.e + 0.5), torch.floor(lam.v + lam.e + 0.5)
            amb = la != lb                                                       # the rounding of the level is in doubt
            col = lambda x: Ev(x.v[:, None], x.e[:, None])                       # noqa: E731
            dx1, dy1, lx1, ly1, sc1 = col(dx), col(dy), col(lx), col(ly), col(sc)
            for lv, use in ((la, torch.ones_like(amb)), (lb, amb)):
                inb = (lv >= 0) & (lv < L) & use
                safe = lv.clamp(0, L - 1).long()
                sg, vd = segs[safe], valid[safe] & inb[:, None]
                ax, ay = Ev.of(sg[..., 0]) * sc1, Ev.of(sg[..., 1]) * sc1
                bx, by = Ev.of(sg[..., 2]) * sc1, Ev.of(sg[..., 3]) * sc1
                ex, ey = bx - ax, by - ay
                den = dx1 * ey - dy1 * ex
                wild = (den.v.abs() - den.e) < EPS9                              # the clamp may apply: t, u unbounded
                wx, wy = ax - lx1, ay - ly1
                t = (wx * ey - wy * ex) / den
                u = (wx * dy1 - wy * dx1) / den
                rob = vd & ~amb[:, None] & ~wild & (u.v - u.e > 0) & (u.v + u.e < 1) & (t.v - t.e > 0)
                pos = vd & (wild | ((u.v + u.e >= 0) & (u.v - u.e <= 1) & (t.v + t.e > 0)))
                t_rob = torch.minimum(t_rob, torch.where(rob, t.v + t.e, INF).amin(-1))
                lo = torch.where(wild, 0.0, (t.v - t.e).clamp_min(0.0))
                t_lo = torch.minimum(t_lo, torch.where(pos, lo, INF).amin(-1))
        r_lo, r_hi = _range_band(t_rob, t_lo, slen)
        outs.append((t_rob < INF, t_lo < INF, r_lo, r_hi))
    return tuple(torch.cat(x) for x in zip(*outs))


# ---------------------------------------------------------------------------------- prop prisms
def prism_band(origin, dh, k, poses, pn, pd, z_lo, z_hi, chunk: int = 4096):
    """`prop_math.ray_prisms_hits(..., inside="hit")` for M beams (flattened: origin (M,3), dh (M,2),
    k (M,), poses (M,C,3), pn (M,C,K,2), pd (M,C,K), z_lo, z_hi (M,C)) -> (must, may, r_lo, r_hi)."""
    dev = k.device
    cy32, sy32 = torch.cos(poses[..., 2]), torch.sin(poses[..., 2])
    live = ((pn.pow(2).sum(-1) > 0.5).sum(-1) >= 3) & (z_hi > z_lo)               # `_slot_is_live`, exactly
    pad = (pn[..., 0] == 0) & (pn[..., 1] == 0) & (pd == INF)
    M, C, K = pn.shape[:3]
    outs = []
    for m0 in range(0, M, chunk):
        s = slice(m0, m0 + chunk)
        ox, oy, oz = Ev.of(origin[s, 0]), Ev.of(origin[s, 1]), Ev.of(origin[s, 2])
        hx, hy, kk = Ev.of(dh[s, 0]), Ev.of(dh[s, 1]), Ev.of(k[s])
        k32 = k[s]
        slen = (1.0 + kk * kk).sqrt()
        m = k32.shape[0]
        full = lambda x: torch.full((m,), x, dtype=torch.float64, device=dev)   # noqa: E731
        t_rob, t_lo = full(INF), full(INF)
        for c in range(C):
            px, py = Ev.of(poses[s, c, 0]), Ev.of(poses[s, c, 1])
            cy, sy = Ev.of(cy32[s, c]), Ev.of(sy32[s, c])
            rx, ry = ox - px, oy - py
            lx, ly = rx * cy + ry * sy, (-rx) * sy + ry * cy
            dx, dy = hx * cy + hy * sy, (-hx) * sy + hy * cy
            in_lo, in_hi, out_lo, out_hi = full(-INF), full(-INF), full(INF), full(INF)
            wild = torch.zeros(m, dtype=torch.bool, device=dev)
            infeasible = torch.zeros(m, dtype=torch.bool, device=dev)

            def fold(den, num, par, ent, lea, real):
                nonlocal in_lo, in_hi, out_lo, out_hi, wild, infeasible
                t = num / den
                in_lo = torch.where(real & ent, torch.maximum(in_lo, t.v - t.e), in_lo)
                in_hi = torch.where(real & ent, torch.maximum(in_hi, t.v + t.e), in_hi)
                out_lo = torch.where(real & lea, torch.minimum(out_lo, t.v - t.e), out_lo)
                out_hi = torch.where(real & lea, torch.minimum(out_hi, t.v + t.e), out_hi)
                viol, sat = par & (num.v + num.e < 0), par & (num.v - num.e >= 0)
                infeasible = infeasible | (real & viol)
                wild = wild | (real & par & ~viol & ~sat)

            for j in range(K):
                real = ~pad[s, c, j]                                           # padding is exactly neutral
                nx, ny = Ev.of(pn[s, c, j, 0]), Ev.of(pn[s, c, j, 1])
                dd = Ev.of(torch.where(real, pd[s, c, j], torch.zeros_like(pd[s, c, j])))
                den = dx * nx + dy * ny
                num = dd - (lx * nx + ly * ny)
                par = (den.v.abs() + den.e) < EPS9                             # parallel in every evaluation
                nonpar = (den.v.abs() - den.e) >= EPS9                         # never parallel, sign fixed
                wild = wild | (real & ~par & ~nonpar)
                fold(den, num, par, nonpar & (den.v < 0), nonpar & (den.v > 0), real)
            par_z = k32.abs() < EPS9                                           # exact: k is an input
            for den, num in ((kk, Ev.of(z_hi[s, c]) - oz), (-kk, oz - Ev.of(z_lo[s, c]))):
                fold(den, num, par_z, ~par_z & (den.v < 0), ~par_z & (den.v > 0), torch.ones_like(par_z))
            th_lo, th_hi = in_lo.clamp_min(0.0), in_hi.clamp_min(0.0)
            lv = live[s, c]
            rob = lv & ~wild & ~infeasible & (out_lo >= th_hi) & (out_lo > 0)
            pos = lv & (wild | (~infeasible & (out_hi >= th_lo) & (out_hi > 0)))
            t_rob = torch.minimum(t_rob, torch.where(rob, th_hi, INF))
            t_lo = torch.minimum(t_lo, torch.where(pos, torch.where(wild, 0.0, th_lo), INF))
        r_lo, r_hi = _range_band(t_rob, t_lo, slen)
        outs.append((t_rob < INF, t_lo < INF, r_lo, r_hi))
    return tuple(torch.cat(x) for x in zip(*outs))


def flatten_beams(sel, origin, dh, k, *per_env):
    """The beams `sel` (B,N) bool of a batch, as flat (M, ...) inputs with their env's per-env tensors."""
    b, n = sel.nonzero(as_tuple=True)
    return (origin[b, n], dh[b, n], k[b, n]) + tuple(t[b] for t in per_env)
