"""Ray and contact maths for finite convex prisms, in the style of `lidar.py::ray_box_hits`.

Two runtime functions, plus the build-time helper that feeds them:

* `section_halfplanes` -- numpy, build time. A convex CCW polygon becomes outward unit normals and
  offsets, padded to a fixed `k_pad`.
* `ray_prisms_hits` -- torch. Beams against finite convex prisms.
* `prism_contacts` -- torch. The car rectangle against those prisms, by SAT.

Everything at runtime is pure torch with no data-dependent Python branching: the only Python loop
is over `C`, which is a static shape. `torch.where` is used freely -- it is a tensor op, not a
branch. That is the precondition for CUDA-graph capture, and a test walks the AST to keep it true.

Capture and CPU/CUDA agreement are **measured**, not inferred, in `gpu_verify.py` ->
`gpu_verify.json`: on an RTX 4060 Ti, torch 2.10, at B=8 x N=1080 beams with C=4 props, K=24, both
functions capture into a `torch.cuda.CUDAGraph` and replay correctly in float32 and float64, hit
masks and contact indices identical to CPU, and replays follow changed inputs rather than repeating
a stale answer. Float32 range agreed with CPU to 1.2e-6 m; float64 to 8.9e-16 m.

Padding, and how nothing here produces a NaN
--------------------------------------------
A padded half-plane is `n = (0, 0)`, `d = +inf`, so it constrains nothing (`0 * t <= inf`). The
arithmetic is arranged so that value never turns into a NaN:

* `num = d - o.n` is `inf - 0 = inf`, never `inf - inf`, because a padded normal is exactly zero.
* `den = d_ray . n` is exactly `0`, so the plane takes the parallel branch, where `num = inf >= 0`
  means "no constraint" and the division result is never selected.
* Vertex reconstruction never divides by a padded plane at all -- padded slots are replaced with a
  finite stand-in *before* the solve, then discarded.

`test_prop_math.py` asserts the absence of NaN on padded inputs rather than trusting this prose.

Disabling a prop slot
---------------------
Two ways, both explicit and both tested, because a per-env prop list is ragged and the padding
entries must not become invisible phantom obstacles:

1. `z_hi <= z_lo` -- the contract's convention.
2. Fewer than three real half-planes -- an all-padding slot.

Either one makes the slot inert for both functions. It is checked as a stated condition rather than
left to fall out of the slab arithmetic.

Beam origin inside a prop
-------------------------
`ray_box_hits` requires `t_in > 0`, so a ray starting inside a box reports *no obstacle*. For a
finite prop that is the more dangerous lie of the two available, so the default here is
`inside="hit"`: the beam returns range 0. Pass `inside="miss"` to match `ray_box_hits` exactly.
The choice is a Python keyword, resolved before any tensor work, and both settings are tested.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

EPS = 1e-9

#: The smallest number of real half-planes a slot needs before it is treated as a prop at all.
MIN_FACES = 3


# ==================================================================== build time (numpy)
def section_halfplanes(poly: np.ndarray, k_pad: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convex CCW polygon (K,2) -> outward unit normals (k_pad,2) and offsets (k_pad,).

    Interior is `{p : n_j . p <= d_j for all j}`. Padding slots are `n = (0,0)`, `d = +inf`.

    Two ordering guarantees that `prism_contacts` depends on, so do not reorder the output:

    * normals come in polygon edge order, so half-planes `j` and `j+1` meet at a real vertex;
    * padding is contiguous at the end, so the first invalid slot after `j` marks the wrap.
    """
    poly = np.asarray(poly, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError(f"polygon must be (K>=3, 2), got {poly.shape}")
    if k_pad < len(poly):
        raise ValueError(f"k_pad {k_pad} is smaller than the polygon's {len(poly)} edges")
    e = np.roll(poly, -1, 0) - poly
    ln = np.linalg.norm(e, axis=1)
    if np.any(ln < 1e-12):
        raise ValueError("polygon has a zero-length edge")
    n = np.stack([e[:, 1], -e[:, 0]], 1) / ln[:, None]      # outward for CCW winding
    d = np.einsum("ij,ij->i", n, poly)
    if np.any((poly @ n.T) - d[None, :] > 1e-9):
        raise ValueError("polygon is not convex, or is wound clockwise")

    n_out = np.zeros((k_pad, 2), np.float64)
    d_out = np.full(k_pad, np.inf, np.float64)
    n_out[:len(poly)] = n
    d_out[:len(poly)] = d
    return n_out, d_out


def prop_halfplanes(prop, k_pad: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """`section_halfplanes` for a `props.Prop`'s declared footprint."""
    return section_halfplanes(prop.envelope.footprint, k_pad)


# ==================================================================== shared torch helpers
def _signed_eps(den: torch.Tensor) -> torch.Tensor:
    """`den` with anything near zero replaced by +-EPS, keeping its sign.

    Only for the vertex-reconstruction 2x2 solve, whose degenerate results are thrown away. It is
    **not** safe for ray clipping -- see `_fold`.
    """
    return torch.where(den.abs() < EPS, torch.where(den < 0, -EPS, EPS), den)


def _fold(t_in: torch.Tensor, t_out: torch.Tensor, den: torch.Tensor, num: torch.Tensor,
          reduce_dim: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clip the ray parameter by one or more half-planes `den * t <= num`.

    A ray parallel to a plane (`den == 0`) is handled as the algebra says, not by nudging the
    denominator. `den * t <= num` with `den == 0` reduces to `0 <= num`, which is either

    * satisfied -- the plane constrains nothing, whatever `t` is; or
    * violated -- the ray lies wholly outside, so no `t` can work.

    Substituting `+EPS` for a zero denominator gets the first case wrong at the boundary: with
    `num == 0` it yields `t = 0` and clamps `t_out` to zero, so a beam grazing exactly along a face
    or exactly along the top plane reports a miss. Those beams are ordinary geometry, not an edge
    case to be dodged with an epsilon shift, so the two branches are written out.
    """
    par = den.abs() < EPS
    t = num / torch.where(par, torch.ones_like(den), den)
    neg, pos = torch.full_like(t, float("-inf")), torch.full_like(t, float("inf"))
    lo = torch.where(~par & (den < 0), t, neg)                  # entering half-planes
    hi = torch.where(~par & (den > 0), t, pos)                  # leaving half-planes
    hi = torch.where(par & (num < 0), neg, hi)                  # parallel and outside: infeasible
    if reduce_dim is not None:
        lo = lo.amax(reduce_dim)
        hi = hi.amin(reduce_dim)
    return torch.maximum(t_in, lo), torch.minimum(t_out, hi)


def _slot_is_live(pn_c: torch.Tensor, z_lo_c: torch.Tensor, z_hi_c: torch.Tensor) -> torch.Tensor:
    """(B,) -- whether prop slot `c` is a real prop for this env.

    Both disable conventions in one place, stated rather than emergent: a flat or inverted z range,
    and a slot with too few real half-planes to bound anything.
    """
    valid = pn_c.pow(2).sum(-1) > 0.5                       # unit normals are 1, padding is 0
    return (valid.sum(-1) >= MIN_FACES) & (z_hi_c > z_lo_c)


def _to_world(pn_c: torch.Tensor, pd_c: torch.Tensor, pose_c: torch.Tensor):
    """Local half-planes -> world. `n_w = R(yaw) n`, `d_w = d + n_w . centre`."""
    yaw = pose_c[:, 2:3]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    nx, ny = pn_c[..., 0], pn_c[..., 1]
    nwx = nx * cy - ny * sy
    nwy = nx * sy + ny * cy
    nw = torch.stack([nwx, nwy], -1)
    dw = pd_c + nwx * pose_c[:, 0:1] + nwy * pose_c[:, 1:2]
    return nw, dw


def _polygon_vertices(nw: torch.Tensor, dw: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Recover the polygon's vertices (B,K,2) from its ordered half-planes.

    Half-planes `j` and `j+1` meet at vertex `j+1`, which is why `section_halfplanes` promises edge
    order and end-contiguous padding. SAT needs these because the car's own edge normals are
    arbitrary directions, and the prop's support along them is not recoverable from `(n, d)` alone.

    Padded slots are replaced with a finite stand-in *before* the 2x2 solve, so no inf or NaN is
    ever produced, and their results are then overwritten with a real vertex so that reducing over
    the full K is safe.
    """
    fill = torch.zeros_like(nw)
    fill[..., 0] = 1.0
    n1 = torch.where(valid.unsqueeze(-1), nw, fill)
    d1 = torch.where(valid, dw, torch.zeros_like(dw))

    n2, d2, v2 = n1.roll(-1, 1), d1.roll(-1, 1), valid.roll(-1, 1)
    first_n = n1[:, :1].expand_as(n2)
    first_d = d1[:, :1].expand_as(d2)
    n2 = torch.where(v2.unsqueeze(-1), n2, first_n)         # wrap past the padding to plane 0
    d2 = torch.where(v2, d2, first_d)

    det = _signed_eps(n1[..., 0] * n2[..., 1] - n1[..., 1] * n2[..., 0])
    vx = (d1 * n2[..., 1] - n1[..., 1] * d2) / det
    vy = (n1[..., 0] * d2 - d1 * n2[..., 0]) / det
    v = torch.stack([vx, vy], -1)
    return torch.where(valid.unsqueeze(-1), v, v[:, :1].expand_as(v))


# ==================================================================== 1. rays
def ray_prisms_hits(origin: torch.Tensor, dh: torch.Tensor, k: torch.Tensor, poses: torch.Tensor,
                    pn: torch.Tensor, pd: torch.Tensor, z_lo: torch.Tensor, z_hi: torch.Tensor,
                    *, inside: str = "hit"):
    """Beams against finite convex prisms.

    origin (B,N,3) world beam origins; dh (B,N,2) unit horizontal directions; k (B,N) dz/ds.
    poses (B,C,3) prop x, y, yaw; pn (B,C,K,2) outward unit normals in **prop-local** coordinates;
    pd (B,C,K) offsets, padding `n=0, d=+inf`; z_lo, z_hi (B,C) section bottom and top above the
    floor. Returns `(range (B,N), hit (B,N) bool)`, range being the 3D distance `t * sqrt(1 + k^2)`.

    The z extent is folded in as one more slab on the same `t`, not checked only at `t_in`, so a
    beam dropping in through the top face hits and a beam passing over the top misses.

    `inside="hit"` (default) returns range 0 when the origin is already inside a prism;
    `inside="miss"` reproduces `ray_box_hits`, where such a beam reports nothing.
    """
    if inside not in ("hit", "miss"):
        raise ValueError(f"inside must be 'hit' or 'miss', got {inside!r}")
    B, N = k.shape
    inf = float("inf")
    best = torch.full((B, N), inf, device=k.device, dtype=k.dtype)
    hit = torch.zeros(B, N, dtype=torch.bool, device=k.device)
    if poses.shape[1] == 0:                                 # static shape, not a data branch
        return best, hit

    slen = torch.sqrt(1.0 + k * k)
    oz = origin[..., 2]
    for c in range(poses.shape[1]):
        pose_c = poses[:, c]
        yaw = pose_c[:, 2:3]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        ox = origin[..., 0] - pose_c[:, 0:1]
        oy = origin[..., 1] - pose_c[:, 1:2]
        lx, ly = ox * cy + oy * sy, -ox * sy + oy * cy      # beam origin in the prop frame
        dx = dh[..., 0] * cy + dh[..., 1] * sy
        dy = -dh[..., 0] * sy + dh[..., 1] * cy

        n = pn[:, c]                                        # (B,K,2)
        nx, ny = n[..., 0].unsqueeze(1), n[..., 1].unsqueeze(1)      # (B,1,K)
        den = dx.unsqueeze(-1) * nx + dy.unsqueeze(-1) * ny          # (B,N,K)
        num = pd[:, c].unsqueeze(1) - (lx.unsqueeze(-1) * nx + ly.unsqueeze(-1) * ny)

        t_in = torch.full((B, N), -inf, device=k.device, dtype=k.dtype)
        t_out = torch.full((B, N), inf, device=k.device, dtype=k.dtype)
        t_in, t_out = _fold(t_in, t_out, den, num, reduce_dim=-1)
        # the z extent, as one more slab on the same parameter
        lo, hi = z_lo[:, c:c + 1], z_hi[:, c:c + 1]
        t_in, t_out = _fold(t_in, t_out, k, hi - oz)                 #  k*t <= z_hi - oz
        t_in, t_out = _fold(t_in, t_out, -k, oz - lo)                # -k*t <= oz - z_lo

        if inside == "hit":
            t_hit = t_in.clamp_min(0.0)
            ok = (t_out >= t_hit) & (t_out > 0.0)
        else:
            t_hit = t_in
            ok = (t_out >= t_in) & (t_in > 0.0)
        ok = ok & _slot_is_live(n, z_lo[:, c], z_hi[:, c]).unsqueeze(1)
        r = t_hit * slen
        closer = ok & (r < best)
        best = torch.where(closer, r, best)
        hit = hit | ok
    return best, hit


# ==================================================================== 2. contacts
def prism_contacts(corners: torch.Tensor, poses: torch.Tensor, pn: torch.Tensor,
                   pd: torch.Tensor, z_lo: torch.Tensor, z_hi: torch.Tensor, car_h):
    """The car rectangle against the prop sections, by SAT.

    corners (B,4,2) car footprint corners in world, consistently wound either way; the prop tensors
    as above; car_h a scalar or (B,) body height. Returns `(depth (B,), normal (B,2), idx (B,))`.
    `depth > 0` means overlap, `normal` is a unit vector pointing **prop -> car**, and `idx` names
    the prop. With no overlap: `depth = 0`, `normal = 0`, `idx = -1`.

    Depth is the **full translation needed to escape**, not the width of the projection overlap.
    Those differ exactly in the case that matters here: when the prop is entirely inside the car,
    the overlap width along an axis is the prop's own width, while escaping means pushing the car
    until the prop clears one of its edges. The tested guarantee is the useful one -- translating
    `corners` by `(depth + eps) * normal` removes the overlap.

    Both polygons' edge normals are candidate axes, so a slender post sitting wholly inside the car
    footprint is found. Corner sampling cannot see that case at all.
    """
    B = corners.shape[0]
    dtype, device = corners.dtype, corners.device
    depth = torch.full((B,), float("-inf"), device=device, dtype=dtype)
    normal = torch.zeros(B, 2, device=device, dtype=dtype)
    idx = torch.full((B,), -1, device=device, dtype=torch.long)
    if poses.shape[1] == 0:
        return torch.zeros(B, device=device, dtype=dtype), normal, idx

    h = torch.as_tensor(car_h, device=device, dtype=dtype).reshape(-1)
    h = h.expand(B) if h.numel() == 1 else h

    # the car's own outward edge normals; the winding is whatever the caller used
    nxt = corners.roll(-1, 1)
    e = nxt - corners
    area2 = (corners[..., 0] * nxt[..., 1] - nxt[..., 0] * corners[..., 1]).sum(1)
    wind = torch.where(area2 < 0, -torch.ones_like(area2), torch.ones_like(area2))[:, None]
    m = torch.stack([e[..., 1], -e[..., 0]], -1) * wind.unsqueeze(-1)
    m = m / m.norm(dim=-1, keepdim=True).clamp_min(EPS)                       # (B,4,2)
    e_off = torch.einsum("bij,bkj->bik", m, corners).amax(-1)                 # (B,4) car support

    for c in range(poses.shape[1]):
        nw, dw = _to_world(pn[:, c], pd[:, c], poses[:, c])                   # (B,K,2), (B,K)
        valid = pn[:, c].pow(2).sum(-1) > 0.5
        live = _slot_is_live(pn[:, c], z_lo[:, c], z_hi[:, c]) \
            & (z_lo[:, c] < h) & (z_hi[:, c] > 0)

        # axes from the prop's own facets: push the car out along +n
        car_proj = torch.einsum("bkj,bij->bki", nw, corners)                  # (B,K,4)
        d_prop = dw - car_proj.amin(-1)                                       # (B,K)
        d_prop = torch.where(valid, d_prop, torch.full_like(d_prop, float("inf")))

        # axes from the car's facets: push the car out along -m, so the prop clears that edge
        verts = _polygon_vertices(nw, dw, valid)                              # (B,K,2)
        prop_proj = torch.einsum("bij,bkj->bik", m, verts)                    # (B,4,K)
        big = torch.full_like(prop_proj, float("inf"))
        prop_min = torch.where(valid.unsqueeze(1), prop_proj, big).amin(-1)   # (B,4)
        d_car = e_off - prop_min                                              # (B,4)

        best_p, arg_p = d_prop.min(dim=1)
        best_c, arg_c = d_car.min(dim=1)
        use_prop = best_p <= best_c
        d_c = torch.where(use_prop, best_p, best_c)
        n_p = torch.gather(nw, 1, arg_p[:, None, None].expand(B, 1, 2)).squeeze(1)
        n_c = -torch.gather(m, 1, arg_c[:, None, None].expand(B, 1, 2)).squeeze(1)
        nrm_c = torch.where(use_prop.unsqueeze(-1), n_p, n_c)

        d_c = torch.where(live, d_c, torch.full_like(d_c, float("-inf")))
        take = d_c > depth
        depth = torch.where(take, d_c, depth)
        normal = torch.where(take.unsqueeze(-1), nrm_c, normal)
        idx = torch.where(take, torch.full_like(idx, c), idx)

    overlap = depth > 0
    depth = torch.where(overlap, depth, torch.zeros_like(depth))
    normal = torch.where(overlap.unsqueeze(-1), normal, torch.zeros_like(normal))
    idx = torch.where(overlap, idx, torch.full_like(idx, -1))
    return depth, normal, idx
