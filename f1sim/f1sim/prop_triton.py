"""Triton kernel for the LiDAR's prop sections: `prop_math.ray_prisms_hits(..., inside="hit")`.

`_prism_kernel` computes exactly what the eager `ray_prisms_hits` computes, operation for operation
in float32 -- the same products, sums and quotients in the same order, each rounded once. As in
`lidar_triton._car_mesh_kernel`, every rounded operation is an explicit IEEE intrinsic (`_mul`,
`_add`, `_sub` -- scalar NVVM intrinsics that neither LLVM's f32x2 packing nor ptxas's FMA
contraction touches; plain `a * b - c * d` was fused on sm_120 even with `enable_fp_fusion=False`),
the divisions and the square root are `div_rn` / `sqrt_rn`, libdevice's flush-to-zero is off,
cos/sin of the prop yaw come from torch, and every max/min that torch evaluates with NaN
propagation (`torch.maximum`, `torch.minimum`, `amax`, `amin`, `clamp_min`) is a NaN-propagating
one here too. One program per BLOCK beams; the prop slots are walked in a loop and each slot's
half-planes are unrolled.

Three things the reference computes and this kernel does not, each because it provably cannot
change an output bit:

* **A dead slot** (`prop_math._slot_is_live` false: `z_hi <= z_lo`, or fewer than `MIN_FACES` real
  normals) has its `ok` masked off in the reference, so nothing it computes reaches `best` or `hit`.
  The kernel evaluates liveness exactly as the reference does (`nx*nx + ny*ny > 0.5`, torch's
  `pow(2)` being `x*x` bit for bit) and a block skips a slot dead for all of its beams.
* **Trailing padding half-planes** (`n = (0, 0)` -- either zero sign -- and `d = +inf`). Whatever
  the beam, such a plane yields `lo = -inf` and `hi = +inf`: with finite `dx, dy` the dot product
  is a signed zero, so the plane is parallel, `num` is `inf` (or NaN when `lx`/`ly` is not finite)
  and `num < 0` is false; with a non-finite `dx`/`dy` the dot product is NaN, so it is neither
  parallel nor entering nor leaving. `-inf` and `+inf` are the identities of the NaN-propagating
  max and min, so a block folds only the half-planes up to the last real one of any of its beams'
  slots -- every one it leaves out is a padding plane for every beam.
* **The order of the reductions over half-planes.** The reference reduces with `amax`/`amin`; the
  kernel folds sequentially. Max and min are exact, and NaN-propagating either way, so the two
  agree except for which signed zero a tie of -0 and +0 returns. That sign never reaches an output:
  `t_in` is only read through `clamp_min(0)` (which returns +0 for both), and `t_out` only through
  `t_out >= t_hit` and `t_out > 0`, which do not see the sign of a zero.

There is no geometric early-out (skipping a slot whose prism the beam's line cannot reach): the
slab arithmetic's rounding near grazing faces would need a per-slot conditioning bound to prove
such a skip never changes a bit, and without it the kernel already runs in well under a
millisecond at training batch sizes (see `prism_hits_triton`).

Only `inside="hit"` -- the LiDAR's, `Lidar._merge_props` -- is implemented.
"""
import torch

from .lidar_triton import HAVE_TRITON
from .prop_math import EPS, MIN_FACES

if HAVE_TRITON:
    import triton
    import triton.language as tl

    from .lidar_triton import _add, _mul, _sub

    @triton.jit
    def _fold1(t_in, t_out, den, num, EPS_: tl.constexpr):
        """`prop_math._fold` for one half-plane `den * t <= num`, without the reduction."""
        par = tl.abs(den) < EPS_
        t = tl.div_rn(num, tl.where(par, 1.0, den))
        lo = tl.where((par == 0) & (den < 0.0), t, float("-inf"))
        hi = tl.where((par == 0) & (den > 0.0), t, float("inf"))
        hi = tl.where(par & (num < 0.0), float("-inf"), hi)
        return (tl.maximum(t_in, lo, propagate_nan=tl.PropagateNan.ALL),
                tl.minimum(t_out, hi, propagate_nan=tl.PropagateNan.ALL))

    @triton.jit
    def _prism_kernel(o_ptr, d_ptr, k_ptr, pose_ptr, cy_ptr, sy_ptr, pn_ptr, pd_ptr, zlo_ptr, zhi_ptr,
                      out_r_ptr, out_h_ptr, n_rays, n_beams, n_slots,
                      K: tl.constexpr, MINF: tl.constexpr, EPS_: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_rays
        ox = tl.load(o_ptr + offs * 3, mask=mask, other=0.0)
        oy = tl.load(o_ptr + offs * 3 + 1, mask=mask, other=0.0)
        oz = tl.load(o_ptr + offs * 3 + 2, mask=mask, other=0.0)
        hx = tl.load(d_ptr + offs * 2, mask=mask, other=1.0)
        hy = tl.load(d_ptr + offs * 2 + 1, mask=mask, other=0.0)
        k = tl.load(k_ptr + offs, mask=mask, other=0.0)
        env = offs // n_beams
        slen = tl.sqrt_rn(_add(1.0, _mul(k, k)))
        best = tl.full([BLOCK], float("inf"), tl.float32)
        hit = offs < 0                                                   # all false
        for c in range(n_slots):
            pc = env * n_slots + c
            zlo = tl.load(zlo_ptr + pc, mask=mask, other=0.0)
            zhi = tl.load(zhi_ptr + pc, mask=mask, other=0.0)
            zl = mask & (zhi > zlo)
            if tl.max(zl.to(tl.int32), axis=0) > 0:
                # liveness as `_slot_is_live`, and where each beam's slot stops being padding
                cnt = tl.zeros([BLOCK], dtype=tl.int32)
                kreal = tl.zeros([BLOCK], dtype=tl.int32)
                for j in tl.static_range(K):
                    nx = tl.load(pn_ptr + (pc * K + j) * 2, mask=mask, other=0.0)
                    ny = tl.load(pn_ptr + (pc * K + j) * 2 + 1, mask=mask, other=0.0)
                    dd = tl.load(pd_ptr + pc * K + j, mask=mask, other=float("inf"))
                    cnt += (_add(_mul(nx, nx), _mul(ny, ny)) > 0.5).to(tl.int32)
                    pad = (nx == 0.0) & (ny == 0.0) & (dd == float("inf"))
                    kreal = tl.where(pad, kreal, j + 1)
                live = zl & (cnt >= MINF)
                kblk = tl.max(tl.where(live, kreal, 0), axis=0)          # > 0 iff any beam's slot is live
                if kblk > 0:
                    px = tl.load(pose_ptr + pc * 3, mask=mask, other=0.0)
                    py = tl.load(pose_ptr + pc * 3 + 1, mask=mask, other=0.0)
                    cy = tl.load(cy_ptr + pc, mask=mask, other=1.0)
                    sy = tl.load(sy_ptr + pc, mask=mask, other=0.0)
                    rx = _sub(ox, px)
                    ry = _sub(oy, py)
                    lx = _add(_mul(rx, cy), _mul(ry, sy))                # beam origin in the prop frame
                    ly = _add(_mul(-rx, sy), _mul(ry, cy))
                    dx = _add(_mul(hx, cy), _mul(hy, sy))
                    dy = _add(_mul(-hx, sy), _mul(hy, cy))
                    # the faces: `_fold(t_in=-inf, t_out=+inf, den, num, reduce_dim=-1)`
                    lo_max = tl.full([BLOCK], float("-inf"), tl.float32)
                    hi_min = tl.full([BLOCK], float("inf"), tl.float32)
                    for j in tl.static_range(K):
                        if j < kblk:
                            nx = tl.load(pn_ptr + (pc * K + j) * 2, mask=mask, other=0.0)
                            ny = tl.load(pn_ptr + (pc * K + j) * 2 + 1, mask=mask, other=0.0)
                            dd = tl.load(pd_ptr + pc * K + j, mask=mask, other=float("inf"))
                            den = _add(_mul(dx, nx), _mul(dy, ny))
                            num = _sub(dd, _add(_mul(lx, nx), _mul(ly, ny)))
                            lo_max, hi_min = _fold1(lo_max, hi_min, den, num, EPS_)
                    t_in = tl.maximum(tl.full([BLOCK], float("-inf"), tl.float32), lo_max,
                                      propagate_nan=tl.PropagateNan.ALL)
                    t_out = tl.minimum(tl.full([BLOCK], float("inf"), tl.float32), hi_min,
                                       propagate_nan=tl.PropagateNan.ALL)
                    # the z extent: `_fold(.., k, hi - oz)` then `_fold(.., -k, oz - lo)`
                    t_in, t_out = _fold1(t_in, t_out, k, _sub(zhi, oz), EPS_)
                    t_in, t_out = _fold1(t_in, t_out, -k, _sub(oz, zlo), EPS_)
                    # inside == "hit"
                    t_hit = tl.maximum(t_in, 0.0, propagate_nan=tl.PropagateNan.ALL)
                    ok = (t_out >= t_hit) & (t_out > 0.0) & live
                    r = _mul(t_hit, slen)
                    closer = ok & (r < best)
                    best = tl.where(closer, r, best)
                    hit = hit | ok
        tl.store(out_r_ptr + offs, best, mask=mask)
        tl.store(out_h_ptr + offs, hit, mask=mask)


def prism_hits_triton(origin, dh, k, poses, pn, pd, z_lo, z_hi, *, inside: str = "hit",
                      block: int = 32, num_warps: int = 1):
    """`prop_math.ray_prisms_hits` on CUDA, bit for bit. origin (B,N,3), dh (B,N,2), k (B,N); poses
    (B,C,3) x, y, yaw; pn (B,C,K,2), pd (B,C,K) prop-local half-planes (padding n=0, d=+inf);
    z_lo, z_hi (B,C); all float32 -> (range (B,N), hit (B,N) bool).

    One warp of 32 beams per program was the fastest tile on beams captured from a race of 4 on
    real:korea_2026_competition with the training obstacle layout (18 slots, about 8 live, K 8),
    RTX 4070 SUPER: 0.084-0.091 ms at B=256 and 0.285-0.288 ms at B=1024, against 10.8-11.1 ms and
    68.3-69.0 ms for the eager function (CUDA-event time)."""
    if inside != "hit":
        raise ValueError(f"prism_hits_triton implements inside='hit' only (the LiDAR's), got {inside!r}")
    B, N = k.shape
    C, K = pn.shape[1], pn.shape[2]
    n = B * N
    if n == 0 or C == 0:
        return (torch.full((B, N), float("inf"), device=k.device, dtype=k.dtype),
                torch.zeros((B, N), device=k.device, dtype=torch.bool))
    if K == 0:
        raise ValueError("prop slots with no half-planes (K == 0): ray_prisms_hits cannot reduce them either")
    out_r = torch.empty((B, N), device=k.device, dtype=torch.float32)
    out_h = torch.empty((B, N), device=k.device, dtype=torch.bool)
    p = poses.contiguous()
    cy, sy = torch.cos(p[..., 2]).contiguous(), torch.sin(p[..., 2]).contiguous()
    with torch.cuda.device(k.device):                    # Triton launches on the current device
        _prism_kernel[(triton.cdiv(n, block),)](
            origin.contiguous(), dh.contiguous(), k.contiguous(), p, cy, sy, pn.contiguous(), pd.contiguous(),
            z_lo.contiguous(), z_hi.contiguous(), out_r, out_h, n, N, C,
            K=K, MINF=MIN_FACES, EPS_=EPS, BLOCK=block, num_warps=num_warps,
            enable_fp_fusion=False, enable_reflect_ftz=False)
    return out_r, out_h
